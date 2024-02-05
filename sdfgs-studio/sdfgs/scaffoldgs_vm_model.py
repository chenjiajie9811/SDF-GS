#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max

from dataclasses import dataclass, field
from typing import Type, Literal, Tuple, Union, Dict, List, cast

from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
# from scene.embedding import Embedding

from sdfgs.gs_renderer import GaussianRenderer, parse_camera_info

from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.encodings import TriplaneEncoding, TensorVMEncoding

from nerfstudio.model_components.losses import L1Loss, MSELoss
from pytorch_msssim import SSIM
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@dataclass
class ScaffoldGaussianModelConfig(ModelConfig):

    _target: Type = field(default_factory=lambda: ScaffoldGaussianModel)

    sh_degree: int = 1
    """maximum degree of spherical harmonics to use"""

    white_background: bool = True

    ssim_lambda: float = 0.2

    use_scale_regularization: bool = True
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    feat_dim: int = 32
    n_offsets: int=5
    voxel_size: float=0.001
    update_depth: int=3
    update_init_factor: int=100
    update_hierachy_factor: int=4
    use_feat_bank : bool = False
    # appearance_dim : int = 32,
    ratio : int = 1
    add_opacity_dist : bool = False
    add_cov_dist : bool = False
    add_color_dist : bool = False

    compute_cov3D_python : bool = False
    convert_SHs_python : bool = False
    scaling_modifier : float = 1.0
    override_color = None

    percent_dense = 0.01

    # densification
    start_stat = 500
    update_from = 1500 
    update_until = 15_000
    check_interval :int = 10
    success_threshold : float = 0.8
    grad_threshold : float = 0.0002
    min_opacity : float = 0.005

    # triplane related
    init_resolution: int = 128
    """initial render resolution"""
    final_resolution: int = 300
    """final render resolution"""
    upsampling_iters: Tuple[int, ...] = (100, 2000, 4000, 8000, 10000, 15000)
    
class ScaffoldGaussianModel(Model):

    config: ScaffoldGaussianModelConfig

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def populate_modules(self):
        self.feat_dim = self.config.feat_dim
        self.n_offsets = self.config.n_offsets
        self.voxel_size = self.config.voxel_size
        self.update_depth = self.config.update_depth
        self.update_init_factor = self.config.update_init_factor
        self.update_hierachy_factor = self.config.update_hierachy_factor
        self.use_feat_bank = self.config.use_feat_bank

        # self.appearance_dim = appearance_dim
        # self.embedding_appearance = None
        self.ratio = self.config.ratio
        self.add_opacity_dist = self.config.add_opacity_dist
        self.add_cov_dist = self.config.add_cov_dist
        self.add_color_dist = self.config.add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        # self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        bg_color = [1, 1, 1] if self.config.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.encoding_dim = 64
        self.encoding = TriplaneEncoding(resolution=self.encoding_dim, num_components=64).cuda()

        self.upsampling_steps = (
            np.round(
                np.exp(
                    np.linspace(
                        np.log(self.config.init_resolution),
                        np.log(self.config.final_resolution),
                        len(self.config.upsampling_iters) + 1,
                    )
                )
            )
            .astype("int")
            .tolist()[1:]
        )
        
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.encoding_dim+3+self.opacity_dist_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = self.add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.encoding_dim+3+self.cov_dist_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(self.encoding_dim+3+self.color_dist_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        # self.crop_box: Optional[OrientedBox] = None
        self.back_color = torch.zeros(3)

        # GS renderer with cuda but not gsplat
        self.renderer = GaussianRenderer(
            self.config.compute_cov3D_python,
            self.config.convert_SHs_python,
            self.config.scaling_modifier,
            self.config.override_color)
        
        # Buffer parameter for training callbacks
        self.voxel_visible_mask = torch.empty(0)
        self.viewspace_point_tensor = torch.empty(0)
        self.visibility_filter = torch.empty(0)
        self.offset_selection_mask = torch.empty(0)
        self.temp_opacity = torch.empty(0)

    def step_cb(self, step):
        self.step = step

    # def eval(self):
    #     self.mlp_opacity.eval()
    #     self.mlp_cov.eval()
    #     self.mlp_color.eval()
    #     # if self.appearance_dim > 0:
    #     #     self.embedding_appearance.eval()
    #     if self.use_feat_bank:
    #         self.mlp_feature_bank.eval()

    # def train(self):
    #     self.mlp_opacity.train()
    #     self.mlp_cov.train()
    #     self.mlp_color.train()
    #     # if self.appearance_dim > 0:
    #     #     self.embedding_appearance.train()
    #     if self.use_feat_bank:                   
    #         self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            # self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)

    # def set_appearance(self, num_cameras):
    #     if self.appearance_dim > 0:
    #         self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    # @property
    # def get_appearance(self):
    #     return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self):
        self.percent_dense = self.config.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        params = {
            "anchor" : [self._anchor],
            "offset": [self._offset],
            "anchor_feat": [self._anchor_feat],
            "opacity": [self._opacity],
            "scaling": [self._scaling],
            "rotation": [self._rotation],

            "encoding_embedding" : self.encoding.parameters(),
            "mlp_opacity" : self.mlp_opacity.parameters(),
            "mlp_cov" : self.mlp_cov.parameters(),
            "mlp_color" : self.mlp_color.parameters()
        }
        if self.use_feat_bank:
            params.update({"mlp_featurebank" : self.mlp_feature_bank.parameters()})

        return params
            
        
   
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, optimizers, tensor, name):
        optimizable_tensors = {}
        param_groups = self.get_param_groups()
        for group, param in param_groups.items():
            if group == name:
                optimizer = optimizers.optimizers[group]
                stored_state = optimizer.state.get(param[0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optimizer.state[param[0]]
                param[0] = nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[param[0]] = stored_state

                optimizable_tensors[group] = param[0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, optimizers, tensors_dict):
        optimizable_tensors = {}
        param_groups = self.get_param_groups()
        for group, param in param_groups.items():
            if  'mlp' in group or \
                'conv' in group or \
                'feat_base' in group or \
                'embedding' in group:
                continue
            assert len(param) == 1
            extension_tensor = tensors_dict[group]

            optimizer = optimizers.optimizers[group]
            old_param = optimizer.param_groups[0]["params"][0]
            stored_state = optimizer.state[old_param]
            # print (group, " ", stored_state.keys())
            
            if len(stored_state.keys()) > 0:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del optimizer.state[old_param]
                param[0] = nn.Parameter(torch.cat((param[0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[param[0]] = stored_state
                optimizer.param_groups[0]["params"] = param
                del old_param

                optimizable_tensors[group] = param[0]
            else:
                param[0] = nn.Parameter(torch.cat((param[0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group] = param[0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        

        
    def _prune_anchor_optimizer(self, optimizers, mask):
        optimizable_tensors = {}
        param_groups = self.get_param_groups()
        for group, param in param_groups.items():
            if  'mlp' in group or \
                'conv' in group or \
                'feat_base' in group or \
                'embedding' in group:
                continue
            optimizer = optimizers.optimizers[group]

            old_param = optimizer.param_groups[0]["params"][0]

            stored_state = optimizer.state[old_param]
            if len(stored_state.keys()) > 0:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[old_param]

                del optimizer.param_groups[0]["params"][0]
                del optimizer.param_groups[0]["params"]
                param[0] = nn.Parameter((param[0][mask].requires_grad_(True)))
                optimizer.param_groups[0]["params"] = param
                optimizer.state[param[0]] = stored_state
                if group == "scaling":
                    scales = param[0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    param[0][:,3:] = temp
                optimizable_tensors[group] = param[0]
            else:
                param[0] = nn.Parameter(param[0][mask].requires_grad_(True))
                if group == "scaling":
                    scales = param[0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    param[0][:,3:] = temp
                optimizable_tensors[group] = param[0]
            
            
        return optimizable_tensors

    def prune_anchor(self, optimizers, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(optimizers, valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, optimizers, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(optimizers, d)

                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                


    def adjust_anchor(self, optimizers, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(optimizers, grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(optimizers, prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            # if self.appearance_dim:
            #     self.embedding_appearance.eval()
            #     emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
            #     emd.save(os.path.join(path, 'embedding_appearance.pt'))
            #     self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    # 'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    # 'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            # if self.appearance_dim > 0:
            #     self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            # if self.appearance_dim > 0:
            #     self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
        
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.check_interval,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=self.config.upsampling_iters,
                func=self.reinit_optimizer_after_upsample,
                args=[training_callback_attributes],
            )
        )
        return cbs
    
    def reinit_optimizer_after_upsample(self, training_callback_attributes: TrainingCallbackAttributes, step: int):
        assert training_callback_attributes.optimizers is not None
        assert training_callback_attributes.pipeline is not None
        index = self.config.upsampling_iters.index(step)
        resolution = self.upsampling_steps[index]

        # upsample the position and direction grids
        self.encoding.upsample_grid(resolution)

        # reinitialize the encodings optimizer
        optimizers_config = training_callback_attributes.optimizers.config
        enc = training_callback_attributes.pipeline.get_param_groups()["encoding_embedding"]
        lr_init = optimizers_config["encoding_embedding"]["optimizer"].lr

        training_callback_attributes.optimizers.optimizers["encoding_embedding"] = optimizers_config["encoding_embedding"][
            "optimizer"
        ].setup(params=enc)
        if optimizers_config["encoding_embedding"]["scheduler"]:
            training_callback_attributes.optimizers.schedulers["encoding_embedding"] = (
                optimizers_config["encoding_embedding"]["scheduler"]
                .setup()
                .get_scheduler(
                    optimizer=training_callback_attributes.optimizers.optimizers["encoding_embedding"], lr_init=lr_init
                )
            )
    
    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step < self.config.start_stat or self.step >= self.config.update_until:
            return
        # update opacity stats
        temp_opacity = self.temp_opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[self.voxel_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[self.voxel_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = self.voxel_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = self.offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = self.visibility_filter
        
        grad_norm = torch.norm(self.viewspace_point_tensor.grad[self.visibility_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def refinement_after(self, optimizers, step):
        assert step == self.step
        if self.step <= self.config.update_from:
            return
        # for (k, v) in optimizers.optimizers.items():
        #     print (k, " ", isinstance(k, torch.Tensor), " ", id(k))
        with torch.no_grad():
            self.adjust_anchor(optimizers, self.config.check_interval, self.config.success_threshold, self.config.grad_threshold, self.config.min_opacity)

        
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        viewpoint_cam = parse_camera_info(camera, self.device)
        self.voxel_visible_mask = self.renderer.prefilter_voxel(viewpoint_cam, self, self.background)
        render_pkg = self.renderer.render(viewpoint_cam, self, self.background, self.voxel_visible_mask, True)
        image, self.viewspace_point_tensor, self.visibility_filter, self.offset_selection_mask, radii, scaling, self.temp_opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        return {
            'rgb' : image.permute(1, 2, 0).clamp(0, 1) # (H, W, 3)
            }
    
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        
        gt_img = batch["image"]
        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["anchor_count"] = self._anchor.shape[0]
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        
        gt_img = batch["image"]
        Ll1 = torch.abs(gt_img - outputs["rgb"]).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], outputs["rgb"].permute(2, 0, 1)[None, ...])
        

        return {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box= None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        
        gt_img = batch["image"]
        predicted_rgb = outputs["rgb"]

        gt_rgb = gt_img.to(self.device)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations

        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict




