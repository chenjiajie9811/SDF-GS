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
from torch import Tensor
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Type, Literal, Tuple, Union, Dict, List, cast, Optional
from jaxtyping import Float
from collections import defaultdict

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

from nerfstudio.cameras.rays import RaySamples, RayBundle
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.encodings import NeRFEncoding, TriplaneEncoding, TensorVMEncoding, HashEncoding
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.fields.sdf_field import LearnedVariance
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import NeuSSampler
from nerfstudio.utils import colormaps

from nerfstudio.model_components.losses import L1Loss, MSELoss
from pytorch_msssim import SSIM
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer, SemanticRenderer

from sdfgs.sdfgs_samplers import ScaffoldGSNeusSampler

import tinycudann as tcnn


@dataclass 
class ScaffoldGSNeusFieldConfig:
    # neus related
    num_layers: int = 2
    """Number of layers for geometric network"""
    hidden_dim: int = 64
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 31
    """Dimension of geometric feature"""
    num_layers_color: int = 4
    """Number of layers for color network"""
    hidden_dim_color: int = 64
    """Number of hidden dimension of color network"""
    bias: float = 0.8
    """Sphere size of geometric initialization"""
    geometric_init: bool = True
    """Whether to use geometric initialization"""
    inside_outside: bool = True
    """Whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear layer"""
    beta_init: float = 0.3

class ScaffoldGSNeusField(nn.Module):
    def __init__(self, config:ScaffoldGSNeusFieldConfig):
        super().__init__()
        self.config = config
        self.encoding_dim = 32

        # we concat inputs position ourselves
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=6, min_freq_exp=0.0, max_freq_exp=5.0, include_input=False
        )

        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        # initialize geometric network
        self.initialize_geo_layers()

        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = LearnedVariance(init_val=self.config.beta_init)

        # color network
        dims = [self.config.hidden_dim_color for _ in range(self.config.num_layers_color)]
        # point, view_direction, normal, feature, embedding
        in_dim = (
            3
            + self.direction_encoding.get_out_dim()
            + 3
            + self.config.geo_feat_dim
        )
        dims = [in_dim] + dims + [3]
        self.num_layers_color = len(dims)

        for layer in range(0, self.num_layers_color - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "clin" + str(layer), lin)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._cos_anneal_ratio = 1.0



    def initialize_geo_layers(self) -> None:
        """
        Initialize layers for geometric network (sdf)
        """
        # MLP with geometric initialization
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]
        in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding_dim
        dims = [in_dim] + dims + [1 + self.config.geo_feat_dim]
        self.num_layers = len(dims)
        self.skip_in = [4]

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            if self.config.geometric_init:
                if layer == self.num_layers - 2:
                    if not self.config.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.config.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.config.bias)
                elif layer == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif layer in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "glin" + str(layer), lin)

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def forward_geonetwork(
            self, 
            inputs: Float[Tensor, "*batch 3"], 
            encoding: None
        ) -> Float[Tensor, "*batch geo_features+1"]:
        """forward the geonetwork"""
        
        
        # !!! The inputs should already be normalized to [0, 1]
        # !!! And then fed into the encoding network
        # the self.get_encoded_input must be called first
        inputs, encoded_inputs = self.get_encoded_input(inputs, encoding)

        # should the input of the positional encoding before or after normalization?
        pe = self.position_encoding(inputs)

        inputs = torch.cat((inputs, pe, encoded_inputs), dim=-1)

        # Pass through layers
        outputs = inputs

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(layer))

            if layer in self.skip_in:
                outputs = torch.cat([outputs, inputs], 1) / np.sqrt(2)

            outputs = lin(outputs)

            if layer < self.num_layers - 2:
                outputs = self.softplus(outputs)
        return outputs
    
    def get_encoded_input(self, x: Float[Tensor, "*batch 3"], encoding = None) -> Float[Tensor, "*batch D"]:
        # map range [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        encoded = encoding(x)
        return x, encoded

    def get_sdf(self, ray_samples: RaySamples, encoding = None) -> Float[Tensor, "num_samples ... 1"]:
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        hidden_output = self.forward_geonetwork(positions_flat, encoding).view(*ray_samples.frustums.shape, -1)
        sdf, _ = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
        return sdf
    
    def get_alpha(
        self,
        ray_samples: RaySamples,
        encoding: None,
        sdf: Optional[Float[Tensor, "num_samples ... 1"]] = None,
        gradients: Optional[Float[Tensor, "num_samples ... 1"]] = None,
    ) -> Float[Tensor, "num_samples ... 1"]:
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            inputs = ray_samples.frustums.get_start_positions()
            inputs.requires_grad_(True)
            with torch.enable_grad():
                hidden_output = self.forward_geonetwork(inputs, encoding)
                sdf, _ = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha

    def get_density(self, ray_samples: RaySamples):
        #@ Use the s_density function?
        raise NotImplementedError
    
    def get_colors(
        self,
        points: Float[Tensor, "*batch 3"],
        directions: Float[Tensor, "*batch 3"],
        normals: Float[Tensor, "*batch 3"],
        geo_features: Float[Tensor, "*batch geo_feat_dim"]
    ) -> Float[Tensor, "*batch 3"]:
        """compute colors"""
        d = self.direction_encoding(directions)

        hidden_input = torch.cat(
            [
                points,
                d,
                normals,
                geo_features.view(-1, self.config.geo_feat_dim),
            ],
            dim=-1,
        )

        for layer in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(layer))

            hidden_input = lin(hidden_input)

            if layer < self.num_layers_color - 2:
                hidden_input = self.relu(hidden_input)

        rgb = self.sigmoid(hidden_input)

        return rgb
        
    def get_outputs(
        self,
        ray_samples: RaySamples,
        encoding = None,
        return_alphas: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        """compute output of ray samples"""
        outputs = {}

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        inputs.requires_grad_(True)
        with torch.enable_grad():
            hidden_output = self.forward_geonetwork(inputs, encoding)
            sdf, geo_feature = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf, inputs=inputs, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature)

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMALS: normals,
                FieldHeadNames.GRADIENT: gradients,
            }
        )

        if return_alphas:
            alphas = self.get_alpha(ray_samples, encoding, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        return outputs
    
    def forward(
        self, 
        ray_samples: RaySamples, 
        encoding = None,
        return_alphas: bool = False
    ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
            return_alphas: Whether to return alpha values
        """
        field_outputs = self.get_outputs(ray_samples, encoding, return_alphas=return_alphas)
        return field_outputs




@dataclass
class ScaffoldGaussianNeuSModelConfig(ModelConfig):

    _target: Type = field(default_factory=lambda: ScaffoldGaussianNeuSModel)

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


    # hash encoding related
    num_levels: int = 16
    """Number of encoding levels"""
    max_res: int = 2048
    """Maximum resolution of the encoding"""
    base_res: int = 16
    """Base resolution of the encoding"""
    log2_hashmap_size: int = 19
    """Size of the hash map"""
    features_per_level: int = 2
    """Number of features per encoding level"""
    use_hash: bool = True
    """Whether to use hash encoding"""
    smoothstep: bool = True
    """Whether to use the smoothstep function"""

    neus_config: ScaffoldGSNeusFieldConfig = ScaffoldGSNeusFieldConfig()
    eikonal_loss_mult = 0.1

    # neus sampler
    num_samples: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 64
    """Number of importance samples"""
    num_up_sample_steps: int = 4
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    base_variance: float = 64
    """fixed base variance in NeuS sampler, the inv_s will be base * 2 ** iter during upsample"""
    perturb: bool = True
    """use to use perturb for the sampled points"""
    num_samples_outside = 32

    

    
class ScaffoldGaussianNeuSModel(Model):

    config: ScaffoldGaussianNeuSModelConfig

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
        # self._anchor_feat = torch.empty(0)
        
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

        # self.encoding_dim = 64
        # self.encoding = TriplaneEncoding(resolution=self.encoding_dim, num_components=64).cuda()

        self.encoding_dim = 32
        growth_factor = np.exp((np.log(self.config.max_res) - np.log(self.config.base_res)) / (self.config.num_levels - 1))
        # feature encoding
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid" if self.config.use_hash else "DenseGrid",
                "n_levels": self.config.num_levels,
                "n_features_per_level": self.config.features_per_level,
                "log2_hashmap_size": self.config.log2_hashmap_size,
                "base_resolution": self.config.base_res,
                "per_level_scale": growth_factor,
                "interpolation": "Smoothstep" if self.config.smoothstep else "Linear",
            },
        )

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

        self.aabb = self.scene_box.aabb.cuda()

        # Neus field
        self.neus = ScaffoldGSNeusField(self.config.neus_config)
        self.neus_sampler = ScaffoldGSNeusSampler(
            num_samples=self.config.num_samples,
            num_samples_importance=self.config.num_samples_importance,
            num_samples_outside=self.config.num_samples_outside,
            num_upsample_steps=self.config.num_up_sample_steps,
            base_variance=self.config.base_variance,
        )

        self.field_background = nn.Parameter(torch.ones(1), requires_grad=False)

        # Collider
        self.collider = AABBBoxCollider(self.scene_box, near_plane=0.05)

        self.renderer_rgb = RGBRenderer(background_color="white" if self.config.white_background else "black")
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normal = SemanticRenderer()

        self.rgb_loss = L1Loss()
        self.eikonal_loss = MSELoss()

        self.anneal_end = 50000

        self.gs_initialized = True




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
        # self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
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
            # "anchor_feat": [self._anchor_feat],
            "opacity": [self._opacity],
            "scaling": [self._scaling],
            "rotation": [self._rotation],

            "encoding_embedding" : self.encoding.parameters(),
            "mlp_opacity" : self.mlp_opacity.parameters(),
            "mlp_cov" : self.mlp_cov.parameters(),
            "mlp_color" : self.mlp_color.parameters(),
        }
        if self.use_feat_bank:
            params.update({"mlp_featurebank" : self.mlp_feature_bank.parameters()})

        if not self.gs_initialized:
            return params
        else:
            params.update(
                {
                    "neus" : self.neus.parameters(),
                    "neus_background" : [self.field_background]
                }
            )
            return params
            
        
   
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        # for i in range(self._anchor_feat.shape[1]):
        #     l.append('f_anchor_feat_{}'.format(i))
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
        # anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
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
        # anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        # anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        # anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        # for idx, attr_name in enumerate(anchor_feat_names):
        #     anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        # self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

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
                'embedding' in group or \
                'neus' in group:
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
                'embedding' in group or\
                'neus' in group:
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
        # self._anchor_feat = optimizable_tensors["anchor_feat"]
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

                # new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                # new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    # "anchor_feat": new_feat,
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
                # self._anchor_feat = optimizable_tensors["anchor_feat"]
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
        
        if self.anneal_end > 0:
            def set_anneal(step):
                anneal = min([1.0, step / self.anneal_end])
                self.neus.set_cos_anneal_ratio(anneal)

            cbs.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
        
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
        # cbs.append(
        #     TrainingCallback(
        #         [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
        #         iters=self.config.upsampling_iters,
        #         func=self.reinit_optimizer_after_upsample,
        #         args=[training_callback_attributes],
        #     )
        # )
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

        
    def encod_input(self, x):
        x = SceneBox.get_normalized_positions(x, self.aabb)
        return self.encoding(x)
    
    def sample_and_forward_field(self, ray_bundle: RayBundle, encoding=None) -> Dict:
        ray_samples : RaySamples = self.neus_sampler(ray_bundle, sdf_fn=self.neus.get_sdf, encoding_fn=encoding)
        field_outputs = self.neus.forward(ray_samples, encoding, return_alphas=True)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )
        bg_transmittance = transmittance[:, -1, :]

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
        }
        return samples_and_field_outputs

    def get_outputs_gs(self, camera: Cameras)-> Dict[str, Union[torch.Tensor, List]]:
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        viewpoint_cam = parse_camera_info(camera, self.device)
        self.voxel_visible_mask = self.renderer.prefilter_voxel(viewpoint_cam, self, self.background)
        render_pkg = self.renderer.render(viewpoint_cam, self, self.background, self.voxel_visible_mask, True)
        image, self.viewspace_point_tensor, self.visibility_filter, self.offset_selection_mask, radii, scaling, self.temp_opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        return {'rgb_gs' : image.permute(1, 2, 0).clamp(0, 1)} # (H, W, 3)}
    
    def get_outputs_neus(self, ray_bundle: RayBundle)-> Dict[str, Union[torch.Tensor, List]]:
        assert (
            ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata
        ), "directions_norm is required in ray_bundle.metadata"

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle, encoding=self.encoding.forward)

        # shortcuts
        field_outputs: Dict[FieldHeadNames, torch.Tensor] = cast(
            Dict[FieldHeadNames, torch.Tensor], samples_and_field_outputs["field_outputs"]
        )
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]
        bg_transmittance = samples_and_field_outputs["bg_transmittance"]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.metadata["directions_norm"]

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb_neus": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            # used to scale z_vals for free space and sdf loss
            "directions_norm": ray_bundle.metadata["directions_norm"],
        }

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})
            outputs.update(samples_and_field_outputs)

        return outputs

    
    def get_outputs(self, camera: Cameras, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:

        outputs_gs = self.get_outputs_gs(camera)
        outputs_neus = self.get_outputs_neus(ray_bundle)

        return {**outputs_gs, **outputs_neus}

    def forward(self, camera: Cameras, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        return self.get_outputs(camera, ray_bundle)
    
    def get_metrics_dict_gs(self, outputs, batch)-> Dict[str, torch.Tensor]:
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        predicted_rgb = outputs["rgb_gs"]

        metrics_dict["psnr_gs"] = self.psnr(predicted_rgb, gt_rgb)
        metrics_dict["anchor_count"] = self._anchor.shape[0]

        return metrics_dict
    
    def get_metrics_dict_neus(self, outputs, batch)-> Dict[str, torch.Tensor]:
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        metrics_dict["psnr_neus"] = self.psnr(outputs["rgb_neus"], gt_rgb)
        return metrics_dict
    
    
    def get_metrics_dict(self, outputs, batch_gs, batch_neus) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        
        metrics_dict = {}
        metrics_dict.update(**self.get_metrics_dict_gs(outputs, batch_gs))
        metrics_dict.update(**self.get_metrics_dict_neus(outputs, batch_neus))
        
        return metrics_dict
    
    def get_loss_dict_gs(self, outputs, batch) -> Dict[str, torch.Tensor]:
        gt_img = batch["image"].to(self.device)
        Ll1 = torch.abs(gt_img - outputs["rgb_gs"]).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], outputs["rgb_gs"].permute(2, 0, 1)[None, ...])
        
        return {
            "loss_gs": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
        }
    
    def get_loss_dict_neus(self, outputs, batch) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_neus"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        loss_dict["rgb_loss_neus"] = self.rgb_loss(image, pred_image)
        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult

        return loss_dict


    def get_loss_dict(self, outputs, batch_gs, batch_neus, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        
        loss_dict = {}
        loss_dict.update(self.get_loss_dict_gs(outputs, batch_gs))
        loss_dict.update(self.get_loss_dict_neus(outputs, batch_neus))

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box= None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        outputs_neus =  self.get_outputs_for_camera_ray_bundle(
            camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
        )
        
        outputs_gs = self.get_outputs_gs(camera.to(self.device))
        return {**outputs_neus, **outputs_gs} # type: ignore
    
    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.get_outputs_neus(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs
    
    def get_image_metrics_and_images(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor],
        with_neus: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        
        gt_img = batch["image"]
        gt_rgb = gt_img.to(self.device)

        predicted_rgb_gs = outputs["rgb_gs"]

        combined_rgb_gs = torch.cat([gt_rgb, predicted_rgb_gs], dim=1)
        
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb_m = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb_gs = torch.moveaxis(predicted_rgb_gs, -1, 0)[None, ...]
        
        psnr_gs = self.psnr(gt_rgb_m, predicted_rgb_gs)
        ssim_gs = self.ssim(gt_rgb_m, predicted_rgb_gs)
        lpips_gs = self.lpips(gt_rgb_m, predicted_rgb_gs)

        metrics_dict = {
            "psnr_gs": float(psnr_gs.item()), 
            "ssim_gs": float(ssim_gs.item()),
            "lpips_gs": float(lpips_gs.item()),
            }
        
        images_dict = {"img_gs": combined_rgb_gs,}
    
        if with_neus:
            predicted_rgb_neus = outputs["rgb_neus"]
            combined_rgb_neus = torch.cat([gt_rgb, predicted_rgb_neus], dim=1)

            predicted_rgb_neus = torch.moveaxis(predicted_rgb_neus, -1, 0)[None, ...]
            psnr_neus = self.psnr(gt_rgb_m, predicted_rgb_neus)
            ssim_neus = self.ssim(gt_rgb_m, predicted_rgb_neus)
            lpips_neus = self.lpips(gt_rgb_m, predicted_rgb_neus)


            metrics_dict["psnr_neus"] = float(psnr_neus.item())
            metrics_dict["ssim_neus"] = float(ssim_neus.item())
            metrics_dict["lpips_neus"] = float(lpips_neus.item())

        
            acc = colormaps.apply_colormap(outputs["accumulation"])
            combined_acc = torch.cat([acc], dim=1)

            depth = colormaps.apply_depth_colormap(
                    outputs["depth"],
                    accumulation=outputs["accumulation"],
                )
            combined_depth = torch.cat([depth], dim=1)

            normal = outputs["normal"]
            normal = (normal + 1.0) / 2.0
            combined_normal = torch.cat([normal], dim=1)

            images_dict_neus = {
                "img_neus": combined_rgb_neus,
                "accumulation": combined_acc,
                "depth": combined_depth,
                "normal": combined_normal,}
            
            images_dict.update(**images_dict_neus)
            
        

        return metrics_dict, images_dict
    
def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device='cuda').split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device='cuda').split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device='cuda').split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    import mcubes
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max#.detach().cpu().numpy()
    b_min_np = bound_min#.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles




