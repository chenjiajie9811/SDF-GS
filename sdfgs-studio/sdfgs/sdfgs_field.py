"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type

from torch import Tensor
from jaxtyping import Float
from sdfgs.embedder import *
import numpy as np

import trimesh

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field, FieldConfig  # for custom Field
from nerfstudio.fields.sdf_field import LearnedVariance
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes

from hashencoder.hashgrid import _hash_encode, HashEncoder

from utils.system_utils import mkdir_p
from utils.sh_utils import eval_sh, SH2RGB, RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.rigid_utils import matrix_to_quaternion, exp_so3


class ImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size = 16, #16,
            end_size = 2048,#2048,
            logmap = 19,#19,
            num_levels=16,#16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            use_grid_feature = True
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim
        
        print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim, 
                    per_level_scale=2, base_resolution=base_size, 
                    log2_hashmap_size=logmap, desired_resolution=end_size)
        
        '''
        # can also use tcnn for multi-res grid as it now supports eikonal loss
        base_size = 16
        hash = True
        smoothstep = True
        self.encoding = tcnn.Encoding(3, {
                        "otype": "HashGrid" if hash else "DenseGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": base_size,
                        "per_level_scale": 1.34,
                        "interpolation": "Smoothstep" if smoothstep else "Linear"
                    })
        '''
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
            print ("input_ch", input_ch)
        print("network architecture")
        print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

    def get_encoding(self, input):
        feature = self.encoding(input / self.divide_factor)
        embed = self.embed_fn(input)
        encoded = torch.cat((embed, feature), dim=-1)
        return encoded

    def forward(self, input):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            feature = self.encoding(input / self.divide_factor)
        else:
            feature = torch.zeros_like(input[:, :1].repeat(1, self.grid_feature_dim))
                    
        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.forward(x)[:,:1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            return gradients

    def get_outputs(self, x):
        
        x.requires_grad_(True)
        with torch.enable_grad():
            output = self.forward(x)
            sdf = output[:,:1]

            feature_vectors = output[:, 1:]
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        return sdf

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self):
        print("grid parameters", len(list(self.encoding.parameters())))
        for p in self.encoding.parameters():
            print(p.shape)
        return self.encoding.parameters()

class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        self.scaling_lin = nn.Linear(d_hidden, 3)
        self.rotation_angle_lin = nn.Linear(d_hidden, 1)


    def forward(self, points, normals, view_dirs, feature_vectors, pred_gauss_params=False):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

            if l == self.num_layers - 2:
                h = x

        if self.squeeze_out:
            x = torch.sigmoid(x)

        ret = {'color' : x}

        if pred_gauss_params:
            scale = torch.sigmoid(self.scaling_lin(h))
            rotation_theta = torch.sigmoid(self.rotation_angle_lin(h)) * 2 * torch.pi

            ret['scale'] = scale
            ret['rotation_angle'] = rotation_theta 

        
        return ret

@dataclass
class SDFGSFieldConfig(FieldConfig):
    _target: Type = field(default_factory=lambda: SDFGSField)
    num_layers: int = 2
    """Number of layers for geometric network"""
    hidden_dim: int = 64
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 15
    """Dimension of geometric feature"""
    num_layers_color: int = 2
    """Number of layers for color network"""
    hidden_dim_color: int = 64
    """Number of hidden dimension of color network"""
    appearance_embedding_dim: int = 32
    """Dimension of appearance embedding"""
    use_appearance_embedding: bool = False
    """Whether to use appearance embedding"""
    bias: float = 0.5
    """Sphere size of geometric initialization"""
    geometric_init: bool = True
    """Whether to use geometric initialization"""
    inside_outside: bool = True
    """Whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear layer"""
    use_grid_feature: bool = True
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.3
    """Init learnable beta value for transformation of sdf to density"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "hash"
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

class SDFGSField(Field):

    def __init__(
        self,
        config: SDFGSFieldConfig
    ) -> None:
        super().__init__()
        self.config = config

        self._cos_anneal_ratio = 1.0

        self.sdf_network = ImplicitNetworkGrid(
            feature_vector_size=config.geo_feat_dim,
            sdf_bounding_sphere=True,
            d_in=3,
            d_out=1,
            dims=[config.hidden_dim] * config.num_layers,
            geometric_init=config.geometric_init,
            bias=config.bias,
            multires=6
        )

        self.deviation_network = LearnedVariance(init_val=config.beta_init)

        self.shared_color_network = RenderingNetwork(
            d_feature=config.geo_feat_dim,
            d_in=9, #(point, normal, viewdir)
            d_out=3,
            d_hidden=self.config.hidden_dim_color,
            n_layers=self.config.num_layers_color,
            multires_view=4
        )

    def get_sdf(self, ray_samples: RaySamples) -> Float[Tensor, "num_samples ... 1"]:
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        hidden_output = self.sdf_network(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, _ = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
        return sdf

    def set_cos_anneal_ratio(self, value):
        self._cos_anneal_ratio = value

    def get_alpha(
        self,
        ray_samples: RaySamples,
        sdf: Optional[Float[Tensor, "num_samples ... 1"]] = None,
        gradients: Optional[Float[Tensor, "num_samples ... 1"]] = None,
    ) -> Float[Tensor, "num_samples ... 1"]:
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            inputs = ray_samples.frustums.get_start_positions()
            sdf, _, gradients = self.sdf_network.get_outputs(inputs)

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

    def get_outputs(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[Tensor] = None,
        return_alphas: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        sdf, geo_feature, gradients = self.sdf_network.get_outputs(inputs)

        rgb = self.shared_color_network(inputs, gradients, directions_flat, geo_feature)['color']

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
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        return outputs

    def forward_geonetwork(self, inputs: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch geo_features+1"]:
        """forward the geonetwork"""
        return self.sdf_network.forward(inputs)

    def forward(
        self, ray_samples: RaySamples, compute_normals: bool = False, return_alphas: bool = False
    ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
            compute normals: not currently used in this implementation.
            return_alphas: Whether to return alpha values
        """
        field_outputs = self.get_outputs(ray_samples, return_alphas=return_alphas)
        return field_outputs


@dataclass 
class MeshGaussiansFieldConfig(SDFGSFieldConfig):
    _target: Type = field(default_factory=lambda: MeshGaussiansField)


class GaussianGeoNetwork(nn.Module):
    def __init__(self, D=64, num_geo_feat=16, num_sdf_feat=16, multires_geo=6, device='cuda'):
        super(GaussianGeoNetwork, self).__init__()

        self.device = device
        self.view_embed_fn, geo_input_ch = get_embedder(multires_geo, 3)
        self.input_ch = (geo_input_ch + num_sdf_feat) + 3 # (3 embedded vertices + 3 normals)

        self.linear = nn.ModuleList(
            [
                nn.Linear(self.input_ch, D),
                nn.Linear(D, D),
            ]
        )

        self.bary_coords_lin = nn.Linear(D, 3)
        self.opacity_lin = nn.Linear(D, 1)
        self.scaling_lin = nn.Linear(D, 3)
        self.rotation_angle_lin = nn.Linear(D, 1)
        self.geo_feat_lin = nn.Linear(D, num_geo_feat)


class MeshGaussiansField(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        

    def __init__(
        self, sh_degree : int, 
        config: SDFGSFieldConfig,
        sdf_network : ImplicitNetworkGrid,
        shared_color_network : RenderingNetwork,
        device='cuda'
    ):
        super(MeshGaussiansField, self).__init__()

        self.config = config
        
        # Gaussian parameters
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.device = device
        self.setup_functions()

        # Networks
        self.geo_network = ImplicitNetworkGrid(
            feature_vector_size=config.geo_feat_dim,
            sdf_bounding_sphere=True,
            d_in=3,
            d_out=1,
            dims=[config.hidden_dim] * config.num_layers,
            geometric_init=config.geometric_init,
            bias=config.bias,
            multires=6
        )

        self.sdf_network = sdf_network

        # self.deviation_network = LearnedVariance(init_val=config.beta_init)
        self.shared_color_network = shared_color_network

        # Mesh related
        self.vertices = torch.empty(0)
        self.faces = torch.empty(0)
        self.center_normal = torch.empty(0)

        # From SuGar
        self.surface_triangle_circle_radius = 1. / 2. / np.sqrt(3.)
        self.bary_coords = torch.tensor([[1/3, 1/3, 1/3]], dtype=torch.float32, device=self.device)[..., None]

        # Initialize the Gaussians from sdf network
        init_mesh = self.marching_cubes(sdf_network, resolution=128)
        self.update_mesh_info(init_mesh)
    
    @torch.no_grad()
    def marching_cubes(self, sdf_network : ImplicitNetworkGrid, resolution=128):
        mesh : trimesh.Trimesh = generate_mesh_with_multires_marching_cubes(
                                    sdf_network, 
                                    resolution, 
                                    bounding_box_min=(-1.5, -1.5, -1.5), 
                                    bounding_box_max=(1.5, 1.5, 1.5), 
                                    isosurface_threshold=0)
        
        return mesh
    
    def update_mesh_info(self, mesh):
        self.update_gaussians_center_info(mesh.vertices, mesh.faces)

    def update_gaussians_center_info(self, vertices, faces):
        self.vertices = torch.tensor(vertices, device=self.device)
        self.faces = torch.tensor(faces, device=self.device)

        self._xyz = 1. / 3. * (self.vertices[:, self.faces[:, 0].flatten()] + \
                                self.vertices[:, self.faces[:, 1].flatten()] + \
                                self.vertices[:, self.faces[:, 2].flatten()])      

        vec1 = self.vertices[:, self.faces[:, 0].flatten()] - self.vertices[:, self.faces[:, 1].flatten()] 
        vec2 = self.vertices[:, self.faces[:, 0].flatten()] - self.vertices[:, self.faces[:, 2].flatten()] 
        cross_pro = torch.cross(vec1, vec2, dim=-1)
        self.center_normal = F.normalize(cross_pro)

           

    def update_gaussians(self, camera_center : Tensor, sdf_network : ImplicitNetworkGrid):
        """
            1. Synchronize the mesh by moving the vertices towards the zero level set 
            according to the estimation of the current sdf network
            2. Update the gaussian xyz and normal according to the updated mesh
            3. Go through the geo_network for estimating the gaussian opacity and output a geo feature vector
            4. Go through the shared rendering network for estimating the shared color, the gaussian scales, and rotation angle
            5. Set up all the up-to-date Gaussians parameters
        """
        #1.
        sdf, _, gradients = sdf_network.get_outputs(self.vertices)
        self.vertices = self.vertices - sdf * F.normalize(gradients, dim=-1)

        #2.
        self.update_gaussians_center_info(self.vertices, self.faces)

        #3.
        geo_output = self.geo_network.forward(self._xyz)
        opacity, geo_feat_vecs = torch.split(geo_output, [1, self.config.geo_feat_dim], dim=-1)

        #4. 
        view_dir = F.normalize(self._xyz - camera_center, dim=-1)
        color_output = self.shared_color_network.forward(self._xyz, self.center_normal, view_dir, geo_feat_vecs, pred_gauss_params=True)

        #5.
        features = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = RGB2SH(color_output['color'].squeeze(0))
        features[:, 3:, 1:] = 0.0
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        self._features_dc.retain_grad()
        self._features_rest.retain_grad()

        self._scaling = self.scaling_inverse_activation(color_output['scale'])
 
        rotation_matrix = exp_so3(self.center_normal.squeeze(0), color_output['rotation_angle'].squeeze(0))
        self._rotation = matrix_to_quaternion(rotation_matrix)
 
        self._opacity = self.inverse_opacity_activation(opacity)



    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.percent_dense = training_args['percent_dense']
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {'params': [self._xyz], 'lr': training_args['position_lr_init'] * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args['feature_lr'], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args['feature_lr'] / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args['opacity_lr'], "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args['scaling_lr'], "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args['rotation_lr'], "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args['position_lr_init']*self.spatial_lr_scale,
                                                    lr_final=training_args['position_lr_final']*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args['position_lr_delay_mult'],
                                                    max_steps=training_args['position_lr_max_steps'])

    def forward(self, **kwargs):
        pass
        









    