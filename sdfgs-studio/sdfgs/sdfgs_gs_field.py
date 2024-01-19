import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float
from sdfgs.embedder import *
import numpy as np
from skimage import measure

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type, cast

from sdfgs.embedder import *
from sdfgs.sdfgs_field import ImplicitNetworkGrid, SDFGSFieldConfig

import trimesh
from nerfstudio.exporter.marching_cubes import evaluate_sdf

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.fields.base_field import Field, FieldConfig

from utils.system_utils import mkdir_p
from utils.sh_utils import eval_sh, SH2RGB, RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.rigid_utils import matrix_to_quaternion, exp_so3


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

        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        self.color_lin = nn.Linear(d_hidden, d_out)
        self.scaling_lin = nn.Linear(d_hidden, 3)
        self.rotation_angle_lin = nn.Linear(d_hidden, 1)


    def forward(self, points, normals, view_dirs, feature_vectors, pred_gauss_params=False):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        
        x = rendering_input

        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        color = self.color_lin(x)

        if self.squeeze_out:
            color = torch.sigmoid(color)

        ret = {'color' : color}

        if pred_gauss_params:
            scale = torch.sigmoid(self.scaling_lin(x))
            rotation_theta = torch.sigmoid(self.rotation_angle_lin(x)) * 2 * torch.pi

            ret['scale'] = scale
            ret['rotation_angle'] = rotation_theta 

        
        return ret

# class GaussianGeoNetwork(nn.Module):
#     def __init__(self,
#                  d_feature,
#                  d_in,
#                  d_hidden,
#                  n_layers,
#                  weight_norm=True,
#                  multires_view=0,
#                 ):
#         super().__init__()

#         dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)]

#         self.embedview_fn = None
#         if multires_view > 0:
#             embedview_fn, input_ch = get_embedder(multires_view)
#             self.embedview_fn = embedview_fn
#             dims[0] += (input_ch - 3)

#         self.num_layers = len(dims)

#         for l in range(0, self.num_layers - 1):
#             out_dim = dims[l + 1]
#             lin = nn.Linear(dims[l], out_dim)

#             if weight_norm:
#                 lin = nn.utils.weight_norm(lin)

#             setattr(self, "lin" + str(l), lin)

#         self.relu = nn.ReLU()

#         self.rotation_angle = nn.Linear(d_hidden, 1)
#         self.scale = nn.Linear(d_hidden, 3)


@dataclass 
class MeshGaussiansFieldConfig(SDFGSFieldConfig):
    _target: Type = field(default_factory=lambda: MeshGaussiansField)
    init_marching_cubes_resolution : int = 128

class MeshGaussiansField(Field):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # L = build_scaling_rotation(scaling_modifier * scaling, rotation)
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

        

    def __init__(
        self, 
        config: MeshGaussiansFieldConfig,
        sh_degree : int, 
        device='cuda'
    ):
        super().__init__()

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

        # self.deviation_network = LearnedVariance(init_val=config.beta_init)
        # self.shared_color_network : RenderingNetwork = None
        self.init_mc_res = self.config.init_marching_cubes_resolution

        # Mesh related
        self.vertices = torch.empty(0)
        self.faces = torch.empty(0)
        self.center_normal = torch.empty(0)

        # From SuGar
        self.surface_triangle_circle_radius = 1. / 2. / np.sqrt(3.)
        self.bary_coords = torch.tensor([[1/3, 1/3, 1/3]], dtype=torch.float32, device=self.device)[..., None]

    def reset_mesh(self, sdf_network : ImplicitNetworkGrid, resolution : int = 128):
        # Reset the Gaussians by the mesh from sdf network
        # Need to be call at least once after initializing this field
        v, f, _ = self.marching_cubes(sdf_network, resolution)
        
        v = (v / resolution - 0.5) * 2.0 * 1.5

        # print ('after marching cubes')
        # print (v.min())
        # print (v.max())
        self.update_gaussians_center_info(np.ascontiguousarray(v), np.ascontiguousarray(f))
    
    def generate_3d_grid(self,
                         resolution, 
                         bounding_box_min=(-1.5, -1.5, -1.5), 
                         bounding_box_max=(1.5, 1.5, 1.5)):
        # Generate 1D coordinates along each axis
        x_coords = torch.linspace(bounding_box_min[0], bounding_box_max[0], resolution)
        y_coords = torch.linspace(bounding_box_min[1], bounding_box_max[1], resolution)
        z_coords = torch.linspace(bounding_box_min[2], bounding_box_max[2], resolution)

        # Create a 3D grid using meshgrid
        x, y, z = torch.meshgrid(x_coords, y_coords, z_coords)

        # Reshape the grid coordinates to obtain a list of 3D points
        coordinates = torch.column_stack((x.ravel(), y.ravel(), z.ravel()))

        return coordinates
    
    @torch.no_grad()
    def marching_cubes(self, sdf_network : ImplicitNetworkGrid, resolution : int = 128):
        coords = self.generate_3d_grid(resolution)
        coords = coords.to(self.device)

        pts_sdf = evaluate_sdf(
            sdf=lambda x: sdf_network.get_sdf_vals(x).contiguous(),
            points=coords)

        z = pts_sdf.detach().cpu().numpy().reshape(resolution, resolution, resolution)
        v, f, n, _ = measure.marching_cubes(volume=z, level=0)

        return v, f, n


    def update_gaussians_center_info(self, vertices, faces):
        # print (vertices.shape)
        # print (type(vertices))
        self.vertices = torch.tensor(vertices, device=self.device)
        self.faces = torch.tensor(faces, device=self.device)

        self._xyz = 1. / 3. * (self.vertices[self.faces[:, 0].flatten()] + \
                                self.vertices[self.faces[:, 1].flatten()] + \
                                self.vertices[self.faces[:, 2].flatten()])      

        vec1 = self.vertices[self.faces[:, 0].flatten()] - self.vertices[self.faces[:, 1].flatten()] 
        vec2 = self.vertices[self.faces[:, 0].flatten()] - self.vertices[self.faces[:, 2].flatten()] 
        cross_pro = torch.cross(vec1, vec2, dim=-1)
        self.center_normal = F.normalize(cross_pro)

           

    def update_gaussians(
            self, 
            camera_center : Tensor, 
            sdf_network : ImplicitNetworkGrid,
            shared_color_network : RenderingNetwork):
        """
            1. Synchronize the mesh by moving the vertices towards the zero level set 
            according to the estimation of the current sdf network
            2. Update the gaussian xyz and normal according to the updated mesh
            3. Go through the geo_network for estimating the gaussian opacity and output a geo feature vector
            4. Go through the shared rendering network for estimating the shared color, the gaussian scales, and rotation angle
            5. Set up all the up-to-date Gaussians parameters
        """
        #1.
        # print ("before vertices move")
        # print (self.vertices)
        
        sdf, _, gradients = sdf_network.get_outputs(self.vertices)
        self.vertices = self.vertices - sdf * F.normalize(gradients, dim=-1)

        # print ("after vertice move")
        # print (self.vertices)

        #2.
        self.update_gaussians_center_info(self.vertices, self.faces)

        #3.
        geo_output = self.geo_network.forward(self._xyz)
        opacity = torch.sigmoid(geo_output[:, :1])
        geo_feat_vecs = geo_output[:, 1:]
        # opacity, geo_feat_vecs = torch.split(geo_output, [1, self.config.geo_feat_dim], dim=-1)
        # opacity[torch.isnan(opacity)] = 0.
        # print ("opacity after set nan to 0", opacity)

        #4. 
        view_dir = F.normalize(self._xyz - camera_center, dim=-1)
        color_output = shared_color_network.forward(self._xyz, self.center_normal, view_dir, geo_feat_vecs, pred_gauss_params=True)

        #5.
        features = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = RGB2SH(color_output['color'].squeeze(0))
        features[:, 3:, 1:] = 0.0
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        # self._features_dc.retain_grad()
        # self._features_rest.retain_grad()

        self._scaling = self.scaling_inverse_activation(color_output['scale'] * self.surface_triangle_circle_radius)
 
        rotation_matrix = exp_so3(self.center_normal.squeeze(0), color_output['rotation_angle'].squeeze(0))
        self._rotation = matrix_to_quaternion(rotation_matrix)
 
        self._opacity = self.inverse_opacity_activation(opacity.reshape(-1, 1))

        
        # print ("opacity", self.get_opacity)
        # print ("rotation", self.get_rotation)
        # print ("scale", self.get_scaling)
        # print ("color output", color_output['color'])



    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        
    @property
    def get_mc_res(self):
        return self.init_mc_res

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

    @property
    def get_num_gaussians(self):
        return self._xyz.shape[0]
    
    def get_covariance(self, scaling_modifier = 1):
        # print ('get_covariance')
        # print (self.get_scaling)
        # print (self.get_scaling.shape)
        
        # print (self._rotation.shape)
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def forward(self, **kwargs):
        pass
        