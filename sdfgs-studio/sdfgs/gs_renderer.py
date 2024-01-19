import os
import math
import torch
from dataclasses import dataclass
from sdfgs.sdfgs_gs_field import MeshGaussiansField
from utils.sh_utils import eval_sh

from diff_gaussian_rasterization_w_depth import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

@dataclass
class GSCameraInfo:
    FoVx : float
    FoVy : float
    image_height : int
    image_width : int
    world_view_transform : torch.Tensor
    full_proj_transform : torch.Tensor
    camera_center : torch.Tensor

# @dataclass
# class GSRenderingConfig:
#     compute_cov3D_python : bool = False
#     convert_SHs_python : bool = False
#     scaling_modifier : float = 1.0
#     override_color = None

class GaussianRenderer:
    def __init__(
        self,
        compute_cov3D_python : bool = False,
        convert_SHs_python : bool = False,
        scaling_modifier : float = 1.0,
        override_color = None
    ):
        
        self.compute_cov3D_python = compute_cov3D_python,
        self.convert_SHs_python = convert_SHs_python,
        self.scaling_modifier = scaling_modifier,
        self.override_color = override_color

    def render(
            self,
            viewpoint_camera : GSCameraInfo, 
            pc : MeshGaussiansField, 
            bg_color : torch.Tensor,
        ):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(self.scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        scales = pc.get_scaling
        rotations = pc.get_rotation
        cov3D_precomp = None

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.override_color is None:
            if self.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = self.override_color

        print ("inside renderer")
        print ("in scale", scales)
        print ("in rotation", rotations)
        print ("self.compute_cov3D_python", self.compute_cov3D_python)
        print ("cov", cov3D_precomp)
        print ("opacity", pc.get_opacity)
        print ("rotation", pc.get_rotation)
        print ("scale", pc.get_scaling)
        # print ("color output", color_output['color'])

        
        
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            # "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }