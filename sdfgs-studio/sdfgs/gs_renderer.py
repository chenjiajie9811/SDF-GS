import os
import math
import torch
from dataclasses import dataclass
from sdfgs.sdfgs_gs_field import MeshGaussiansField
# from sdfgs.scaffoldgs_model import ScaffoldGaussianModel
from utils.sh_utils import eval_sh
from einops import repeat

from nerfstudio.cameras.cameras import Cameras

from diff_gaussian_rasterization import (
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

def projection_matrix(znear, zfar, fovx, fovy, device = "cpu"):
    """
        Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


def parse_camera_info(camera: Cameras, device):
    """
        Prepare the necessary data for the gaussian splatting renderer
    """

    # shift the camera to center of scene looking at center
    R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
    T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1

    # flip the z and y axes to align with gsplat conventions
    R_edit = torch.diag(torch.tensor([1, -1, -1], device=device, dtype=R.dtype))
    R = R @ R_edit

    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ T
    view_mat = torch.eye(4, device=R.device, dtype=R.dtype)
    view_mat[:3, :3] = R_inv
    view_mat[:3, 3:4] = T_inv

    view_mat = view_mat.transpose(0, 1) # From gs cameras.py

    # calculate the FOV of the camera given fx and fy, width and height
    fovx = 2 * math.atan(camera.width / (2 * camera.fx))
    fovy = 2 * math.atan(camera.height / (2 * camera.fy))

    proj_mat = projection_matrix(0.001, 1000, fovx, fovy, device=device).transpose(0,1)

    full_proj_mat = view_mat.unsqueeze(0).bmm(proj_mat.unsqueeze(0)).squeeze(0) # From gs cameras.py
    camera_center = view_mat.inverse()[3, :3]

    gs_cam_info = GSCameraInfo(
            FoVx=fovx,
            FoVy=fovy,
            image_height=camera.height,
            image_width=camera.width,
            world_view_transform=view_mat,
            full_proj_transform=full_proj_mat,
            camera_center=camera_center
        )

    return gs_cam_info

def generate_neural_gaussians(viewpoint_camera : GSCameraInfo, pc , visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    # feat = pc._anchor_feat[visible_mask]
    
    anchor = pc.get_anchor[visible_mask]
    feat = pc.encoding(anchor)
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    # if pc.appearance_dim > 0:
    #     camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
    #     # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
    #     appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    # if pc.appearance_dim > 0:
    #     if pc.add_color_dist:
    #         color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
    #     else:
    #         color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    # else:
    if pc.add_color_dist:
        color = pc.get_color_mlp(cat_local_view)
    else:
        color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot



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

    def prefilter_voxel(self, viewpoint_camera : GSCameraInfo, pc , bg_color : torch.Tensor):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
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
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_anchor


        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        # if self.compute_cov3D_python:
        #     cov3D_precomp = pc.get_covariance(self.scaling_modifier)
        # else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        

        radii_pure = rasterizer.visible_filter(means3D = means3D,
            scales = scales[:,:3],
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        return radii_pure > 0

    def render(
            self,
            viewpoint_camera : GSCameraInfo, 
            pc , 
            bg_color : torch.Tensor, 
            visible_mask=None,
            retain_grad=False):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
        is_training = True#pc.get_color_mlp.training
            
        if is_training:
            xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
        else:
            xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
        if retain_grad:
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
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = color,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)

        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        if is_training:
            return {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "selection_mask": mask,
                    "neural_opacity": neural_opacity,
                    "scaling": scaling,
                    }
        else:
            return {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    }

    
    # def render_(
    #         self,
    #         viewpoint_camera : GSCameraInfo, 
    #         pc : MeshGaussiansField, 
    #         bg_color : torch.Tensor,
    #     ):
    #     """
    #     Render the scene. 
        
    #     Background tensor (bg_color) must be on GPU!
    #     """
    
    #     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    #     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    #     try:
    #         screenspace_points.retain_grad()
    #     except:
    #         pass

    #     # Set up rasterization configuration
    #     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    #     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    #     raster_settings = GaussianRasterizationSettings(
    #         image_height=int(viewpoint_camera.image_height),
    #         image_width=int(viewpoint_camera.image_width),
    #         tanfovx=tanfovx,
    #         tanfovy=tanfovy,
    #         bg=bg_color,
    #         scale_modifier=self.scaling_modifier,
    #         viewmatrix=viewpoint_camera.world_view_transform,
    #         projmatrix=viewpoint_camera.full_proj_transform,
    #         sh_degree=pc.active_sh_degree,
    #         campos=viewpoint_camera.camera_center,
    #         prefiltered=False,
    #     )

    #     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    #     means3D = pc.get_xyz
    #     means2D = screenspace_points
    #     opacity = pc.get_opacity

    #     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    #     # scaling / rotation by the rasterizer.
    #     scales = None
    #     rotations = None
    #     cov3D_precomp = None
    #     if self.compute_cov3D_python:
    #         cov3D_precomp = pc.get_covariance(self.scaling_modifier)
    #     else:
    #         scales = pc.get_scaling
    #         rotations = pc.get_rotation

    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation
    #     cov3D_precomp = None

    #     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    #     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    #     shs = None
    #     colors_precomp = None
    #     if self.override_color is None:
    #         if self.convert_SHs_python:
    #             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    #             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #         else:
    #             shs = pc.get_features
    #     else:
    #         colors_precomp = self.override_color

    #     print ("inside renderer")
    #     print ("in scale", scales)
    #     print ("in rotation", rotations)
    #     print ("self.compute_cov3D_python", self.compute_cov3D_python)
    #     print ("cov", cov3D_precomp)
    #     print ("opacity", pc.get_opacity)
    #     print ("rotation", pc.get_rotation)
    #     print ("scale", pc.get_scaling)
    #     # print ("color output", color_output['color'])

        
        
    #     # Rasterize visible Gaussians to image, obtain their radii (on screen).
    #     rendered_image, radii, rendered_depth = rasterizer(
    #         means3D=means3D,
    #         means2D=means2D,
    #         shs=shs,
    #         colors_precomp=colors_precomp,
    #         opacities=opacity,
    #         scales=scales,
    #         rotations=rotations,
    #         cov3D_precomp=cov3D_precomp,
    #     )

    #     rendered_image = rendered_image.clamp(0, 1)

    #     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    #     # They will be excluded from value updates used in the splitting criteria.
    #     return {
    #         "image": rendered_image,
    #         "depth": rendered_depth,
    #         # "alpha": rendered_alpha,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter": radii > 0,
    #         "radii": radii,
    #     }