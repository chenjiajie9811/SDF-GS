"""
Implementation of the SDFGS 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Type, Literal, Tuple, Union, Dict, List, cast

from sdfgs.sdfgs_field import ImplicitNetworkGrid
from sdfgs.sdfgs_gs_field import RenderingNetwork, MeshGaussiansField, MeshGaussiansFieldConfig
from sdfgs.gs_renderer import GSCameraInfo, GaussianRenderer

from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.cameras.cameras import Cameras

from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import normalized_depth_scale_and_shift

from nerfstudio.model_components.losses import L1Loss, MSELoss
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def projection_matrix(znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"):
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

    proj_mat = projection_matrix(0.001, 1000, fovx, fovy, device=device)

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



@dataclass
class SDFGSGaussianModelConfig(ModelConfig):

    _target: Type = field(default_factory=lambda: SDFGSGaussianModel)

    gs_field: MeshGaussiansFieldConfig = field(default_factory=MeshGaussiansFieldConfig)

    sh_degree: int = 0
    """maximum degree of spherical harmonics to use"""

    white_background: bool = True

    ssim_lambda: float = 0.2

    use_scale_regularization: bool = True
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """


class SDFGSGaussianModel(Model):
    config: SDFGSGaussianModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.scene_contraction = SceneContraction(order=float("inf"))

        # background
        bg_color = [1, 1, 1] if self.config.white_background else [0, 0, 0]
        self.background_color = torch.tensor(bg_color, dtype=torch.float32).cuda()

        # field
        self.field : MeshGaussiansField = self.config.gs_field.setup(
            sh_degree=self.config.sh_degree,
        )

        # renderer
        self.renderer = GaussianRenderer()

        # losses
        self.rgb_loss = L1Loss()
        self.eikonal_loss = MSELoss()
        # self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        self.step = 0

    def step_cb(self, step):
        self.step = step

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        param_groups = {}
        param_groups["field_gs"] = list(self.field.parameters())
        return param_groups


    def get_outputs(self, view_camera: Cameras, sdf_network : ImplicitNetworkGrid, shared_color_network : RenderingNetwork) -> Dict[str, Union[torch.Tensor, List]]:
        if not isinstance(view_camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert view_camera.shape[0] == 1, "Only one camera at a time"

        gs_camera_info = parse_camera_info(view_camera, self.device)

        self.field.update_gaussians(gs_camera_info.camera_center, sdf_network, shared_color_network)

        render_pkg = self.renderer.render(gs_camera_info, self.field, self.background_color)

        # image, depth, viewspace_point_tensor, visibility_filter, radii = \
        #     render_pkg["image"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        return {"rgb": render_pkg["image"], "depth": render_pkg["depth"]}

    def forward(self, view_camera: Cameras, sdf_network : ImplicitNetworkGrid, shared_color_network : RenderingNetwork):
        return self.get_outputs(view_camera, sdf_network, shared_color_network)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        # d = self._get_downscale_factor()
        # if d > 1:
        #     newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
        #     gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        # else:
        gt_img = batch["image"]
        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr_gs"] = self.psnr(predicted_rgb, gt_rgb)

        # self.camera_optimizer.get_metrics_dict(metrics_dict)
        metrics_dict["gaussian_count"] = self.field.get_num_gaussians
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # d = self._get_downscale_factor()
        # if d > 1:
        #     newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
        #     gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        # else:
        gt_img : torch.Tensor = batch["image"]
        Ll1 = torch.abs(gt_img - outputs["rgb"]).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], outputs["rgb"].permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.field.get_scaling)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(self.config.max_gauss_ratio)
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        return {
            "main_loss_gs": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg_gs": scale_reg,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, view_camera: Cameras, sdf_network : ImplicitNetworkGrid, shared_color_network : RenderingNetwork) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert view_camera is not None, "must provide camera to gaussian model"
        # self.set_crop(obb_box)
        outs = self.get_outputs(view_camera, sdf_network, shared_color_network)
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
        # d = self._get_downscale_factor()
        # if d > 1:
        #     newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
        #     gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        #     predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        # else:
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
        metrics_dict = {"psnr_gs": float(psnr.item()), "ssim_gs": float(ssim)}  # type: ignore
        metrics_dict["lpips_gs"] = float(lpips)

        images_dict = {"img_gs": combined_rgb}

        return metrics_dict, images_dict