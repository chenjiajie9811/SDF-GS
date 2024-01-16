"""
Implementation of the SDFGS 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Type, Literal, Tuple, Union, Dict, List, cast

from sdfgs.sdfgs_field import SDFGSField, SDFGSFieldConfig, MeshGaussiansField, MeshGaussiansFieldConfig
from sdfgs.gs_renderer import GSCameraInfo, GaussianRenderer

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

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

    white_background = True
    


class SDFGSGaussianModel(Model):
    config: SDFGSGaussianModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.scene_contraction = SceneContraction(order=float("inf"))

        # background
        bg_color = [1, 1, 1] if self.config.white_background else [0, 0, 0]
        self.background_color = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        # field
        self.field : MeshGaussiansField = self.config.gs_field.setup()

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

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        return param_groups


    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        gs_camera_info = parse_camera_info(camera, self.device)

        #TODO:
        self.field.update_gaussians(gs_camera_info.camera_center, sdf_network)

        render_output = self.renderer.render(gs_camera_info, self.field, self.background_color)







       

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        loss_dict["rgb_loss"] = self.rgb_loss(image, pred_image)
        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult

            # foreground mask loss
            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_label = batch["fg_mask"].float().to(self.device)
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        Args:
            outputs: Outputs of the model.
            batch: Batch of data.

        Returns:
            A dictionary of metrics.
        """
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])

        normal = outputs["normal"]
        normal = (normal + 1.0) / 2.0

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        if "depth" in batch:
            depth_gt = batch["depth"].to(self.device)
            depth_pred = outputs["depth"]

            # align to predicted depth and normalize
            scale, shift = normalized_depth_scale_and_shift(
                depth_pred[None, ..., 0], depth_gt[None, ...], depth_gt[None, ...] > 0.0
            )
            depth_pred = depth_pred * scale + shift

            combined_depth = torch.cat([depth_gt[..., None], depth_pred], dim=1)
            combined_depth = colormaps.apply_depth_colormap(combined_depth)
        else:
            depth = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
            )
            combined_depth = torch.cat([depth], dim=1)

        if "normal" in batch:
            normal_gt = (batch["normal"].to(self.device) + 1.0) / 2.0
            combined_normal = torch.cat([normal_gt, normal], dim=1)
        else:
            combined_normal = torch.cat([normal], dim=1)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "normal": combined_normal,
        }

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        return metrics_dict, images_dict