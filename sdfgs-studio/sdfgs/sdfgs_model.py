"""
Implementation of the SDFGS 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Type, Literal, Tuple, Dict, List, cast

from sdfgs.sdfgs_field import SDFGSField, SDFGSFieldConfig, ImplicitNetworkGrid

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
from nerfstudio.model_components.ray_samplers import NeuSSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer, SemanticRenderer
from nerfstudio.fields.vanilla_nerf_field import NeRFField

from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import normalized_depth_scale_and_shift

from nerfstudio.model_components.losses import L1Loss, MSELoss
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from sdfgs.sdfgs_gs_field import RenderingNetwork

from sdfgs.sdfgs_samplers import NeuSAccSampler
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig

import nerfacc

@dataclass
class NeuSAccModelConfig(NeuSModelConfig):

    _target: Type = field(default_factory=lambda: NeuSAccModel)
    sky_loss_mult: float = 0.01
    """Sky segmentation normal consistency loss multiplier."""

class NeuSAccModel(NeuSModel):


    config: NeuSAccModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.sampler = NeuSAccSampler(aabb=self.scene_box.aabb, neus_sampler=self.sampler)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # add sampler call backs
        sdf_fn = lambda x: self.field.forward_geonetwork(x)[:, 0].contiguous()
        inv_s = self.field.deviation_network.get_variance
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_binary_grid_w_sdf,
                kwargs={"sdf_fn": sdf_fn, "inv_s": inv_s},
            )
        )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_step_size,
                kwargs={"inv_s": inv_s},
            )
        )

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        # bootstrap with original Neus
        # if self.sampler._update_counter.item() <= 0:
        #     return super().get_outputs(ray_bundle)

        ray_samples, ray_indices = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf, alpha_fn=self.field.get_alpha, neus_fields=self.field)

        if ray_samples.shape[0] > 0:
            field_outputs = self.field(ray_samples, return_alphas=True)

            
            n_rays = ray_bundle.shape[0] 
            weights, _ = nerfacc.render_weight_from_alpha(
                field_outputs[FieldHeadNames.ALPHA].reshape(-1),
                ray_indices=ray_indices,
                n_rays=n_rays,
            )
            
            rgb = nerfacc.accumulate_along_rays(
                weights=weights, ray_indices=ray_indices, values=field_outputs[FieldHeadNames.RGB], n_rays=n_rays
            )
            normal = nerfacc.accumulate_along_rays(
                weights=weights, ray_indices=ray_indices, values=field_outputs[FieldHeadNames.NORMALS], n_rays=n_rays
            )

            accumulation = nerfacc.accumulate_along_rays(weights=weights, ray_indices=ray_indices, values=None, n_rays=n_rays)
            depth = nerfacc.accumulate_along_rays(
                weights=weights,
                ray_indices=ray_indices,
                values=(ray_samples.frustums.starts + ray_samples.frustums.ends) / 2,
                n_rays=n_rays,
            )

            # the rendered depth is point-to-point distance and we should convert to depth
            depth = depth / ray_bundle.metadata["directions_norm"]

            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
                "normal": normal,
            }

            if self.training:
                grad_points = field_outputs[FieldHeadNames.GRADIENT]
                outputs.update({"eik_grad": grad_points})
        else:
            zeros = torch.zeros((ray_bundle.shape[0], 3), dtype=torch.float32, device=self.device)
            outputs = {"rgb": zeros, "accumulation": zeros[:, :1], "depth": zeros[:, :1], "normal": zeros}
            if self.training:
                outputs.update({"eik_grad": zeros})

        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        # images_dict.update({"img_bg" : outputs['bg_rgb']})

        return metrics_dict, images_dict

    # def get_metrics_dict(self, outputs, batch):
    #     metrics = super().get_metrics_dict(outputs, batch)
    #     metrics["acc_step_size"] = self.sampler.step_size
    #     return metrics

@dataclass
class SDFGSModelConfig(ModelConfig):

    _target: Type = field(default_factory=lambda: SDFGSModel)
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
    
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 4.0
    """How far along the ray to stop sampling."""
    far_plane_bg: float = 1000.0
    """How far along the ray to stop sampling of the background model."""
    background_color: Literal["random", "last_sample", "white", "black"] = "black"
    """Whether to randomize the background color."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    eikonal_loss_mult: float = 0.1
    """Monocular normal consistency loss multiplier."""
    fg_mask_loss_mult: float = 0.01
    """Foreground mask loss multiplier."""
    mono_normal_loss_mult: float = 0.0
    """Monocular normal consistency loss multiplier."""
    mono_depth_loss_mult: float = 0.0
    """Monocular depth consistency loss multiplier."""
    sdf_field: SDFGSFieldConfig = field(default_factory=SDFGSFieldConfig)
    """Config for SDF Field"""
    background_model: Literal["grid", "mlp", "none"] = "mlp"
    """background models"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for background"""
    periodic_tvl_mult: float = 0.0
    """Total variational loss multiplier"""
    overwrite_near_far_plane: bool = False
    """whether to use near and far collider from command line"""


class SDFGSModel(Model):
    config: SDFGSModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.scene_contraction = SceneContraction(order=float("inf"))

        self.field : SDFGSField = self.config.sdf_field.setup()

        if self.config.background_model == "grid":
            raise NotImplementedError
        elif self.config.background_model == "mlp":
            position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
            )
            direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
            )

            self.field_background = NeRFField(
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
                spatial_distortion=self.scene_contraction,
            )
        else:
            # dummy background model
            self.field_background = nn.Parameter(torch.ones(1), requires_grad=False)

        self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)

        self.sampler = NeuSSampler(
            num_samples=self.config.num_samples,
            num_samples_importance=self.config.num_samples_importance,
            num_samples_outside=self.config.num_samples_outside,
            num_upsample_steps=self.config.num_up_sample_steps,
            base_variance=self.config.base_variance,
        )

        self.anneal_end = 50000
        
        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )

        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normal = SemanticRenderer()

        # losses
        self.rgb_loss = L1Loss()
        self.eikonal_loss = MSELoss()
        # self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_shared_color_network(self) -> RenderingNetwork:
        return self.field.shared_color_network

    def get_sdf_network(self) -> ImplicitNetworkGrid:
        return self.field.sdf_network

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        param_groups = {}
        param_groups["field_sdf"] = list(self.field.parameters())
        param_groups["field_sdf_background"] = (
            [self.field_background]
            if isinstance(self.field_background, nn.Parameter)
            else list(self.field_background.parameters())
        )
        return param_groups

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)
        field_outputs = self.field(ray_samples, return_alphas=True)
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

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        assert (
            ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata
        ), "directions_norm is required in ray_bundle.metadata"

        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

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

        # background model
        if self.config.background_model != "none":
            assert isinstance(self.field_background, torch.nn.Module), "field_background should be a module"
            assert ray_bundle.fars is not None, "fars is required in ray_bundle"
            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            assert ray_bundle.fars is not None
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            # use the same background model for both density field and occupancy field
            assert not isinstance(self.field_background, nn.Parameter)
            field_outputs_bg = self.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
            depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg)
            accumulation_bg = self.renderer_accumulation(weights=weights_bg)

            # merge background color to foregound color
            rgb = rgb + bg_transmittance * rgb_bg

            bg_outputs = {
                "bg_rgb": rgb_bg,
                "bg_accumulation": accumulation_bg,
                "bg_depth": depth_bg,
                "bg_weights": weights_bg,
            }
        else:
            bg_outputs = {}

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            # used to scale z_vals for free space and sdf loss
            "directions_norm": ray_bundle.metadata["directions_norm"],
        }
        outputs.update(bg_outputs)

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})
            outputs.update(samples_and_field_outputs)

        if "weights_list" in samples_and_field_outputs:
            weights_list = cast(List[torch.Tensor], samples_and_field_outputs["weights_list"])
            ray_samples_list = cast(List[torch.Tensor], samples_and_field_outputs["ray_samples_list"])

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs


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
        loss_dict["rgb_loss_sdf"] = self.rgb_loss(image, pred_image)
        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss_sdf"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult

            # foreground mask loss
            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_label = batch["fg_mask"].float().to(self.device)
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss_sdf"] = (
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
        metrics_dict["psnr_sdf"] = self.psnr(outputs["rgb"], image)
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
            "img_sdf": combined_rgb,
            "img_bg_sdf" : outputs['bg_rgb'],
            "accumulation_sdf": combined_acc,
            "depth_sdf": combined_depth,
            "normal_sdf": combined_normal,
        }

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr_sdf": float(psnr.item()), "ssim_sdf": float(ssim)}  # type: ignore
        metrics_dict["lpips_sdf"] = float(lpips)

        return metrics_dict, images_dict