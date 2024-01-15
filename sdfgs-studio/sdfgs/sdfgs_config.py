"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from sdfgs.sdfgs_datamanager import (
    SDFGSDataManagerConfig,
)
from sdfgs.sdfgs_model import SDFGSModelConfig
from sdfgs.sdfgs_pipeline import (
    SDFGSPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
    CosineDecaySchedulerConfig
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


SDFGS = MethodSpecification(
    config=TrainerConfig(
        method_name="SDFGS",  
        steps_per_eval_image=500,
        steps_per_eval_all_images=50000000,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=SDFGSPipelineConfig(
            datamanager=SDFGSDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=SDFGSModelConfig(
                eval_num_rays_per_chunk=1024,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            # "proposal_networks": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            # },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=1000, learning_rate_alpha=0.05, max_steps=200001),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=1000, learning_rate_alpha=0.05, max_steps=200000),
            },

            # "fields": {
            #     "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            #     "scheduler": CosineDecaySchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
            # },
            # "field_background": {
            #     "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            #     "scheduler": CosineDecaySchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
            # },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="SDFGS.",
)