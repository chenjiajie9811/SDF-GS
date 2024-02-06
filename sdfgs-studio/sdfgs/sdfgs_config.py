"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from pathlib import Path

from sdfgs.sdfgs_datamanager import (
    SDFGSDataManagerConfig,
)
from sdfgs.sdfgs_model import SDFGSModelConfig, NeuSAccModelConfig
from sdfgs.sdfgs_field import SDFGSFieldConfig
from sdfgs.sdfgs_gs_model import SDFGSGaussianModelConfig
from sdfgs.sdfgs_pipeline import (
    SDFGSPipelineConfig,
    NeuSAccPipeline,NeuSAccPipelineConfig
)
# from sdfgs.scaffoldgs_model import ScaffoldGaussianModel, ScaffoldGaussianModelConfig
from sdfgs.scaffoldgs_vm_model import ScaffoldGaussianModelConfig
from sdfgs.scaffoldgs_pipeline import ScafflodGSPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
    CosineDecaySchedulerConfig
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.models.neus import NeuSModelConfig


# SDFGS = MethodSpecification(
#     config=TrainerConfig(
#         method_name="SDFGS",  
#         steps_per_eval_image=250,
#         steps_per_eval_all_images=50000000,
#         steps_per_save=2000,
#         max_num_iterations=30000,
#         mixed_precision=False,
#         pipeline=SDFGSPipelineConfig(
#             datamanager=SDFGSDataManagerConfig(
#                 # dataparser=BlenderDataParserConfig(
#                 #     data=Path('/usr/stud/chj/storage/user/chj/datasets/lego')
#                 # ),
#                 dataparser=ColmapDataParserConfig(
#                     data=Path('/usr/stud/chj/storage/user/chj/datasets/db/playroom'),
#                     colmap_path=Path("sparse/0"),
#                     load_3D_points=True
#                 ),
#                 train_num_rays_per_batch=4096,
#                 eval_num_rays_per_batch=4096,
#             ),
#             # model=SDFGSModelConfig(
#             #     eval_num_rays_per_chunk=1024,
#             # ),
#             model=None,
#             sdf_model=SDFGSModelConfig(
#                 eval_num_rays_per_chunk=512,
#                 # background_model='none'
#             ),
#             gs_model=SDFGSGaussianModelConfig(

#             )
#         ),
#         optimizers={
#             # TODO: consider changing optimizers depending on your custom method
#             # "proposal_networks": {
#             #     "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#             #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
#             # },
#             "field_sdf": {
#                 "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
#                 "scheduler": CosineDecaySchedulerConfig(warm_up_end=1000, learning_rate_alpha=0.05, max_steps=200001),
#             },
#             "field_sdf_background": {
#                 "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
#                 "scheduler": CosineDecaySchedulerConfig(warm_up_end=1000, learning_rate_alpha=0.05, max_steps=200000),
#             },
#             "field_gs": {
#                 "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
#                 "scheduler": CosineDecaySchedulerConfig(warm_up_end=1000, learning_rate_alpha=0.05, max_steps=200000),
#             },

#             # "fields": {
#             #     "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
#             #     "scheduler": CosineDecaySchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
#             # },
#             # "field_background": {
#             #     "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
#             #     "scheduler": CosineDecaySchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
#             # },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="tensorboard",
#     ),
#     description="SDFGS.",
# )

# SDFGS = MethodSpecification(
#     config=TrainerConfig(
#         method_name="SDFGS",  
#         steps_per_eval_image=2000,
#         steps_per_eval_all_images=50000000,
#         steps_per_save=2000,
#         max_num_iterations=30000,
#         mixed_precision=False,
#        pipeline=NeuSAccPipelineConfig(
#         datamanager=VanillaDataManagerConfig(
#             # dataparser=BlenderDataParserConfig(
#             #         data=Path('/usr/stud/chj/storage/user/chj/datasets/lego')
#             #     ),
#             dataparser=ColmapDataParserConfig(
#                     data=Path('/usr/stud/chj/storage/user/chj/datasets/db/playroom'),
#                     colmap_path=Path("sparse/0"),
#                     load_3D_points=True
#                 ),
#             train_num_rays_per_batch=512,
#             eval_num_rays_per_batch=512,
#         ),
#         model=NeuSAccModelConfig(
#             eval_num_rays_per_chunk=512,
#             # sdf_field=SDFGSFieldConfig(
#             #     bias=0.8
#             # )
#             background_color='white',
#             sdf_field=SDFFieldConfig(
#                 num_layers=2,
#                 hidden_dim=128,
#                 geo_feat_dim=31,
#                 num_layers_color=4,
#                 hidden_dim_color=64,
#                 bias=0.8,
#                 geometric_init=True,
#                 # encoding_type='tensorf_vm',
#             ),
#             background_model="none"
#             ),
#     ),
#     optimizers={
#         "fields": {
#             "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
#             "scheduler": CosineDecaySchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
#         },
#         "field_background": {
#             "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
#             "scheduler": CosineDecaySchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
#         },
#     },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="tensorboard",
#     ),
#     description="SDFGS.",
# )

SDFGS = MethodSpecification(
    config=TrainerConfig(
        method_name="SDFGS",  
        steps_per_eval_image=100,
        steps_per_eval_all_images=50000000,
        steps_per_save=200,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=ScafflodGSPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                # dataparser=BlenderDataParserConfig(
                #         data=Path('/usr/stud/chj/storage/user/chj/datasets/lego')
                #     ),
                dataparser=ColmapDataParserConfig(
                        data=Path('/usr/stud/chj/storage/user/chj/datasets/db/playroom'),
                        colmap_path=Path("sparse/0"),
                        scale_factor=0.7, # try to make all the scene inside the -1, 1 box
                        load_3D_points=True,
                    ),
            ),
            model=ScaffoldGaussianModelConfig(),
        ),
        optimizers={
            "anchor": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "offset": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-4,
                    max_steps=30000,
                ),
            },
            "anchor_feat": {
                "optimizer": AdamOptimizerConfig(lr=0.0075, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.007, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.002, eps=1e-15), 
                "scheduler": None
            },
            "encoding_embedding": {
                "optimizer": AdamOptimizerConfig(lr=0.002, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-4,
                    max_steps=30000,
                ),
            },

            "mlp_opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.002, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-4,
                    max_steps=30000,
                ),
            },
            "mlp_cov": {
                "optimizer": AdamOptimizerConfig(lr=0.04, eps=1e-15),
                "scheduler": None
            },
            "mlp_color": {
                "optimizer": AdamOptimizerConfig(lr=0.008, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00005,
                    max_steps=30000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description="SDFGS.",
)