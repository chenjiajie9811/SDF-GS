"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch
from torch.nn import Parameter
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from sdfgs.sdfgs_datamanager import SDFGSDataManagerConfig
from sdfgs.scaffoldgs_neus_model import ScaffoldGaussianModel, ScaffoldGaussianModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.pipelines.base_pipeline import (
    Pipeline,
    VanillaPipeline,
    VanillaPipelineConfig,
    module_wrapper
)
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

from nerfstudio.utils import profiler

from pathlib import Path

from utils.graphics_utils import BasicPointCloud


@dataclass
class SDFGSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SDFGSPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = SDFGSDataManagerConfig(
        dataparser=ColmapDataParserConfig(
                        data=Path('/usr/stud/chj/storage/user/chj/datasets/db/playroom'),
                        colmap_path=Path("sparse/0"),
                        scale_factor=0.7, # try to make all the scene inside the -1, 1 box
                        load_3D_points=True,
                    ),
    )
    """specifies the datamanager config"""
    model: ScaffoldGaussianModelConfig = ScaffoldGaussianModelConfig()
    """specifies the gs model config"""
    steps_sdf_init: int = 250
    """steps to run volume rendering alone for initialization"""
    interval_reset_mesh: int = 250
    """interval to run marching cubes from the sdf field"""


class SDFGSPipeline(Pipeline):


    def __init__(
        self,
        config: SDFGSPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        
        # data manager
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]

        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        # model
        self._model : ScaffoldGaussianModel = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self._model.to(device)

        # set up model
        aabb = self.datamanager.train_dataset.scene_box.aabb
        mask = (pts[:, 0] >= aabb[0, 0]) & (pts[:, 0] < aabb[1, 0]) &\
                (pts[:, 1] >= aabb[0, 1]) & (pts[:, 1] < aabb[1, 1]) &\
                (pts[:, 2] >= aabb[0, 2]) & (pts[:, 2] < aabb[1, 2])
        pts = pts[mask]
        pts_rgb = pts_rgb[mask]

        pc = BasicPointCloud(points=pts, colors=pts_rgb / 255, normals=torch.zeros_like(pts))
        self._model.create_from_pcd(pc, 1.0)
        self._model.training_setup()

        
        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                ScaffoldGaussianModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

        # self.device = device

    @property
    def model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._model)
    
    
    @profiler.time_function
    def get_train_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, ray_batch, camera, cam_batch = self.datamanager.next_train(step)

        model_outputs = self._model.forward(camera, ray_bundle)  
        metrics_dict = self._model.get_metrics_dict(model_outputs, cam_batch, ray_batch)
        loss_dict = self._model.get_loss_dict(model_outputs, cam_batch, ray_batch)

        return model_outputs, loss_dict, metrics_dict
    
    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError
    
    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()

        ray_bundle, ray_batch, camera, cam_batch = self.datamanager.next_eval(step)
        model_outputs = self._model.forward(camera, ray_bundle)  
        metrics_dict = self._model.get_metrics_dict(model_outputs, cam_batch, ray_batch)
        loss_dict = self._model.get_loss_dict(model_outputs, cam_batch, ray_batch)

        self.train()

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        model_outputs = self.model.get_outputs_for_camera(camera)  
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(model_outputs, batch)
        
        self.train()

        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics():
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        raise NotImplementedError
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
    
        