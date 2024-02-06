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
from sdfgs.sdfgs_model import SDFGSModel, SDFGSModelConfig, NeuSAccModel, NeuSAccModelConfig
from sdfgs.sdfgs_gs_model import SDFGSGaussianModel, SDFGSGaussianModelConfig
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

from nerfstudio.utils import profiler


@dataclass
class SDFGSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SDFGSPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = SDFGSDataManagerConfig()
    """specifies the datamanager config"""
    sdf_model: ModelConfig = SDFGSModelConfig()
    """specifies the sdf model config"""
    gs_model: ModelConfig = SDFGSGaussianModelConfig()
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
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._sdf_model : SDFGSModel = config.sdf_model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self._sdf_model.to(device)

        self._gs_model : SDFGSGaussianModel = config.gs_model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler
        )
        self._gs_model.to(device)
        self._gs_model.field.reset_mesh(self._sdf_model.get_sdf_network(), resolution=self._gs_model.field.get_mc_res)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                SDFGSModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

        # self.device = device

    @property
    def gs_model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._gs_model)
    
    @property
    def sdf_model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._sdf_model)
    
    @profiler.time_function
    def get_train_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, ray_batch, camera, cam_batch = self.datamanager.next_train(step)
        sdf_model_outputs = self._sdf_model(ray_bundle)  
        sdf_metrics_dict = self.sdf_model.get_metrics_dict(sdf_model_outputs, ray_batch)
        sdf_loss_dict = self.sdf_model.get_loss_dict(sdf_model_outputs, ray_batch, sdf_metrics_dict)
        
        loss_dict = {**sdf_loss_dict}
        metrics_dict = {**sdf_metrics_dict}

        if step == self.config.steps_sdf_init or step % self.config.interval_reset_mesh == 0:
            self._gs_model.field.output_mesh(f'outputs/debug_mesh/{step}_before.ply')
            self._gs_model.field.reset_mesh(self._sdf_model.get_sdf_network(), resolution=self._gs_model.field.get_mc_res)
            self._gs_model.field.output_mesh(f'outputs/debug_mesh/{step}_after.ply')

        if step > self.config.steps_sdf_init:
            gs_model_outputs = self._gs_model(camera, self._sdf_model.get_sdf_network(), self._sdf_model.get_shared_color_network())
            gs_metrics_dict = self.gs_model.get_metrics_dict(gs_model_outputs, cam_batch)
            gs_loss_dict = self.gs_model.get_loss_dict(gs_model_outputs, cam_batch, gs_metrics_dict)

            loss_dict.update(gs_loss_dict)
            metrics_dict.update(gs_metrics_dict)

        self._gs_model.step_cb(step)

        return sdf_model_outputs, loss_dict, metrics_dict
    
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
        sdf_model_outputs = self._sdf_model(ray_bundle)  
        sdf_metrics_dict = self.sdf_model.get_metrics_dict(sdf_model_outputs, ray_batch)
        sdf_loss_dict = self.sdf_model.get_loss_dict(sdf_model_outputs, ray_batch, sdf_metrics_dict)
        
        loss_dict = {**sdf_loss_dict}
        metrics_dict = {**sdf_metrics_dict}

        if step > self.config.steps_sdf_init:
            gs_model_outputs = self._gs_model(camera, self._sdf_model.get_sdf_network(), self._sdf_model.get_shared_color_network())
            gs_metrics_dict = self.gs_model.get_metrics_dict(gs_model_outputs, cam_batch)
            gs_loss_dict = self.gs_model.get_loss_dict(gs_model_outputs, cam_batch, gs_metrics_dict)

            loss_dict.update(gs_loss_dict)
            metrics_dict.update(gs_metrics_dict)

        self._gs_model.step_cb(step)
        self.train()

        return sdf_model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        sdf_model_outputs = self.sdf_model.get_outputs_for_camera(camera)  
        sdf_metrics_dict, sdf_images_dict = self.sdf_model.get_image_metrics_and_images(sdf_model_outputs, batch)

        metrics_dict = {**sdf_metrics_dict}
        images_dict = {**sdf_images_dict}

        if step > self.config.steps_sdf_init:
            gs_model_outputs = self._gs_model.get_outputs_for_camera(camera, self._sdf_model.get_sdf_network(), self._sdf_model.get_shared_color_network())
            gs_metrics_dict, gs_images_dict = self.gs_model.get_image_metrics_and_images(gs_model_outputs, batch)

            metrics_dict.update(gs_metrics_dict)
            images_dict.update(gs_images_dict)
        
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
        sdf_model_callbacks = self.sdf_model.get_training_callbacks(training_callback_attributes)
        gs_model_callbacks = self.gs_model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + sdf_model_callbacks + gs_model_callbacks
        return callbacks
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        sdf_model_params = self.sdf_model.get_param_groups()
        gs_model_params = self.gs_model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **sdf_model_params, **gs_model_params}
    

@dataclass
class NeuSAccPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NeuSAccPipeline)
    """target class to instantiate"""
    model: ModelConfig = NeuSAccModelConfig()

class NeuSAccPipeline(VanillaPipeline):


    def __init__(
        self,
        config: NeuSAccPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            # pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]


        # to modify the sampling method 
        # def occ_eval_fn(x):
        #     hidden_output = self.model.field.forward_geonetwork(x)
        #     sdf, _ = torch.split(hidden_output, [1, self.model.field.config.geo_feat_dim], dim=-1)
        #     return sdf.abs()
        
        # self.model.sampler.grid._update(
        #     step=0,
        #     occ_eval_fn=occ_eval_fn,
        #     occ_thre= 0.05
        # )
        # self.model.sampler.update_binary_grid_w_points(pts)
        