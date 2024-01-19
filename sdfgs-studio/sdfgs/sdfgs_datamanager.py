

import random
from copy import deepcopy

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.nn import Parameter
from functools import cached_property

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle

from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    TDataset,
    variable_res_collate
)

from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig

from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import (
    PatchPixelSamplerConfig,
    PixelSampler,
    PixelSamplerConfig,
)

from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)

from nerfstudio.model_components.ray_generators import RayGenerator

from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE

@dataclass 
class SDFGSDataManagerConfig(DataManagerConfig):
    """A hybrid data manager for both ray-based and full image model"""
    _target: Type = field(default_factory=lambda: SDFGSDataManager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=BlenderDataParserConfig)

    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = -1
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    patch_size: int = 1
    """Size of patch to sample from. If > 1, patch-based sampling will be used."""
    pixel_sampler: PixelSamplerConfig = field(default_factory=PixelSamplerConfig)
    """Specifies the pixel sampler used to sample pixels from images."""

    cache_images: Literal["cpu", "gpu"] = "cpu"
    """Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device."""

class SDFGSDataManager(DataManager, Generic[TDataset]):
    config: SDFGSDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: FullImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()

        # if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
        #     CONSOLE.print("Train dataset has over 500 images, overriding cach_images to cpu", style="bold yellow")
        #     self.config.cache_images = "cpu"
        # self.cached_train, self.cached_eval = self.cache_images(self.config.cache_images)
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"

        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break

        super().__init__()

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )
    
    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")

        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None and "fisheye_crop_radius" in dataset.cameras.metadata:
            fisheye_crop_radius = dataset.cameras.metadata["fisheye_crop_radius"]

        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )
    
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict, Cameras, Dict]:
        """Returns the next batch of data from the train dataloader.
        
        Also returns a Camera for the full image pipeline"""
        # From vanilla data manager
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        # print (image_batch.keys())
        # print ('imageidx', image_batch['image_idx'])
        # print ('image.shape', image_batch['image'].shape)
        # print ('len(self.iter_train_image_dataloader)', len(self.train_image_dataloader))
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        ray_batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = ray_batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        # From full image data manager
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        # print("len(self.cached_train) ", len(self.cached_train))
        # print("len(self.train_unseen_cameras) ", len(self.train_unseen_cameras))
        img_data = deepcopy(image_batch['image'][image_idx])
        # img_data["image"] = img_data["image"].to(self.device)
        img_data = img_data.to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx

        return ray_bundle, ray_batch, camera, img_data

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict, Cameras, Dict]:
        """Returns the next batch of data from the eval dataloader.
        
        Also returns a Camera for the full image pipeline"""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        ray_batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = ray_batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)

        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        img_data = deepcopy(image_batch['image'][image_idx])
        img_data["image"] = img_data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)

        return ray_bundle, ray_batch, camera, img_data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data
    
    def get_train_rays_per_batch(self) -> int:
        if self.train_pixel_sampler is not None:
            return self.train_pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        if self.eval_pixel_sampler is not None:
            return self.eval_pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch
    
    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[SDFGSDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is SDFGSDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is SDFGSDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is SDFGSDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def cache_images(self, cache_images_option):
        cached_train = []
        cached_eval = []
        CONSOLE.log("Caching / undistorting train images")
        for i in tqdm(range(len(self.train_dataset)), leave=False):
            # cv2.undistort the images / cameras
            data = self.train_dataset.get_data(i)
            camera = self.train_dataset.cameras[i].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                continue
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()
            if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
                distortion_params = np.array(
                    [
                        distortion_params[0],
                        distortion_params[1],
                        distortion_params[4],
                        distortion_params[5],
                        distortion_params[2],
                        distortion_params[3],
                        0,
                        0,
                    ]
                )
                if np.any(distortion_params):
                    newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
                    image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
                else:
                    newK = K
                    roi = 0, 0, image.shape[1], image.shape[0]
                # crop the image and update the intrinsics accordingly
                x, y, w, h = roi
                image = image[y : y + h, x : x + w]
                if "depth_image" in data:
                    data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
                # update the width, height
                self.train_dataset.cameras.width[i] = w
                self.train_dataset.cameras.height[i] = h
                if "mask" in data:
                    mask = data["mask"].numpy()
                    mask = mask.astype(np.uint8) * 255
                    if np.any(distortion_params):
                        mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
                    mask = mask[y : y + h, x : x + w]
                    data["mask"] = torch.from_numpy(mask).bool()
                K = newK

            elif camera.camera_type.item() == CameraType.FISHEYE.value:
                distortion_params = np.array(
                    [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
                )
                newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
                )
                # and then remap:
                image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                if "mask" in data:
                    mask = data["mask"].numpy()
                    mask = mask.astype(np.uint8) * 255
                    mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
                    data["mask"] = torch.from_numpy(mask).bool()
                K = newK
            else:
                raise NotImplementedError("Only perspective and fisheye cameras are supported")
            data["image"] = torch.from_numpy(image)

            cached_train.append(data)

            self.train_dataset.cameras.fx[i] = float(K[0, 0])
            self.train_dataset.cameras.fy[i] = float(K[1, 1])
            self.train_dataset.cameras.cx[i] = float(K[0, 2])
            self.train_dataset.cameras.cy[i] = float(K[1, 2])

        CONSOLE.log("Caching / undistorting eval images")
        for i in tqdm(range(len(self.eval_dataset)), leave=False):
            # cv2.undistort the images / cameras
            data = self.eval_dataset.get_data(i)
            camera = self.eval_dataset.cameras[i].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                continue
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()
            if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
                distortion_params = np.array(
                    [
                        distortion_params[0],
                        distortion_params[1],
                        distortion_params[4],
                        distortion_params[5],
                        distortion_params[2],
                        distortion_params[3],
                        0,
                        0,
                    ]
                )
                if np.any(distortion_params):
                    newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
                    image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
                else:
                    newK = K
                    roi = 0, 0, image.shape[1], image.shape[0]
                # crop the image and update the intrinsics accordingly
                x, y, w, h = roi
                image = image[y : y + h, x : x + w]
                # update the width, height
                self.eval_dataset.cameras.width[i] = w
                self.eval_dataset.cameras.height[i] = h
                if "mask" in data:
                    mask = data["mask"].numpy()
                    mask = mask.astype(np.uint8) * 255
                    if np.any(distortion_params):
                        mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
                    mask = mask[y : y + h, x : x + w]
                    data["mask"] = torch.from_numpy(mask).bool()
                K = newK

            elif camera.camera_type.item() == CameraType.FISHEYE.value:
                distortion_params = np.array(
                    [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
                )
                newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
                )
                # and then remap:
                image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                if "mask" in data:
                    mask = data["mask"].numpy()
                    mask = mask.astype(np.uint8) * 255
                    mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
                    data["mask"] = torch.from_numpy(mask).bool()
                K = newK
            else:
                raise NotImplementedError("Only perspective and fisheye cameras are supported")
            data["image"] = torch.from_numpy(image)

            cached_eval.append(data)

            self.eval_dataset.cameras.fx[i] = float(K[0, 0])
            self.eval_dataset.cameras.fy[i] = float(K[1, 1])
            self.eval_dataset.cameras.cx[i] = float(K[0, 2])
            self.eval_dataset.cameras.cy[i] = float(K[1, 2])

        if cache_images_option == "gpu":
            for cache in cached_train:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
            for cache in cached_eval:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
        else:
            for cache in cached_train:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
            for cache in cached_eval:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()

        return cached_train, cached_eval


# @dataclass
# class SDFGSDataManagerConfig(VanillaDataManagerConfig):
#     """Template DataManager Config

#     Add your custom datamanager config parameters here.
#     """

#     _target: Type = field(default_factory=lambda: SDFGSDataManager)
#     dataparser: AnnotatedDataParserUnion = field(default_factory=BlenderDataParserConfig)
#     """Specifies the dataparser used to unpack the data."""


# class SDFGSDataManager(VanillaDataManager):
#     """Template DataManager

#     Args:
#         config: the DataManagerConfig used to instantiate class
#     """

#     config: SDFGSDataManagerConfig

#     def __init__(
#         self,
#         config: SDFGSDataManagerConfig,
#         device: Union[torch.device, str] = "cpu",
#         test_mode: Literal["test", "val", "inference"] = "val",
#         world_size: int = 1,
#         local_rank: int = 0,
#         **kwargs,  # pylint: disable=unused-argument
#     ):
#         super().__init__(
#             config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
#         )

#     def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
#         """Returns the next batch of data from the train dataloader."""
#         self.train_count += 1
#         image_batch = next(self.iter_train_image_dataloader)
#         assert self.train_pixel_sampler is not None
#         assert isinstance(image_batch, dict)
#         batch = self.train_pixel_sampler.sample(image_batch)
#         ray_indices = batch["indices"]
#         ray_bundle = self.train_ray_generator(ray_indices)
#         return ray_bundle, batch