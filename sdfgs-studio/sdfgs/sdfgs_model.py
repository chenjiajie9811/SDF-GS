"""
Implementation of the SDFGS 
"""
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import NeuSSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig


@dataclass
class SDFGSModelConfig(NeuSModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: SDFGSModel)


class SDFGSModel(NeuSModel):
    """Template Model."""

    config: SDFGSModelConfig

    def populate_modules(self):
        super().populate_modules()

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.