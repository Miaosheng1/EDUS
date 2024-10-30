# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base Model implementation which takes in RayBundles
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox

from nerfstudio.model_components.scene_colliders import NearFarCollider


# Model related configs
@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: Model)
    """target class to instantiate"""
    enable_collider: bool = True
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    """parameters to instantiate scene collider with"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    """parameters to instantiate density field with"""
    eval_num_rays_per_chunk: int = 1024
    """specifies number of rays per chunk during eval"""


class Model(nn.Module):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        world_size: int = 1,
        local_rank: int = 0,
        device = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        self.num_train_data = num_train_data
        self.kwargs = kwargs
        self.collider = None
        self.world_size = world_size
        self.local_rank = local_rank
        self.config_file = kwargs.get('config_path',None)
        self.intrinsics = kwargs.get('intrinsics',None)

        assert kwargs.get("volume_dict") is not None
        assert kwargs.get('config_path') is not None

        self.populate_modules(volume_dict=kwargs.get("volume_dict",None), 
                              lidar_data=kwargs.get("lidar_data",None),
                              device=device
                              )  # populate the modules

        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device


    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        # default instantiates optional modules that are common among many networks
        # NOTE: call `super().populate_modules()` in subclasses

        if self.config.enable_collider:
            self.collider = NearFarCollider(
                near_plane=self.config.collider_params["near_plane"], far_plane=self.config.collider_params["far_plane"]
            )

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

    @abstractmethod
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    @abstractmethod
    def Debug_featurevolume(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """


    def forward(self, ray_bundle: RayBundle,mask=None) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        ##
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)  ## Set near =0.01  far =10.0

        return self.get_outputs(ray_bundle,mask=mask)
        # return self.Debug_featurevolume(ray_bundle)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        # pylint: disable=unused-argument
        # pylint: disable=no-self-use
        return {}

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
    @abstractmethod
    def get_lidar_depth(self,ray_bundle,sample_num_samples=256,step=0):
        """Computes lidar depth.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk//4
        if self.config.inference_dataset != "off":
            self.field.inference_dataset = self.config.inference_dataset
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            ray_bundle.set_source_image(source_images=camera_ray_bundle.source_images,
                                        source_poses=camera_ray_bundle.source_poses,
                                        source_depth=camera_ray_bundle.source_depth)
            outputs = self.forward(ray_bundle=ray_bundle)

            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs
    

    @torch.no_grad()
    def get_outputs_for_fixed_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        pixel_x, pixel_y = 286, 1134
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        selected_index = pixel_y  + pixel_x * image_width

        camera_ray_bundle_ = camera_ray_bundle.flatten()
        # camera_ray_bundle_.source_images = camera_ray_bundle.source_images
        # camera_ray_bundle_.source_poses = camera_ray_bundle.source_poses
        # camera_ray_bundle_.source_depth = camera_ray_bundle.source_depth

        ray_bundle = camera_ray_bundle_[selected_index-1:selected_index]

        num_rays_per_chunk = self.config.eval_num_rays_per_chunk//4
        if self.config.inference_dataset != "off":
            self.field.inference_dataset = self.config.inference_dataset
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
       
        ray_bundle.set_source_image(source_images=camera_ray_bundle.source_images,
                                    source_poses=camera_ray_bundle.source_poses,
                                    source_depth=camera_ray_bundle.source_depth)
        outputs = self.forward(ray_bundle=ray_bundle)

        for output_name, output in outputs.items():  # type: ignore
            outputs_lists[output_name].append(output)

        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs
    



    # @torch.no_grad()
    # def get_outputs_for_fixed_raybundle(self, camera_ray_bundle: RayBundle,pixel_x=None, pixel_y = None) -> Dict[str, torch.Tensor]:

    #     if self.config.inference_dataset != "off":
    #         self.field.inference_dataset = self.config.inference_dataset
    #     image_height, image_width = camera_ray_bundle.origins.shape[:2]
    #     num_rays = len(camera_ray_bundle)
    #     outputs_lists = defaultdict(list)

    #     selected_index = pixel_y * image_width + pixel_x
    #     BBx = camera_ray_bundle.bbx
    #     Test_id = camera_ray_bundle.test_id
    #     Train_id = camera_ray_bundle.train_id
    #     ## 对于 camear_ray_bundle 进行 shuffle
    #     camera_ray_bundle = camera_ray_bundle.flatten()
    #     ray_bundle = camera_ray_bundle[selected_index]
    #     ray_bundle.bbx = BBx
    #     ray_bundle.test_id = Test_id
    #     ray_bundle.train_id = Train_id

    #     outputs = self.forward(ray_bundle=ray_bundle)
    #     for output_name, output in outputs.items():  # type: ignore
    #         outputs_lists[output_name].append(output)
    #     outputs = {}
    #     for output_name, outputs_list in outputs_lists.items():
    #         if not torch.is_tensor(outputs_list[0]):
    #             # TODO: handle lists of tensors as well
    #             continue
    #         outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
    #     return outputs

    @abstractmethod
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        TODO: This shouldn't return a loss

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: dictionary of pre-trained model states
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state["model"].items()}
        self.load_state_dict(state)  # type: ignore
