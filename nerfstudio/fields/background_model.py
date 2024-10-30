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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field
from torch.nn import ParameterList

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


def get_normalized_directions(directions: TensorType["bs":..., 3]):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class BackgroundFiled(Field):
    """ Generalizable Background & Sky Model

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        appearance_embedding_dim: dimension of appearance embedding
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
        use_pred_sky: Model the Sky for EDUS
    """

    def __init__(
        self,
        aabb,
        num_images: int,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        appearance_embedding_dim: int = 32,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_pred_sky: bool = True,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 8},
        )

        self.mlp_base = tcnn.Network(
            n_input_dims= self.position_encoding.n_output_dims + 9,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 4,
            },
        )

        self.mlp_head = tcnn.Network(
            n_input_dims=3 + self.geo_feat_dim + self.appearance_embedding_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

        ## Genearalizable Sky Model
        self.use_pred_sky = use_pred_sky
        if self.use_pred_sky:
            self.mlp_pred_backgroud = ParameterList(
                tcnn.Network(
                    n_input_dims=3 + 9,
                    n_output_dims=3,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "Sigmoid",
                        "n_neurons": 64,
                        "n_hidden_layers": 1,
                    },
                ) for _ in range(1)
            ) 

    def get_density_factor_fields(self, ray_samples: RaySamples, sampled_rgb=None):
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)

        """Position Encoding"""
        xyz_embedding = self.position_encoding(positions_flat)
        x = torch.cat([xyz_embedding,sampled_rgb.view(-1,9)],dim=1)
        h = self.mlp_base(x).view(*ray_samples.frustums.shape, -1)

        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # choose an density function is very improtant for foreground/background decomposition.
        # trunc_exp improve the model capacity of bakcground。 
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None, scene_id=0,sampled_rgb= None):
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        outputs_shape = ray_samples.frustums.directions.shape[:-1]
        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        ## Image-based Rendering for background model
        h = torch.cat(
            [
                directions_flat,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        if self.use_pred_sky:
            dirs = directions_flat.view(*outputs_shape, -1)[:,0,:]
            sky_sampled_color = sampled_rgb.view(*outputs_shape, -1)[:,-1,:]
            bg_h = torch.cat(
                [
                    sky_sampled_color,
                    dirs,
                ],
                dim=-1
            )
            bg_rgb = self.mlp_pred_backgroud[0](bg_h).to(directions)
            outputs.update({FieldHeadNames.BG_RGB: bg_rgb})
        return outputs
