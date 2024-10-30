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
from rich.console import Console
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
)

from nerfstudio.field_components.spatial_distortions import (
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field


try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass
CONSOLE = Console(width=120)


def get_normalized_directions(directions: TensorType["bs":..., 3]):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class EDUSField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
            self,
            aabb,
            num_images: int,
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            appearance_embedding_dim: int = 32,
            use_semantics: bool = False,
            use_individual_appearance_embedding: bool = False,
            use_average_appearance_embedding: bool = False,
            inference_dataset="off",
            Optimize_scene=False,
            num_scenes: int = 1,
            spatial_distortion: Optional[SpatialDistortion] = None,
            volume_size=None,
            scale_factor = None,
            bbx_min = None,
            num_neighbour_select = 1,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_individual_appearance_embedding = use_individual_appearance_embedding
        self.use_semantics = use_semantics
        self.inference_dataset = inference_dataset
        self.testset_embedding_index = []
        self.num_scenes = num_scenes
        self.Optimize_scene = Optimize_scene
        self.num_neighbour_select = num_neighbour_select

        self.binocular = True

        self.bounding_min = bbx_min / scale_factor ## fewshot
        self.voxel_size = torch.tensor([0.2,0.2,0.2])  # 80scene 设置体素大小
        self.volume_size = volume_size
        self.scale_factor = scale_factor
        self.feature_dim_in = 16
        self.feature_dim_out = 128

        self.position_encoding = NeRFEncoding(in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True)
       
        # semantics
        mlp_num_layers = 3
        mlp_layer_width = 128

        self.mlp_semantic = tcnn.Network(
            n_input_dims=self.geo_feat_dim,
            n_output_dims=mlp_layer_width,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": mlp_layer_width,
                "n_hidden_layers": mlp_num_layers,
            },
        )


        """ Standard MLP; layer=6 width =128"""
        self.mlp_base = tcnn.Network(
                n_input_dims=self.feature_dim_in + 3*self.num_neighbour_select + self.position_encoding.get_out_dim(),
                n_output_dims=self.feature_dim_out,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.feature_dim_out,
                    "n_hidden_layers": 5,
                },
        )

          ## appearance embedding:
        if self.use_individual_appearance_embedding or self.use_average_appearance_embedding:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
            self.view_head = nn.ModuleList([nn.Linear(self.feature_dim_out + 3 + self.appearance_embedding_dim, 128)])
        else:
            self.view_head = nn.ModuleList([nn.Linear(self.feature_dim_out + 3, 128)])
        self.color_mlp = tcnn.Network(
                n_input_dims=128,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers-1,
                },
            )
        
        self.mlp_density = tcnn.Network(
            n_input_dims=self.feature_dim_in,
            n_output_dims= 1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers,
            },
        )


    def get_grid_coords(self, position_w):
        position_w /= self.scale_factor  ## scale 到 world 系
        pts = position_w - self.bounding_min.to(position_w)
        ## trlinear 插值 // 整除是不对的,会造成 blocky
        x_index = pts[..., 0] / self.voxel_size[0]
        y_index = pts[..., 1] / self.voxel_size[1]
        z_index = pts[..., 2] / self.voxel_size[2]
        """ Normalize the point coordinates to [-1,1]"""

        dhw = torch.stack([x_index, y_index, z_index], dim=1)

        index = dhw.clone().long()
        dhw[..., 0] = dhw[..., 0] / self.volume_size[0] * 2 - 1
        dhw[..., 1] = dhw[..., 1] / self.volume_size[1] * 2 - 1
        dhw[..., 2] = dhw[..., 2] / self.volume_size[2] * 2 - 1
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords, index

    def interpolate_features(self, grid_coords, feature_volume):
        grid_coords = grid_coords[None, None, ...]
        feature = F.grid_sample(feature_volume,
                                grid_coords,
                                mode='bilinear',
                                align_corners=True,
                                )
        return feature

    def get_density_factor_fields(self, ray_samples: RaySamples, scene_id=None, feats_volume=None):
        positions = ray_samples.frustums.get_positions()
        positions_flat = positions.view(-1, 3)

        # feats_volume = self.encoder(self.voxel)
        ## debug
        grid_coords, index = self.get_grid_coords(positions_flat)
        grid_coords = grid_coords.view(*ray_samples.shape[:2], -1)
        index = index.view(*ray_samples.shape[:2],-1)


        feat = self.interpolate_features(grid_coords=grid_coords, feature_volume=feats_volume).permute(3, 4, 1, 0, 2)
        feat = feat.squeeze()

        density_feat = feat

        # h = self.mlp_feature(torch.cat([feat.view(-1,self.feature_dim_in),xyz_embedding],dim=-1))
        density_feat = self.mlp_density(density_feat.view(-1,self.feature_dim_in)).view(*ray_samples.shape[:2], -1)
        # density = self.mlp_density(feat.view(-1, 16)).view(*ray_samples.shape[:2], -1)

        ## convert feat to density and density_embedding
        # density_before_activation, _ = torch.split(density_feat, [1, self.geo_feat_dim], dim=-1)
        # self._density_before_activation = density_before_activation
        density = F.relu(density_feat.to(positions))
        density = torch.nan_to_num(density)

        feat_dict = {
                    #   "semantic_embedding": base_mlp_out,
                      "feature": feat,
        }
        return density, feat_dict

    def get_pos_pred_semantics_KITTI(self,positions,scene_id=0,feats_volume=None):
        positions_flat = positions.view(-1, 3)
        grid_coords, index = self.get_grid_coords(positions_flat)
        shape = (8192,2)
        grid_coords = grid_coords.view(*shape, -1)

        feat = self.interpolate_features(grid_coords=grid_coords, feature_volume=feats_volume).permute(3, 4, 1, 0, 2)
        feat = feat.squeeze()

        ## Regress Density
        density_feat = self.mlp_density(feat.view(-1, self.feature_dim_in)).view(*shape, -1)
 
        ## convert feat to density and density_embedding
        _, semantics_input = torch.split(density_feat, [1, self.geo_feat_dim], dim=-1)
        semantics_input = semantics_input.view(-1, self.geo_feat_dim)
        x = self.mlp_semantic(semantics_input)
        x = torch.nan_to_num(x, nan=0.0)
        assert not torch.any(torch.isnan(x))
        semantics = self.field_head_semantic(x)
        return semantics

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None, scene_id=0,color_2d= None):
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        # d = self.direction_encoding(directions_flat)

        h = density_embedding['feature']
        # semantic_embedding = density_embedding['semantic_embedding']

        """hash encoding"""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        positions_flat = positions.view(-1, 3)
        xyz_embedding = self.position_encoding(positions_flat)


        """ 验证 color volume 的质量"""
        outputs_shape = ray_samples.frustums.directions.shape[:-1]
        input_color = color_2d.reshape(*outputs_shape, -1)


        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)  ##[N_rays, N_samples, 32]
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            elif self.use_individual_appearance_embedding and self.inference_dataset == "trainset":
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance(camera_indices)

            elif self.use_individual_appearance_embedding and self.inference_dataset == "testset":
                Testset_embedding_index = torch.tensor(self.testset_embedding_index).to('cuda')
                test_id = Testset_embedding_index[camera_indices[0][0]]
                test_id = test_id.expand_as(camera_indices)
                ## 双目的 话 -2
                if self.binocular:
                    latent_code = 0.5 * (self.embedding_appearance(test_id) + self.embedding_appearance(test_id - 2))
                    
                else:
                    ## 单目的 话 -1
                    # latent_code = 0.5 * (self.embedding_appearance(test_id) + self.embedding_appearance(test_id - 1))
                    latent_code = self.embedding_appearance(test_id)
                embedded_appearance = torch.ones((*directions.shape[:-1], self.appearance_embedding_dim),
                                                 device=directions.device) * latent_code
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        h = torch.cat(
            [
            input_color.view(-1,3*self.num_neighbour_select),
            h.view(-1,self.feature_dim_in),
            xyz_embedding],dim=-1)

        x = self.mlp_base(h) 

        x = torch.cat([x,
                       directions_flat,
                       embedded_appearance.view(-1,self.appearance_embedding_dim)
                       ],dim=-1)
        x = self.view_head[0](x)
        x = F.relu(x)
        rgb = self.color_mlp(x).view(*outputs_shape, -1).to(directions)

        if torch.isnan(rgb).any():
            rgb = torch.nan_to_num(rgb, nan=0.0)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

    @torch.no_grad()
    def eval_pos_density_factor_fields(self, positions, feats_volume,scene_id=0):
        
        positions_flat = positions.view(-1, 3)
        grid_coords, index = self.get_grid_coords(positions_flat)
        grid_coords = grid_coords.view(1,-1,3)
        feat = self.interpolate_features(grid_coords=grid_coords, feature_volume=feats_volume).permute(3, 4, 1, 0, 2)
        feat = feat.squeeze()

        density_feat = self.mlp_density(feat.view(-1,self.feature_dim_in))
        
        density = F.relu(density_feat.to(positions))
        density = torch.nan_to_num(density)

        return density


    @torch.no_grad()
    def eval_pos_pred_semantics_KITTI(self, positions, feats_volume, scene_id=0):
        positions_flat = positions.view(-1, 3) 
        grid_coords, index = self.get_grid_coords(positions_flat)
        grid_coords = grid_coords.view(1,-1,3)
        
        feat = self.interpolate_features(grid_coords=grid_coords, feature_volume=feats_volume).permute(3, 4, 1, 0, 2)
        feat = feat.squeeze()

        ## Regress Density
        density_feat = self.mlp_density(feat.view(-1, 16))
 
        ## convert feat to density and density_embedding
        _, semantics_input = torch.split(density_feat, [1, self.geo_feat_dim], dim=-1)
        semantics_input = semantics_input.view(-1, self.geo_feat_dim)
        x = self.mlp_semantic(semantics_input)
        x = torch.nan_to_num(x, nan=0.0)
        assert not torch.any(torch.isnan(x))
        semantics = self.field_head_semantic(x.float())

        semantic_labels = torch.argmax(torch.nn.functional.softmax(semantics, dim=-1), dim=-1)
        return semantic_labels
    

if __name__ == "__main__":
    voxel = np.load("/data1/smiao/multiscene_kitti_seg/few_show/voxel/train_pose_fewshow.npy")
    voxel = torch.from_numpy(voxel).to('cuda')
    coord = voxel.permute(3, 0, 1, 2).float()
    voxel_size = 0.1  # 设置体素大小

    batch_size = 1
    coor = coord[None, ...]

    net = VoxelNet(in_channels=3).to('cuda')
    ans = net.forward(coor)
    print(ans.shape)

    exit()

