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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle

from nerfstudio.field_components.point_encoder import VoxelEncoder
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.field_components.spade_generator import FeatureVolumeGenerator
from nerfstudio.fields.neuralpoint_fields import EDUSField
from nerfstudio.model_components.losses import (
    MSELoss,
    Hierarchy_distortion_loss,
    fg_bg_entroyloss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler, HierarchyImportanceSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from rich.console import Console
from einops import rearrange
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
from nerfstudio.model_components.LidarRay import LidarRay
import torch.nn.functional as F
from nerfstudio.fields.background_model import BackgroundFiled
import torch.nn as nn
import copy
from nerfstudio.model_components.projection import Projector

CONSOLE = Console(width=120)

@dataclass
class EDUSConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: EDUSModel)
    far_plane_bg: float = 5.0
    """How far along the ray to stop sampling of the background model."""
    background_color: Literal["random", "last_sample", "white", "black"] = "black"
    """Whether to randomize the background color."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    use_individual_appearance_embedding: bool = True
    """Whether to use individual appearance embedding or zeros for inference."""
    inference_dataset: Literal["off", "trainset", "testset"] = "off"
    """In inference, we need to interpoate the appearance embedding of the input image.
    if trainset, we retrive each appearance embedding of the frame;
    if testset, we need to interpolate the appearance embedding from the training inputs;
    """
    sky_model: bool = True
    """When model the infinate sky"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for backgound"""
    enable_lidar: bool = True
    """ Use lidar supervision in the traning stage"""


class EDUSModel(Model):

    config: EDUSConfig

    def populate_modules(self, volume_dict=None,lidar_data=None,device=None):
        """Set the fields and modules."""
        super().populate_modules()

        self.scene_contraction = SceneContraction(order=float("inf"))
        self.base_conf = OmegaConf.load(self.config_file)
        assert volume_dict is not None
        ## Encoder 3D voxel
        voxel_bank = volume_dict['data']
        volume_res = (voxel_bank.shape[2],voxel_bank.shape[1],voxel_bank.shape[3])
        self.encoder = FeatureVolumeGenerator(init_res=8,
                                              volume_res= volume_res, ## YXZ
                                              out_channel=16,
                                              input_channels=3,
                                              z_dim_oasis=0).to(device)
        self.num_scenes = self.base_conf["training_scenes"]

        ## Input noisy RGB volume
        self.voxel = torch.from_numpy(voxel_bank)
        self.voxel = self.voxel.permute(0,4,1,2,3).float().to('cuda')
        self.scale_factor = volume_dict['scale_factor']

        # Fields
        self.field = EDUSField(
            self.scene_box.aabb,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_individual_appearance_embedding=self.base_conf['individual_embedding'],
            use_average_appearance_embedding = self.base_conf['average_embedding'],
            inference_dataset=self.config.inference_dataset,
            volume_size=self.voxel.shape[2:],
            scale_factor = volume_dict['scale_factor'],
            bbx_min = self.scene_box.aabb[0],
            num_neighbour_select= self.base_conf['data_manager']['num_neighbour_select'],
        )

         # renderers
        self.field_background = BackgroundFiled(
                self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_average_appearance_embedding= True,
        )
            

        self.sampler = HierarchyImportanceSampler(
            num_samples=self.base_conf['sampling']['n_sampling'],
            num_samples_importance=self.base_conf['sampling']['n_importance'],
            num_upsample_steps=self.base_conf['sampling']['up_sample_steps'],
        )

        self.projector = Projector(intrinsics=self.intrinsics)


        ## Load pretrained Checkpoint 
        if self.base_conf['Optimize_scene']:
            ckpt_path = self.base_conf['ckpt_path']
            self.encoder.load_state_dict(torch.load(ckpt_path)['encoder'],strict=False)

            ## filed checkpoint
            state_dict = torch.load(ckpt_path)['field']
            state_dict.pop('aabb')
            """NOTE: Add Appearance Embedding
            We designate the 1st training scene appearance embedding for the novel scenes;
            This appearance embedding vector can be further finetuned in the finetuning stage
            """
            if state_dict['embedding_appearance.embedding.weight'] is not None:
                original_weight = state_dict['embedding_appearance.embedding.weight']
                adjusted_weight = original_weight[:self.num_train_data] 
                state_dict['embedding_appearance.embedding.weight'] = adjusted_weight
            self.field.load_state_dict(state_dict,strict=False)

            ## background appearance embedding
            bg_state_dict = torch.load(ckpt_path)['bg_field']
            if bg_state_dict['embedding_appearance.embedding.weight'] is not None:
                original_weight = bg_state_dict['embedding_appearance.embedding.weight']
                adjusted_weight = original_weight[:self.num_train_data] 
                bg_state_dict['embedding_appearance.embedding.weight'] = adjusted_weight
            bg_state_dict.pop('aabb')
            self.field_background.load_state_dict(bg_state_dict,strict=False)

            CONSOLE.print(f"[bold red] Load Pretrained Model {ckpt_path} ！\n")
        else:
            CONSOLE.print("[bold red]Not Load Pretrained Model! Training from scratch.  \n")
        
        ## Random Mask Points
        self.maskpoints = self.base_conf["Mask_points"]
        self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)

        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )

        """ Init the feature volume in test-time optization """
        self.init_volume = self.base_conf['init_volume']
        if self.init_volume:
            CONSOLE.print("[green] Initalize the feature volume \n")
            voxel = self.voxel.to("cuda")
            voxel = rearrange(voxel, 'B C W H D -> B C D H W')
            feat_volume = self.encoder(voxel) 
            feat_volume = rearrange(feat_volume, 'B C H W D -> B C W H D')
            self.volume = RefVolume(feat_volume.detach()).to("cuda")
            self.feat_volume = self.volume.data
        else:
            self.feat_volume = None

         # Collider
        self.collider = AABBBoxCollider(scene_box=self.scene_box)    
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer('expected')

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()


        ## 3D Lidar surprision
        if lidar_data is not None:
            self.LidarRay = LidarRay(
                LidarRays=lidar_data['lidarRays'],
                scale_factor= lidar_data['scale_factor'],
                device=device
            )
        
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        if self.init_volume:
            param_groups['voxel_encoder'] = list(self.volume.parameters())
        else:
            param_groups['voxel_encoder'] = list(self.encoder.parameters())
   
        param_groups['field_background'] = list(self.field_background.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle, mask):

        ray_samples, _ = self.sampler(ray_bundle, density_fn=self.field.get_density_factor_fields,
                                      feat_volume=self.feat_volume)
        
        """ Retrive 2D feature from the source image"""
        assert ray_bundle.source_poses is not None
        n_views = int(ray_bundle.source_poses.shape[0]-1) # type: ignore
        sampled_rgb = self.projector.compute(xyz = ray_samples, 
                                            train_imgs = ray_bundle.source_images,
                                            train_cameras = ray_bundle.source_poses[:n_views,...],
                                            train_depths = ray_bundle.source_depth * self.scale_factor,
                                            )  # [N_rays, N_samples, N_views, 3] 
                                        

        field_outputs = self.field(ray_samples, compute_normals=False, scene_id=ray_bundle.scene_id[0], # type: ignore
                                   feature_volume=self.feat_volume,color_volume = sampled_rgb)
        
        field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights, _ = ray_samples.get_weights_and_transmittance(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights, bg_color=None, fg_accumulation=None)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # Run Generalizable background model
        if  self.config.sky_model:
            bg_transmittance = accumulation

            # Modify the RayBundle nears and fars 
            dv_raybundle = copy.deepcopy(ray_bundle)
            dv_raybundle.nears = ray_bundle.fars 
            dv_raybundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ## Sample points outside the bounding box
            ray_samples_bg = self.sampler_bg(dv_raybundle)
            sampled_bg_rgb = self.projector.compute(xyz = ray_samples_bg,
                                                    train_imgs = ray_bundle.source_images,
                                                    train_cameras = ray_bundle.source_poses[:n_views,...])  # [N_rays, N_samples, N_views, 3]
 
            field_outputs_bg = self.field_background.background_generate(ray_samples=ray_samples_bg,
                                                                         sampled_rgb=sampled_bg_rgb,
                                                                         scene_id = ray_bundle.scene_id[0]) # type: ignore
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])
          
            rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], 
                                       weights=weights_bg,
                                       bg_color=field_outputs_bg[FieldHeadNames.BG_RGB],
                                       fg_accumulation = accumulation)

            # merge background color to forgound color
            rgb = rgb + (1.0-bg_transmittance) * rgb_bg
            accumulation_bg = self.renderer_accumulation(weights=weights_bg)
        else:
            accumulation_bg = None


        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "weight": weights,
            "ray_samples": ray_samples,
            "dv_accumulation": accumulation_bg,
        }


        if not self.training:
             depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg) # type: ignore
             outputs["bg_accumulation"] = (1.0-bg_transmittance) * accumulation_bg
             outputs['bg_depth'] = (1.0-bg_transmittance) * depth_bg
             outputs['bg_rgb'] = (1.0-bg_transmittance) * torch.sum(weights_bg * field_outputs_bg[FieldHeadNames.RGB], dim=-2)
             outputs['sky'] = field_outputs_bg[FieldHeadNames.BG_RGB]* (1- torch.clamp(accumulation_bg + bg_transmittance,min=0,max=1) )

        return outputs
    

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = Hierarchy_distortion_loss(outputs["weight"], outputs["ray_samples"])

        if self.config.enable_lidar:
            metrics_dict["urf_sigma"] = self.LidarRay.depth_sigma
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None, step=0, ray_bundle=None):
        loss_dict = {}
        image = batch["image"].to(self.device)

        sky_loss_mult = self.base_conf['loss_coeff'].sky_loss_coffe
        fgbg_entroyloss_mult = self.base_conf['loss_coeff'].fgbg_entropyloss
        URF_loss_mult = self.base_conf['loss_coeff'].URF_entropyloss

        if self.config.sky_model:
            ## apply BCE loss to segment the sky region
            segmap = batch['mask'].clone() ## 1 channel 
            sky_mask = segmap == 255

            loss_dict["rgb_loss"] = self.rgb_loss(outputs["rgb"], image)
            loss_dict["sky_loss"] = sky_loss_mult * F.binary_cross_entropy_with_logits(
                torch.clamp(outputs['accumulation']+ outputs['dv_accumulation'] ,min=1e-3,max=1-1e-3) , sky_mask.float())
        else:
            ## no sky model
            segmap = batch['mask'].clone() ## 1 channel 
            sky_mask = segmap == 255
            loss_dict["rgb_loss"] = self.rgb_loss(outputs["rgb"], image)
            loss_dict["sky_loss"] = sky_loss_mult * F.binary_cross_entropy_with_logits(
                outputs['accumulation'].clip(1e-3, 1.0 - 1e-3) + outputs['dv_accumulation'].clip(1e-3, 1.0 - 1e-3) , sky_mask.float())

        if self.training:
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            loss_dict['fg_bg_entroy'] = fgbg_entroyloss_mult * fg_bg_entroyloss(
                        mask_pred_cr=outputs["accumulation"].clip(1e-3, 1.0 - 1e-3).squeeze())

            if self.config.enable_lidar and step > 30:
                """ 3D lidar supervision"""
                loss_dict['URF'] = URF_loss_mult * self.cal_urf_loss(scene_id=ray_bundle.scene_id[0])
    
        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict
    
    def cal_urf_loss(self,scene_id=0):
        lidar_raysamples = self.LidarRay.get_lidar_samples(scene_id=scene_id)
        lidar_denity,_ = self.field.get_density_factor_fields(ray_samples=lidar_raysamples,
                                                                scene_id=scene_id,
                                                                feats_volume=self.feat_volume)
        lidar_weights = lidar_raysamples.get_weights(lidar_denity)
        depth = self.renderer_depth(weights=lidar_weights, ray_samples=lidar_raysamples)
        URF_loss = self.LidarRay.cal_urf_loss(lidar_samples=lidar_raysamples,weights=lidar_weights,predicted_depth=depth)
        return URF_loss


    def knn(self,voxel):
        voxel = voxel[0].permute(1,2,3,0)
        mask_size = (40, 40, 60, 1)
        start_position = (
        torch.randint(0, voxel.shape[0] - mask_size[0] + 1, (1,)).item(),
        torch.randint(0, voxel.shape[1] - mask_size[1] + 1, (1,)).item(),
        torch.randint(0, voxel.shape[2] - mask_size[2] + 1, (1,)).item(),
        )
        mask = torch.ones_like(voxel)
        mask[start_position[0]: start_position[0] + mask_size[0],
            start_position[1]: start_position[1] + mask_size[1],
            start_position[2]: start_position[2] + mask_size[2],
            ] = 0
        voxel = voxel*mask[...,:1]
        return voxel.permute(3,0,1,2)[None,...]

class RefVolume(nn.Module):
    def __init__(self, volume):
        super(RefVolume, self).__init__()
        self.data = nn.Parameter(volume)
    
    def forward(self):
        exit()
    
    
    




