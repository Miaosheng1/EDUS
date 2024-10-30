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

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import (
    MSELoss,
    Hierarchy_distortion_loss,
    ScaleAndShiftInvariantLoss,
)
from nerfstudio.model_components.ray_samplers import HierarchyImportanceSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
import torch.nn.functional as F


@dataclass
class GVSNerfModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: GVSNerfModel)
    near_plane: float = 0.01
    """How far along the ray to start sampling."""
    far_plane: float = 8
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "white", "black"] = "white"
    """Whether to randomize the background color."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 1024
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int] = (256, 96)
    """Number of samples per ray for the proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 1000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.001
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed noramls."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    use_individual_appearance_embedding: bool = True
    """Whether to use individual appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    background_mlp: bool = True
    """Whether to predict normals or not."""
    inference_dataset: Literal["off", "trainset", "testset"] = "off"
    """When in inference, which dataset will be loaded."""
    sky_model:bool = True
    """When model the infinate sky"""


class GVSNerfModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self,occ3d_dict=None):
        """Set the fields and modules."""
        super().populate_modules()

        scene_contraction = SceneContraction(order=float("inf"))
        self.base_conf = OmegaConf.load(self.config_file)

        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_bg=self.config.background_mlp,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_individual_appearance_embedding=self.config.use_individual_appearance_embedding,
            num_semantic_classes= occ3d_dict["num_class"] if occ3d_dict is not None else 100,
            inference_dataset = self.config.inference_dataset,
            num_scenes= self.base_conf["training_scenes"],
            base_conf=self.base_conf['field'],
            Optimize_scene= self.base_conf['Optimize_scene']
        )


        self.sampler = HierarchyImportanceSampler(
            num_samples = self.base_conf['sampling']['n_sampling'],
            num_samples_importance = self.base_conf['sampling']['n_importance'],
            num_upsample_steps  = self.base_conf['sampling']['up_sample_steps'],
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        if occ3d_dict is not None:
            self.Occ3d = Occ3d(occ_gt=  occ3d_dict['occ3d_voxel'],
                               bounding_box_min = occ3d_dict['bounding_box_min'],
                               bounding_box_max = occ3d_dict['bounding_box_max'],
                               voxel_size= occ3d_dict['voxel_size'],
                               num_calss=occ3d_dict['num_class'],
                               free_state=occ3d_dict['Free_state'])

        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        # self.renderer_depth = DepthRenderer(method='expected')
        self.renderer_depth = DepthRenderer('expected')  ## 默认为median
        self.renderer_normals = NormalsRenderer()
        self.renderer_semantics = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean",ignore_index=23)
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        ## filename_index
        self.img_filename = None
        

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups


    def get_outputs(self, ray_bundle: RayBundle,mask):
        
        self.field.set_coeff_optimizer(scene_id=ray_bundle.scene_id[0])

        ray_samples,_ = self.sampler(ray_bundle,density_fn = self.field.get_density_factor_fields)
        field_outputs = self.field(ray_samples, compute_normals=False,scene_id = ray_bundle.scene_id[0])

        weights,transmittance = ray_samples.get_weights_and_transmittance(field_outputs[FieldHeadNames.DENSITY])
 
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights,bg_color = field_outputs[FieldHeadNames.BG_RGB],mask = mask)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            # "transmittance":transmittance,
            "depth": depth,
            "weight":weights,
            "density":field_outputs[FieldHeadNames.DENSITY],
            "ray_samples":ray_samples,
        }
        if field_outputs[FieldHeadNames.SEMANTICS] is not None:
            field_outputs[FieldHeadNames.SEMANTICS] = torch.nan_to_num(field_outputs[FieldHeadNames.SEMANTICS], nan=0.0)
            outputs["semantics"] = self.renderer_semantics(field_outputs[FieldHeadNames.SEMANTICS], weights=weights)

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            # metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
            metrics_dict["distortion"] = Hierarchy_distortion_loss(outputs["weight"], outputs["ray_samples"])
            
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None,step=0,ray_bundle=None):
        loss_dict = {}
        image = batch["image"].to(self.device)

        semantic_loss_mult = self.base_conf['loss_coeff'].semantic_loss_coffe
        sky_loss_mult = self.base_conf['loss_coeff'].sky_loss_coffe
        
        segmap = batch['mask'].clone()
        sky_mask = torch.logical_not(segmap == 23)

        if self.config.sky_model:
            ## sky model 必须是 BCE loss 如果没有 density loss 和 weight loss 的情况下
            color_error = image - outputs["rgb"]
            loss_dict["rgb_loss"] = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / 4096.0
            loss_dict["sky_loss"] =  sky_loss_mult * F.binary_cross_entropy_with_logits(outputs['accumulation'].clip(1e-3, 1.0-1e-3),sky_mask.float())
        else:
            ## no sky model
            loss_dict["rgb_loss"] = self.rgb_loss(outputs["rgb"],image)

       
        """  mono depth loss (monosdf paper)
        if 'depth' in batch and step % 2 == 0:
            remain_mask = sky_mask.reshape(1, 32, -1).bool().squeeze(dim=0)
            kernel_size = 8
            remain_mask = kornia.morphology.erosion(remain_mask[None,None].float(), 
                                                    torch.ones([kernel_size,kernel_size], device=remain_mask.device))[0,0].bool()
            depth_gt = batch["depth"].to(self.device)[..., None]
            depth_pred = outputs["depth"]
            loss_dict["monodepth_loss"] = 0.1*self.depth_loss(depth_pred.reshape(1, 32, -1), 
                                                                (depth_gt * 2 + 0.5).reshape(1, 32, -1), remain_mask[None,...])
        """
    
        """          Monodepth depth ranking Loss
        depth_gt = batch["depth"].to(self.device)
        loss_dict['monodepth'] = 0.1 * depth_ranking_loss(rendered_depth=depth_pred, gt_depth=depth_gt)
        """
        

        if self.training:
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

            # if step % 10 == 0:
            #     # if self.covert2trainId:
            #     # segmap = torch.from_numpy(assigntrainID(segmap.reshape(-1))).to(self.device)
            #     pred_semantics = outputs["semantics"]
            #     gt_semantics = segmap
            #     loss_dict["semantics_loss"] = semantic_loss_mult * self.cross_entropy_loss(pred_semantics,
            #                                                                                gt_semantics.long().squeeze(dim=-1))

             ## Lidar depth loss
            # if 'depth' in batch:
            #     mask = torch.nonzero(batch['depth'][..., 0]-8,as_tuple=False)[...,0]
            #     depth_gt = batch["depth"].to(self.device)[mask]
                
            #     loss_dict["depth_loss"] = depth_loss(weights=outputs['weight'][mask],
            #                                     ray_samples=outputs['ray_samples'][mask],
            #                                     termination_depth = depth_gt,
            #                                     predicted_depth=outputs["depth"][mask],
            #                                         sigma=0.01,
            #                                         is_euclidean=True,
            #                                         directions_norm=None,
            #                                         depth_loss_type=DepthLossType.DS_NERF,
            #                                     )
            
            if hasattr(self,"Occ3d") :
                ## semantic loss
                # loss_dict["semantic_loss"] = semantic_loss_mult * self.cal_volume_semantics(ray_samples=outputs['ray_samples'],
                #                                                                             step=step,
                #                                                                             scene_id=ray_bundle.scene_id[0]
                #                                                                     )  
                
                ## Add Density Loss
                density_loss,weight_loss,_ = self.cal_density_loss(ray_bundle=ray_bundle)
                loss_dict['density_loss'] = self.base_conf['loss_coeff'].denisty_loss_coff * density_loss
                ## 这个weight loss 会让Ray 在bbx 内部长一个 peak
                loss_dict['weight_loss'] = 1 * F.l1_loss(weight_loss, torch.ones_like(weight_loss))

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

        # normals to RGB for visualization. TODO: use a colormap
        if "normals" in outputs:
            images_dict["normals"] = (outputs["normals"] + 1.0) / 2.0
        if "pred_normals" in outputs:
            images_dict["pred_normals"] = (outputs["pred_normals"] + 1.0) / 2.0

        # for i in range(self.config.num_proposal_iterations):
        #     key = f"prop_depth_{i}"
        #     prop_depth_i = colormaps.apply_depth_colormap(
        #         outputs[key],
        #         accumulation=outputs["accumulation"],
        #     )
        #     images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
    
    
    
    # def cal_volume_semantics(self,ray_bundle,sample_num_samples=50,step=0):
    #     ray_bundle = self.Occ3d.collider_near_far(ray_bundle[::50])
    #     ray_samples = self.Occ3d.uniform_sampler(ray_bundle, num_samples=sample_num_samples)
    #     position = ray_samples.frustums.get_positions().reshape(-1,3)
        
    #     BATCH_SIZE = 8192 * 2
    #     selected_index = torch.randint(low=0, high=position.shape[0], size=[BATCH_SIZE])
    #     position = position[selected_index]

    #     ## waymo
    #     # gt_semantics = self.Occ3d.get_occ3d_gt(position_w=position.clone())
    #     # pred_semantics = self.field.get_pos_pred_semantics(positions=position.clone())

    #     ## KITTI-360 仅仅对 非空的点进行 约束
    #     gt_semantics =  self.Occ3d.get_occ3d_gt_KITTI(position_w=position.clone(),scene_id=ray_bundle.scene_id[0])
    #     pred_semantics = self.field.get_pos_pred_semantics_KITTI(positions=position.clone(),scene_id=ray_bundle.scene_id[0])
    #     Nonzero_index = torch.nonzero(gt_semantics).squeeze(dim=1)

    #     ## 得到 GT_labels
    #     # labels = torch.argmax(torch.nn.functional.softmax(gt_semantics, dim=-1), dim=-1)
    #     # non_empty_region = torch.logical_not(gt_semantics == 15)
    #     # surface_index = torch.nonzero(non_empty_region)

    #     semantic_loss = self.Occ3d.cross_entropy_loss(pred_semantics[Nonzero_index], gt_semantics[Nonzero_index])
    #     return semantic_loss
    

    def cal_volume_semantics(self,ray_samples=None,step=0,scene_id = 0):
        assert ray_samples is not None
        with torch.no_grad():
            position = ray_samples.frustums.get_positions().reshape(-1,3)
        
        BATCH_SIZE = 8192 * 2
        selected_index = torch.randint(low=0, high=position.shape[0], size=[BATCH_SIZE])
        position = position[selected_index]

        ## waymo
        # gt_semantics = self.Occ3d.get_occ3d_gt(position_w=position.clone())
        # pred_semantics = self.field.get_pos_pred_semantics(positions=position.clone())

        ## KITTI-360 仅仅对 非空的点进行 约束
        gt_semantics =  self.Occ3d.get_occ3d_gt_KITTI(position_w=position.clone(),scene_id=scene_id)
        pred_semantics = self.field.get_pos_pred_semantics_KITTI(positions=position.clone(),scene_id=scene_id)
        Nonzero_index = torch.nonzero(gt_semantics).squeeze(dim=1)

        ## 得到 GT_labels
        # labels = torch.argmax(torch.nn.functional.softmax(gt_semantics, dim=-1), dim=-1)
        # non_empty_region = torch.logical_not(gt_semantics == 15)
        # surface_index = torch.nonzero(non_empty_region)

        semantic_loss = self.Occ3d.cross_entropy_loss(pred_semantics[Nonzero_index], gt_semantics[Nonzero_index])
        return semantic_loss
    
   
    # def get_lidar_depth(self,ray_bundle,sample_num_samples=256):
    #     ray_bundle = self.Occ3d.collider_near_far(ray_bundle)
    #     ray_samples = self.Occ3d.uniform_sampler(ray_bundle, num_samples=sample_num_samples)
    #     position = ray_samples.frustums.get_positions().reshape(-1,3)

    #     gt_semantics = self.Occ3d.get_occ3d_gt_KITTI(position_w=position.clone(),scene_id=ray_bundle.scene_id[0])

    #     non_empty_region = torch.logical_not(gt_semantics == 0).reshape(*ray_samples.shape[:2])
    #     surface_index = torch.argmax(non_empty_region.int(),dim=-1)

    #     ## Verified Implentation
    #     depth = (ray_samples.frustums.starts + ray_samples.frustums.ends)/2
    #     depth = depth.squeeze(dim=-1)
    #     gt_depth = torch.tensor([10]).repeat(len(ray_samples)).to(depth)
    #     valid_mask = surface_index > 0
    #     valid_ray = torch.nonzero(surface_index)
    #     if valid_ray.shape[0] == 0:
    #         return gt_depth,valid_mask
    #     gt_depth[valid_ray] = depth[valid_ray,surface_index[valid_ray]]

    #     return gt_depth.unsqueeze(dim=-1),valid_mask.unsqueeze(dim=-1)
    
    def cal_density_loss(self,ray_bundle, sample_num_samples = 40):
        ray_bundle = self.Occ3d.collider_near_far(ray_bundle)
        ray_samples = self.Occ3d.uniform_sampler(ray_bundle, num_samples=sample_num_samples)
        position = ray_samples.frustums.get_positions().reshape(-1,3)
      
        # gt_semantics =  self.Occ3d.get_occ3d_gt(position_w=position.clone())
        gt_semantics =  self.Occ3d.get_occ3d_gt_KITTI(position_w=position.clone(),scene_id=ray_bundle.scene_id[0])

        ## 找出整条光线为空的 （这些不加loss）
        empty_region = (gt_semantics == self.Occ3d.free_state)
        Non_empty_region = torch.logical_not(empty_region.view(*ray_samples.shape[:2]).contiguous())
        Non_empty_ray = torch.sum(Non_empty_region,dim=-1)
        Valid_occ3d_ray = (Non_empty_ray != 0)

        ## valid_ray_samples 是有效光线上的采样点
        valid_ray_gt_semantics = gt_semantics.view(*ray_samples.shape[:2],-1).contiguous()[Valid_occ3d_ray,...]
        valid_ray_samples = ray_samples[Valid_occ3d_ray]

        ## 非空光线上的empty 点加上Loss
        valid_point_mask =  (valid_ray_gt_semantics ==  self.Occ3d.free_state ).flatten()
        ## 如果一条光线 是全 empty，则证明Occ3d 有问题，不应该加入到 density Loss 当中
        # Valid_ray_denity = self.field.get_density(ray_samples=valid_ray_samples)[0]  ## waymo
        Valid_ray_denity = self.field.get_density_factor_fields(ray_samples=valid_ray_samples,scene_id=ray_bundle.scene_id[0])[0]
        
        ## 且这些valid ray 在bbx 内部的accumulation 约束到 1
        Valid_weight = valid_ray_samples.get_weights(Valid_ray_denity)
        Valid_denity = Valid_ray_denity.flatten()[valid_point_mask]

        return torch.mean(Valid_denity) , torch.sum(Valid_weight,dim=-2), valid_point_mask
    
    """ Return a valid  Ray Mask whether intersected with Occ3d"""
    def is_intersenction_with_occ3d(self,ray_bundle, sample_num_samples=80):
        import copy
        ray_bundle = copy.deepcopy(ray_bundle)
        ray_bundle = self.Occ3d.collider_near_far(ray_bundle)
        ray_samples = self.Occ3d.uniform_sampler(ray_bundle, num_samples=sample_num_samples)
        position = ray_samples.frustums.get_positions().reshape(-1,3)
        # gt_semantics =  self.Occ3d.get_occ3d_gt(position_w=position.clone())
        gt_semantics =  self.Occ3d.get_occ3d_gt_KITTI(position_w=position.clone(),scene_id=ray_bundle.scene_id[0])

        ## 找出整条光线为空的 （这些不加loss）
        empty_region = (gt_semantics == self.Occ3d.free_state)
        Non_empty_region = torch.logical_not(empty_region.view(*ray_samples.shape[:2]).contiguous())
        Non_empty_ray = torch.sum(Non_empty_region,dim=-1)
        Valid_occ3d_ray = (Non_empty_ray != 0)

        return Valid_occ3d_ray


