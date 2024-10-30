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
Code for sampling pixels.
"""

import random
from typing import Dict

import torch

from nerfstudio.utils.images import BasicImages


def collate_image_dataset_batch(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Operates on a batch of images and samples pixels to use for generating rays.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    # only sample within the mask, if the mask is in the batch
    # if "mask" in batch:
    #     nonzero_indices = torch.nonzero(batch["mask"][..., 0].to(device), as_tuple=False)
    #     chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_per_batch)
    #     indices = nonzero_indices[chosen_indices]
    # else:
    
    indices = torch.floor(
        torch.rand((num_rays_per_batch, 3), device=device)
        * torch.tensor([num_images, image_height, image_width], device=device)
    ).long()

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key not in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points") and value is not None
    }

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    if "sparse_sfm_points" in batch:
        collated_batch["sparse_sfm_points"] = batch["sparse_sfm_points"].images[c[0]]

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]
    scene_id = torch.tensor([[0]]).to('cuda')
    return collated_batch,scene_id


## GVSNerf 从 Random Scene_id序列中 选择 一个 进行学习
def collate_imagebatch_per_step(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False, 
                                num_scenes: int = 10, num_imgs_per_scene: int = 39):
    device = batch["image"].device
    cuda_device = batch["image_idx"].device

    num_images, image_height, image_width, _ = batch["image"].shape
    current_scene_idx = torch.randint(low=0,high= num_scenes ,size=(1,)).to(cuda_device)
    # current_scene_idx = torch.tensor([1]).to('cuda')
    c = torch.where((batch["image_idx"] >= num_imgs_per_scene* current_scene_idx) &  \
                    (batch["image_idx"] < (current_scene_idx+1)*num_imgs_per_scene))[0]

    indices = torch.floor(
        torch.rand((num_rays_per_batch, 3), device=device)
        * torch.tensor([num_images, image_height, image_width], device=device)
    ).long()

    random_indices = torch.randint(0, c.shape[0], (num_rays_per_batch,)).to(cuda_device)
    c = c[random_indices]
    indices[:,0] = c
    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    
    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key not in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points") and value is not None
    }

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    if "sparse_sfm_points" in batch:
        collated_batch["sparse_sfm_points"] = batch["sparse_sfm_points"].images[c[0]]

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]  ## 因为这里的 batch_idx 必须和  Pose idx 对齐才行
    collated_batch["indices"] = indices  # with the abs camera indices
   
    return collated_batch,current_scene_idx


## GVSNerf 从 Random Scene_id序列中 选择 single Image 和 source image
def collate_single_image_per_step(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False, 
                                num_scenes: int = 10, num_imgs_per_scene: int = 39):
    device = batch["image"].device
    cuda_device = batch["image_idx"].device

    num_images, image_height, image_width, _ = batch["image"].shape
    current_scene_idx = torch.randint(low=0,high= num_scenes ,size=(1,)).to(cuda_device)

    c = torch.where((batch["image_idx"] >= num_imgs_per_scene* current_scene_idx) &  \
                    (batch["image_idx"] < (current_scene_idx+1)*num_imgs_per_scene))[0]

    indices = torch.floor(
        torch.rand((num_rays_per_batch, 3), device=device)
        * torch.tensor([num_images, image_height, image_width], device=device)
    ).long()

    random_indices = torch.randint(0, c.shape[0], (1,)).to(cuda_device)
    c = c[random_indices]
    indices[:,0] = c
    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    
    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key not in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points") and value is not None
    }

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape


    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]  ## 因为这里的 batch_idx 必须和  Pose idx 对齐才行
    collated_batch["indices"] = indices  # with the abs camera indices
   
    return collated_batch,current_scene_idx



def collate_image_dataset_batch_list(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
    a list.

    We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
    The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
    since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    device = batch["image"][0].device
    num_images = len(batch["image"])

    # only sample within the mask, if the mask is in the batch
    all_indices = []
    all_images = []
    all_fg_masks = []

    if "mask" in batch:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            # nonzero_indices = torch.nonzero(batch["mask"][i][..., 0], as_tuple=False)
            nonzero_indices = batch["mask"][i]

            chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_in_batch)
            indices = nonzero_indices[chosen_indices]
            indices = torch.cat([torch.full((num_rays_in_batch, 1), i, device=device), indices], dim=-1)
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
            if "fg_mask" in batch:
                all_fg_masks.append(batch["fg_mask"][i][indices[:, 1], indices[:, 2]])

    else:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            image_height, image_width, _ = batch["image"][i].shape
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            indices = torch.floor(
                torch.rand((num_rays_in_batch, 3), device=device)
                * torch.tensor([1, image_height, image_width], device=device)
            ).long()
            indices[:, 0] = i
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
            if "fg_mask" in batch:
                all_fg_masks.append(batch["fg_mask"][i][indices[:, 1], indices[:, 2]])

    indices = torch.cat(all_indices, dim=0)

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key != "image_idx"
        and key != "image"
        and key != "mask"
        and key != "fg_mask"
        and key != "sparse_pts"
        and value is not None
    }

    collated_batch["image"] = torch.cat(all_images, dim=0)
    if len(all_fg_masks) > 0:
        collated_batch["fg_mask"] = torch.cat(all_fg_masks, dim=0)

    if "sparse_pts" in batch:
        rand_idx = random.randint(0, num_images - 1)
        collated_batch["sparse_pts"] = batch["sparse_pts"][rand_idx]

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


class PixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample(self, image_batch: Dict,step):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictioary so we don't modify the original
            pixel_batch = collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], BasicImages):
            image_batch = dict(image_batch.items())  # copy the dictioary so we don't modify the original
            image_batch["image"] = image_batch["image"].images
            if "mask" in image_batch:
                image_batch["mask"] = image_batch["mask"].images
            # TODO clean up
            if "fg_mask" in image_batch:
                image_batch["fg_mask"] = image_batch["fg_mask"].images
            if "sparse_pts" in image_batch:
                image_batch["sparse_pts"] = image_batch["sparse_pts"].images
            pixel_batch = collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(self,PatchPixelSampler):
            num_images, image_height, image_width, _ = image_batch["image"].shape
            device = image_batch["image"].device
            pixel_batch,scene_id = self.sample_method(batch= image_batch ,num_images=num_images,image_height=image_height,image_width=image_width,device=device)

        elif isinstance(image_batch["image"], torch.Tensor):
            
            pixel_batch,scene_id = collate_image_dataset_batch(image_batch, self.num_rays_per_batch, 
                                                          keep_full_image=self.keep_full_image)

        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch,scene_id
    
    def sample_multi_scene(self, image_batch: Dict,num_scenes:int = 10, num_imgs_per_scene: int = 39, sample_single_image = True):
        
        if sample_single_image:
            pixel_batch,scene_id = collate_single_image_per_step(image_batch, self.num_rays_per_batch, 
                                                            keep_full_image=self.keep_full_image,
                                                            num_scenes=num_scenes,
                                                            num_imgs_per_scene=num_imgs_per_scene,
                                                            )
        else:
            pixel_batch,scene_id = collate_imagebatch_per_step(image_batch, self.num_rays_per_batch, 
                                                            keep_full_image=self.keep_full_image,
                                                            num_scenes=num_scenes,
                                                            num_imgs_per_scene=num_imgs_per_scene,
                                                            )
        return pixel_batch,scene_id
            

    
    def sample_fisheye(self, image_batch, step, patch_size=1):
        num_images, image_width, image_height, _ = image_batch.shape
        mask = image_batch[:,:,:,-1]
        nonzero_indices = torch.nonzero(mask, as_tuple=False)
        chosen_indices = random.sample(range(len(nonzero_indices)), k=self.num_rays_per_batch)
        indices = nonzero_indices[chosen_indices]
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

        ray_data = image_batch[c, y, x].to('cuda')
        collated_batch = {
            "ray_o": ray_data[..., 0:3],
            "ray_d": ray_data[..., 3:6],
            "image": ray_data[..., 6:9],
            "indices": torch.stack([c, y, x], dim=0).permute(1, 0)
        }

        return collated_batch


def collate_image_dataset_batch_equirectangular(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Operates on a batch of equirectangular images and samples pixels to use for
    generating rays. Rays will be generated uniformly on the sphere.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    # TODO(kevinddchen): make more DRY
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    # only sample within the mask, if the mask is in the batch
    if "mask" in batch:
        # TODO(kevinddchen): implement this
        raise NotImplementedError("Masking not implemented for equirectangular images.")

    # We sample theta uniformly in [0, 2*pi]
    # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
    # This is done by inverse transform sampling.
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/
    num_images_rand = torch.rand(num_rays_per_batch, device=device)
    phi_rand = torch.acos(1 - 2 * torch.rand(num_rays_per_batch, device=device)) / torch.pi
    theta_rand = torch.rand(num_rays_per_batch, device=device)
    indices = torch.floor(
        torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
        * torch.tensor([num_images, image_height, image_width], device=device)
    ).long()

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None}

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


class EquirectangularPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    # overrides base method
    def sample(self, image_batch: Dict):

        pixel_batch = collate_image_dataset_batch_equirectangular(
            image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
        )
        return pixel_batch




class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        config: the PatchPixelSamplerConfig used to instantiate class
    """

    patch_size: int  = 32

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = self.patch_size**2

    # overrides base method
    def sample_method(
        self,
        batch: Dict,
        num_images: int,
        image_height: int,
        image_width: int,
        device: str = "cpu",
    ):
        
        sub_bs = 1 
        indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
            [num_images, image_height - self.patch_size, image_width - self.patch_size],
            device=device,
        )

        indices = (
            indices.view(sub_bs, 1, 1, 3)
            .broadcast_to(sub_bs, self.patch_size, self.patch_size, 3)
            .clone()
        )

        yys, xxs = torch.meshgrid(
            torch.arange(self.patch_size, device=device), torch.arange(self.patch_size, device=device)
        )
        indices[:, ..., 1] += yys
        indices[:, ..., 2] += xxs

        indices = torch.floor(indices).long()
        indices = indices.flatten(0, 2)


        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key not in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points") and value is not None
        }


        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        scene_id = torch.tensor([[0]])
        return collated_batch,scene_id


# class PairPixelSamplerConfig(PixelSamplerConfig):
#     """Config dataclass for PairPixelSampler."""

#     radius: int = 2
#     """max distance between pairs of pixels."""
#     num_rays_per_batch: int = 4096


# class PairPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
#     """Samples pair of pixels from 'image_batch's. Samples pairs of pixels from
#         from the images randomly within a 'radius' distance apart. Useful for pair-based losses.

#     Args:
#         config: the PairPixelSamplerConfig used to instantiate class
#     """

#     # def __init__(self, config: PairPixelSamplerConfig, **kwargs) -> None:
#     #     # self.config = config
#     #     self.radius = 2
#     #     self.num_rays_per_batch = 4096
#     #     # super().__init__(self.config, **kwargs)
#     #     # self.rays_to_sample = self.num_rays_per_batch // 2

#     radius: int  = 2
#     num_rays_per_batch: int = 4096

#     # overrides base method
#     def sample_method(  # pylint: disable=no-self-use
#         self,
#         batch: Dict,
#         num_images: int,
#         image_height: int,
#         image_width: int,
#         mask = None,
#         device = "cpu",
#     ):
#         rays_to_sample = self.rays_to_sample
       
#         rays_to_sample = self.rays_to_sample
#         if batch_size is not None:
#             assert (
#                 int(batch_size) % 2 == 0
#             ), f"PairPixelSampler can only return batch sizes in multiples of two (got {batch_size})"
#             rays_to_sample = batch_size // 2

#         s = (rays_to_sample, 1)
#         ns = torch.randint(0, num_images, s, dtype=torch.long, device=device)
#         hs = torch.randint(self.radius, image_height - self.radius, s, dtype=torch.long, device=device)
#         ws = torch.randint(self.radius, image_width - self.radius, s, dtype=torch.long, device=device)
#         indices = torch.concat((ns, hs, ws), dim=1)

#         pair_indices = torch.hstack(
#             (
#                 torch.zeros(rays_to_sample, 1, device=device, dtype=torch.long),
#                 torch.randint(-self.radius, self.radius, (rays_to_sample, 2), device=device, dtype=torch.long),
#             )
#         )
#         pair_indices += indices
#         indices = torch.hstack((indices, pair_indices)).view(rays_to_sample * 2, 3)


#         c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
#         collated_batch = {
#             key: value[c, y, x]
#             for key, value in batch.items()
#             if key not in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points") and value is not None
#         }
#         # Needed to correct the random indices to their actual camera idx locations.
#         indices[:, 0] = batch["image_idx"][c]
#         collated_batch["indices"] = indices  # with the abs camera indices

#         scene_id = torch.tensor([[0]])
#         return collated_batch,scene_id