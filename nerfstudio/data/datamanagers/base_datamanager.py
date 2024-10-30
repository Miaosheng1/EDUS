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
Datamanager.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch
import tyro
from rich.progress import Console
from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import Literal
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.dataparsers.zeronpt_dataparser import ZeronptDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset, GeneralizedDataset
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler, PixelSampler,PatchPixelSampler
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate

from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.misc import IterableWrapper
from nerfstudio.data.datamanagers.utils import get_source_images_from_current_imageid, eval_source_images_from_current_imageid,render_trajectory_source_pose
from omegaconf import OmegaConf
CONSOLE = Console(width=120)

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[  # Omit prefixes of flags in subcommands.
    tyro.extras.subcommand_type_from_defaults(
        {
            "nerfstudio-data": NerfstudioDataParserConfig(),
            "zeronpt-data":ZeronptDataParserConfig(),

        },
        prefix_names=False,  # Omit prefixes in subcommands themselves.
    )
]
"""Union over possible dataparser types, annotated with metadata for tyro. This is the
same as the vanilla union, but results in shorter subcommand names."""


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:
        1. A Raybundle: This will contain the rays we are sampling, with latents and
            conditionals attached (everything needed at inference)
        2. A "batch" of auxilury information: This will contain the mask, the ground truth
            pixels, etc needed to actually train, score, etc the model

    Rationale:
    Because of this abstraction we've added, we can support more NeRF paradigms beyond the
    vanilla nerf paradigm of single-scene, fixed-images, no-learnt-latents.
    We can now support variable scenes, variable number of images, and arbitrary latents.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data


    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """

    train_dataset: Optional[Dataset] = None
    eval_dataset: Optional[Dataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None

    def __init__(self):
        """Constructor for the DataManager class.

        Subclassed DataManagers will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__() BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
        super().__init__()
        self.train_count = 0
        self.eval_count = 0
        if self.train_dataset and self.test_mode != "inference":
            self.setup_train()
        if self.eval_dataset and self.test_mode != "inference":
            self.setup_eval()
        ''' PreLoaded all fisheye rays to shuffle'''
        self.batch_fisheye_rays = None

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def iter_train(self):
        """The __iter__ function for the train iterator.

        This only exists to assist the get_train_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.train_count = 0

    def iter_eval(self):
        """The __iter__ function for the eval iterator.

        This only exists to assist the get_eval_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.eval_count = 0

    def get_train_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_train and next_train functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_train_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_eval and next_eval functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_eval_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""
        raise NotImplementedError

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""
        raise NotImplementedError

    @abstractmethod
    def next_train(self, step: int) -> Tuple:
        """Returns the next batch of data from the train data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple:
        """Returns the next batch of data from the eval data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple:
        """Returns the next eval image."""
        raise NotImplementedError

  

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}


@dataclass
class VanillaDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: VanillaDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = ZeronptDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
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
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    num_neighbour_select: int = 3
    """ number of Select the neighbour images ( used in the neural point) """


class VanillaDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: VanillaDataManagerConfig
    train_dataset: InputDataset
    eval_dataset: InputDataset

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        config_path: str = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser = self.config.dataparser.setup()
        self.config_setup = OmegaConf.load(config_path)
        self.num_neighbour_select = self.config_setup.data_manager.num_neighbour_select

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        super().__init__()

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return GeneralizedDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split="train"),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return GeneralizedDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: InputDataset, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return PixelSampler(*args, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        torch.autograd.set_detect_anomaly(True)
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_patch_sampler = PatchPixelSampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 2,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras.to(self.device),
            self.train_camera_optimizer,  # should be shared between train and eval.
        )
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            image_indices=self.config.eval_image_indices,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )
        # print("done")

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader).copy()

        ## 是否在 MultiScene 场景采样
        if self.train_dataset.metadata.get('num_scenes'):
            batch,scene_id = self.train_pixel_sampler.sample_multi_scene(image_batch,
                                                                    num_scenes=self.train_dataset.metadata['num_scenes'],
                                                                    num_imgs_per_scene=self.train_dataset.metadata['num_img_per_scene'],
                                                                    sample_single_image=True)
            
            image_id = batch['indices'][0,0]
            source_images,src_poses,src_depths = get_source_images_from_current_imageid(image_batch = image_batch,
                                                                   scene_id = scene_id,
                                                                   image_id = image_id,
                                                                   all_pose = self.train_dataset.cameras.camera_to_worlds.to(self.device),
                                                                   num_imgs_per_scene=self.train_dataset.metadata['num_img_per_scene'],
                                                                   num_select=self.num_neighbour_select)
        else:
            batch, scene_id = self.train_pixel_sampler.sample(image_batch,step)
            
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        ray_bundle.scene_id = scene_id.repeat(*ray_bundle.origins.shape[:1],1)

        ## Set source_images and source_poses
        ray_bundle.set_source_image(source_images=source_images,source_poses=src_poses,source_depth=src_depths)

        return ray_bundle, batch
    

    def undistortion_from_pixel(self,x,y,z_dist,mirror):

        z_after = torch.sqrt(x * x + y * y).to(self.device)
        indx = torch.abs(z_dist[:, 1:] - z_after[None, :]).argmin(dim=0)
        x *= (z_dist[indx, 0] + mirror)
        y *= (z_dist[indx, 0] + mirror)
        return x,y

    def next_train_fisheye_shuffle(self, step: int) -> Tuple[RayBundle, Dict]:
        self.train_count += 1
        batch = self.train_pixel_sampler.sample_fisheye(self.batch_fisheye_rays, step)

        rays_o,rays_d, true_color, indices = batch['ray_o'],batch['ray_d'],batch['image'],batch['indices']
        directions_norm = torch.norm(rays_d, dim=-1, keepdim=True)

        camera_indx = torch.tensor(indices[:,0] + len(self.train_dataset))[:,None].to(self.device)
        # camera_indx = torch.tensor(indices[:, 0]).unsqueeze(-1)

        from nerfstudio.cameras.rays import RayBundle
        raybundle = RayBundle(
            origins=rays_o,
            directions=rays_d,
            pixel_area=None,
            camera_indices=0,
            directions_norm=directions_norm,
            times=None,
        )
        raybundle.camera_indices = camera_indx.int()
        indices[:, 0] += len(self.train_dataset)
        batch = {
            "image": true_color,
            "indices": indices
        }
        return raybundle, batch


    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        batch,scene_id = self.eval_pixel_sampler.sample(image_batch,step = 0)
        
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        ray_bundle.scene_id = scene_id.repeat(*ray_bundle.origins.shape)

         ## 在 Raybundle  中设置 source_images 和 source_poses
        ray_bundle.set_source_image(source_images=source_images,source_poses=src_poses,source_depth=src_depths)

        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            if isinstance(batch["image"], BasicImages):  # If this is a generalized dataset, we need to get image tensor
                batch["image"] = batch["image"].images[0]
                camera_ray_bundle = camera_ray_bundle.reshape((*batch["image"].shape[:-1], 1))
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])

            if self.eval_dataset.metadata.get('num_scenes'):
                scene_id = image_idx // self.eval_dataset.metadata['num_img_per_scene'] ## 这个根据不同的场景需要修改
                train_image_batch =  next(self.iter_train_image_dataloader).copy()
                source_images,src_poses, src_depths= eval_source_images_from_current_imageid(image_batch= train_image_batch,
                                                                   scene_id = scene_id,
                                                                   eval_image_id = image_idx,
                                                                   num_select=self.num_neighbour_select,
                                                                   all_pose = self.train_dataset.cameras.camera_to_worlds.to(self.device),
                                                                   eval_poses = self.eval_dataset.cameras.camera_to_worlds.to(self.device),
                                                                   num_imgs_per_scene=self.train_dataset.metadata['num_img_per_scene'])

                camera_ray_bundle.set_source_image(source_images=source_images,source_poses=src_poses,source_depth=src_depths)

            else:
                scene_id = 0

            return image_idx, camera_ray_bundle, batch, scene_id
        raise ValueError("No more eval images")
    

    def next_fixed_eval_image(self, step: int,scene_id:int = 0, ) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.fixed_indices_eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            if isinstance(batch["image"], BasicImages):  # If this is a generalized dataset, we need to get image tensor
                batch["image"] = batch["image"].images[0]
                camera_ray_bundle = camera_ray_bundle.reshape((*batch["image"].shape[:-1], 1))
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])

            
            # scene_id = 0 ## 这个根据不同的场景需要修改
            train_image_batch =  next(self.iter_train_image_dataloader).copy()
            source_images,src_poses,src_depth = eval_source_images_from_current_imageid(image_batch= train_image_batch,
                                                                scene_id = scene_id,
                                                                eval_image_id = image_idx,
                                                                num_select=self.num_neighbour_select,
                                                                all_pose = self.train_dataset.cameras.camera_to_worlds.to(self.device),
                                                                eval_poses = self.eval_dataset.cameras.camera_to_worlds.to(self.device),
                                                                num_imgs_per_scene=self.train_dataset.metadata['num_img_per_scene'])

            camera_ray_bundle.set_source_image(source_images=source_images,source_poses=src_poses,source_depth=src_depth)

            

            return image_idx, camera_ray_bundle, batch, scene_id
        raise ValueError("No more eval images")
    
    def set_refrence_image_for_interpolate_image(self,camera_raybundle,c2w=None):
        train_image_batch =  next(self.iter_train_image_dataloader).copy()
        source_images,src_poses,src_depth = render_trajectory_source_pose(image_batch= train_image_batch,
                                                                scene_id = 0,
                                                                target_pose=c2w.to('cuda').float(),
                                                                num_select=self.num_neighbour_select,
                                                                all_pose = self.train_dataset.cameras.camera_to_worlds.to(self.device),
                                                                num_imgs_per_scene=self.train_dataset.metadata['num_img_per_scene'])

        camera_raybundle.set_source_image(source_images=source_images,source_poses=src_poses,source_depth=src_depth)

        return camera_raybundle
    

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups


@dataclass
class FlexibleDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: FlexibleDataManager)
    """Target class to instantiate."""
    train_num_images_to_sample_from: int = 1
    """Number of images to sample during training iteration."""


class FlexibleDataManager(VanillaDataManager):
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        additional_output = {}
        if "src_imgs" in image_batch.keys():
            ray_indices = ray_indices.to(image_batch["src_idxs"].device)
            assert (ray_indices[:, 0] == image_batch["image_idx"]).all()
            additional_output["uv"] = ray_indices[:, 1:]
            additional_output["src_idxs"] = image_batch["src_idxs"][0]
            additional_output["src_imgs"] = image_batch["src_imgs"][0]
            additional_output["src_cameras"] = self.train_dataset._dataparser_outputs.cameras[
                image_batch["src_idxs"][0]
            ]
        return ray_bundle, batch, additional_output
