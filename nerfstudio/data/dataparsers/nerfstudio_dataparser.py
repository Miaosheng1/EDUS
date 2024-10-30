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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
import cv2 as cv
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600

@dataclass
class NerfstudioDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nerfstudio)
    """target class to instantiate"""
    data: Path = Path("data/nerfstudio/poster")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "none"
    """The method to use for orientation."""
    center_poses: bool = True
    """Whether to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""
    annotation_3d = None
    """annotation 3D bbx in kitti360"""
    use_fisheye: bool = False
    """ use fisheye """
    mannual_assigned: bool = True 
    """ mannual assigned train/test number """
    include_sky_mask: bool = False
    "whether use depth"
    include_semantic_map: bool = False

    config_file: Optional[str] = 'config/test_GVS_nerf.yaml'
    """GVS nerf config filepath"""
    include_occ3d: bool = True


@dataclass
class Nerfstudio(DataParser):
    """Nerfstudio DatasetParser"""

    config: NerfstudioDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        meta = load_from_json(self.config.data / "transforms.json")
        image_filenames = []
        mask_filenames = []
        semantic_filenames = []
        poses = []
        num_skipped_image_filenames = 0
        img_index=[]

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for sub_i,frame in enumerate(meta["frames"]):
            if sub_i > 60:
                continue
            filepath = PurePath(frame["file_path"])
            fname = self._get_fname(filepath)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(meta["k1"]) if "k1" in meta else 0.0,
                        k2=float(meta["k2"]) if "k2" in meta else 0.0,
                        k3=float(meta["k3"]) if "k3" in meta else 0.0,
                        k4=float(meta["k4"]) if "k4" in meta else 0.0,
                        p1=float(meta["p1"]) if "p1" in meta else 0.0,
                        p2=float(meta["p2"]) if "p2" in meta else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = PurePath(frame["mask_path"])
                mask_fname = self._get_fname(mask_filepath, downsample_folder_prefix="masks_")
                mask_filenames.append(mask_fname)
            if "leader_board" in meta and meta['leader_board']:
                index = str(fname).split('/')[-1]
                img_index.append(index)
        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        if self.config.include_sky_mask:
            mask_filename = self.config.data/"mask"
            mask_filenames += [os.path.join(mask_filename, f) for f in sorted(os.listdir(mask_filename))]
        if self.config.include_semantic_map:
            semantic_filename = self.config.data/"semantics"
            semantic_filenames += [os.path.join(semantic_filename, f) for f in sorted(os.listdir(semantic_filename))]

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        i_all = np.arange(num_images)
        ## 50% dropout Setting
        if "dropout" in meta and meta['dropout']:
            i_train = []
            for i in range(0, num_images, 2):
                if i % 4 == 0:
                    i_train.extend([i,i+1])
            i_train = np.array(i_train)
            num_train_images = len(i_train)
            i_eval = np.setdiff1d(i_all, i_train)[:-2]  # Demo kitti360
        elif "leader_board" in meta and meta['leader_board']:
            num_eval_images = int(meta['num_test'])
            i_train = i_all[:(num_images - num_eval_images)]
            i_eval = np.setdiff1d(i_all, i_train)
        elif self.config.mannual_assigned: 
            i_eval = np.arange(4,num_images,5)
            i_train = np.setdiff1d(i_all, i_eval)  
            # i_eval = sorted(np.concatenate([np.arange(10,num_images,61),np.arange(20,num_images,61),
            #                          np.arange(30,num_images,61)]))
            # i_train = np.setdiff1d(i_all, i_eval)
            # index =  np.arange(0,len(i_train),2) 
            # i_train = i_train[index]
        else:
            self.config.train_split_percentage = 0.8
            num_train_images = math.ceil(num_images * self.config.train_split_percentage)
            num_eval_images = num_images - num_train_images
            i_train = np.linspace(
                0, num_images - 1, num_train_images, dtype=int
            )  # equally spaced training images starting and ending at 0 and num_images-1
            i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images

        if split == "train":
            indices = i_train
            print(f"Train View:  {indices}\n" + f"Train View Num: {len(i_train)}")
        elif split in ["val", "test"]:
            indices = i_eval
            print(f"Test View: {indices}\n" + f"Test View Num: {len(i_eval)}")
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
      

        diff_mean_poses = torch.mean(poses[:,:3,-1], dim=0)

        # Scale poses[translation]
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(poses[:, :3, 3]))

        poses[:, :3, 3] *= scale_factor * self.config.scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        semantic_filenames = [semantic_filenames[i] for i in indices] if len(semantic_filenames) > 0 else []
        poses = poses[indices]


        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-12.8, -9, -20], [12.8, 3.8, 31.2]], dtype=torch.float32
            ) * scale_factor
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

      
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            semantics=semantic_filenames if len(semantic_filenames) > 0 else None
        )

        if self.config.include_occ3d and split == "train":
            ## 记得修改 voxel_size 的大小，如果选择不同的res 
            Occ3d_voxel = np.load(self.config.data / "3340_volume.npy")[None,...]
            bounding_box_min = np.array([-12.8, -9, -20]) 
            bounding_box_max = np.array([12.8, 3.8, 31.2]) 
            voxel_size = np.array([0.2, 0.2, 0.2])
            num_class = 45
            occ3d_dict = {
                "occ3d_voxel":Occ3d_voxel,
                "bounding_box_min":bounding_box_min,
                "bounding_box_max":bounding_box_max,
                "voxel_size":voxel_size,
                "num_class":num_class,
                "Free_state":0,
                "scale_factor":scale_factor,
            }
            dataparser_outputs.occ3d_dict = occ3d_dict


        return dataparser_outputs


    def _get_fname(self, filepath: PurePath, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxillary image data, e.g. masks
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(self.config.data / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (self.config.data / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return self.config.data / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return self.config.data / filepath