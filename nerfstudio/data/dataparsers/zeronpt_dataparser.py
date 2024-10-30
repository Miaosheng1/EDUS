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
from typing import Optional, Type, Tuple

import numpy as np
import torch
from PIL import Image
from rich.console import Console
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
class ZeronptDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Zeronpt)
    """target class to instantiate"""
    data: Path = Path("data/nerfstudio/poster")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    mannual_assigned: bool = True 
    """ Mannual assigned train/test number """
    include_sky_mask: bool = True
    """Whether load sky binary Mask"""
    include_semantic_map: bool = False
    """Whether load semantic"""
    include_depth_map: bool = False
    """Whether load predicted depth map"""
    config_file: Optional[str] = 'config/test_GVS_nerf.yaml'
    """EDUS config filepath"""
    drop50: bool = True
    """Drop50 setting"""
    drop90: bool = False
    """Drop90 setting"""
    drop80: bool = False
    """Drop90 setting"""
    mono_voxel: bool = False
    """Use mono voxel"""
    bounding_box_min: Tuple[float, float, float] = (-12.8, -9, -20)
    """Minimum of the bounding box, the bounding box range is decided by the scene """
    bounding_box_max: Tuple[float, float, float] = (12.8, 3.8, 31.2)
    """Maximum of the bounding box,the bounding box range is decided by the scene """
    voxel_size: float = 0.2

@dataclass
class Zeronpt(DataParser):
    """Nerfstudio DatasetParser"""

    config: ZeronptDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):

        meta = load_from_json(self.config.data / "transforms.json")
        image_filenames = []
        mask_filenames = []
        semantic_filenames = []
        depth_filenames = []
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

        for _,frame in enumerate(meta["frames"]):

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
        if self.config.include_depth_map:
            depth_filename = self.config.data/"mono_depth"
            depth_filenames += [os.path.join(depth_filename, f) for f in sorted(os.listdir(depth_filename))]     

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        i_all = np.arange(num_images)
        voxel_dir = None
        ## 50% dropout Setting
        if self.config.drop50: 
            voxel_dir = PurePath("drop50_voxel")
            i_eval = sorted(np.concatenate([np.arange(2,num_images,4), np.arange(3,num_images,4)],axis=0))
            i_train = np.setdiff1d(i_all, i_eval) 
            i_eval = i_eval[::2]
            i_eval = [2, 6, 14, 18, 22, 26, 34, 38, 42, 46, 54, 58]
        elif self.config.drop80: 
            voxel_dir = PurePath("drop80_voxel")
            i_train = np.array([0,1,10,11,20,21,30,31,40,41,50,51,60]) 
            i_eval = [2, 6, 14, 18, 22, 26, 34, 38, 42, 46, 54, 58]
        elif self.config.drop90:  
            voxel_dir = PurePath("drop90_voxel")
            i_train = np.array([0,1,20,21,40,41,60,61])
            i_eval = [2, 6, 14, 18, 22, 26, 34, 38, 42, 46, 54, 58]
        else:
            raise("Error, select Drop50, Drop80 or Drop90")
            exit()

        if split == "train":
            indices = i_train
            print(f"Train View:  {indices}\n" + f"Train View Num: {len(i_train)}")
        elif split in ["val", "test"]:
            indices = i_eval
            print(f"Test View: {indices}\n" + f"Test View Num: {len(i_eval)}")
        else:
            raise ValueError(f"Unknown dataparser split {split}")


        poses = torch.from_numpy(np.array(poses).astype(np.float32))
      
        # Scale poses[translation]
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(poses[:, :3, 3]))

        poses[:, :3, 3] *= scale_factor * self.config.scale_factor
        print(scale_factor)
        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        semantic_filenames = [semantic_filenames[i] for i in indices] if len(semantic_filenames) > 0 else []
        poses = poses[indices]


        # in x,y,z order
        # assumes that the scene is centered at the origin
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

        self.config.num_scenes = 1
        num_img_per_scene = len(indices) // self.config.num_scenes
        metadata = {"num_scenes":self.config.num_scenes,
                    "num_img_per_scene":num_img_per_scene}

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            semantics=semantic_filenames if len(semantic_filenames) > 0 else None,
            depths=depth_filenames if len(depth_filenames) > 0 else None, 
        )

        """ Load the 3D voxelized pointcloud"""
        seq_id = str(self.config.data).split('/')[-1].split("_")[-2]
        if self.config.mono_voxel:
            voxelized_data = np.load(self.config.data /  f"{seq_id}_mono_volume.npy")[None,...]
        else:
            voxelized_data = np.load(self.config.data / voxel_dir / f"{seq_id}_volume.npy")[None,...]
        bounding_box_min = self.config.bounding_box_min 
        bounding_box_max = self.config.bounding_box_max 
        voxel_size = np.array([self.config.voxel_size]*3)

        Input_volume = {
            "data":voxelized_data,
            "bounding_box_min":bounding_box_min,
            "bounding_box_max":bounding_box_max,
            "voxel_size":voxel_size,
            "scale_factor":scale_factor,
        }
        dataparser_outputs.volume_dict = Input_volume


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