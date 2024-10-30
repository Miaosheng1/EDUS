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
Code for camera paths.
"""


import torch
from nerfstudio.cameras.camera_utils import get_interpolated_poses_many
from nerfstudio.cameras.cameras import Cameras



def get_interpolated_camera_path(cameras: Cameras, steps: int) -> Cameras:
    """Generate a camera path between 1st and last cameras.

    Args:
        cameras: Cameras object containing intrinsics of all cameras.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        A new set of cameras along a path.
    """
    Ks = cameras.get_intrinsics_matrices()
    poses = cameras.camera_to_worlds.cpu().numpy()[[0,-1],...]
    poses = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps)[:,:3,:]

    cameras = Cameras(fx=Ks[0, 0, 0], fy=Ks[0, 1, 1], cx=Ks[0, 0, 2], cy=Ks[0, 1, 2], 
                      height=cameras.height[0],
                      width=cameras.width[0],
                      camera_to_worlds=torch.from_numpy(poses).to('cuda').float())
    return cameras

