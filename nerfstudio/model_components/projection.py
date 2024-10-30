# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class Projector():
    def __init__(self,intrinsics):
        self.intrinsics = intrinsics

    def inbound(self, pixel_locations, h, w):
        '''
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        '''
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras):
        '''
        project 3D points into cameras
        :param xyz: [..., 3]
        :param train_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        '''
        train_cameras = train_cameras * torch.tensor([1, -1, -1,1],device="cuda")
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        train_intrinsics = self.intrinsics[:num_views,...].cuda()  # [n_views, 4, 4]
        train_poses = train_cameras.reshape(-1, 4, 4)  # [n_views, 4, 4]

        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
        projections = train_intrinsics.bmm(torch.inverse(train_poses)) \
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        mask = projections[..., 2] > 0   # a point is invalid if behind the camera

        depth = projections[..., 2].reshape((num_views, ) + original_shape)
        return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
               mask.reshape((num_views, ) + original_shape),\
               depth
    
    def compute_angle(self, xyz, query_camera, train_cameras):
        '''
        :param xyz: [..., 3]
        :param query_camera: [34, ]
        :param train_cameras: [n_views, 34]
        :return: [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        '''
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        num_views = len(train_poses)
        query_pose = query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)  # [n_views, 4, 4]
        ray2tar_pose = (query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2tar_pose /= (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2train_pose /= (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views, ) + original_shape + (4, ))
        return ray_diff

    def compute(self,  xyz, train_imgs, train_cameras, train_depths=None):
        '''
        :param xyz: [n_rays, n_samples, 3]
        :param query_camera: [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :param train_imgs: [1, n_views, h, w, 3]
        :param train_cameras: [1, n_views, 34]
        :param featmaps: [n_views, d, h, w]
        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],
                 ray_diff: [n_rays, n_samples, 4],
                 mask: [n_rays, n_samples, 1]
        '''
        img = train_imgs.detach()
        xyz = xyz.frustums.get_positions()
        train_imgs = train_imgs.permute(0, 3, 1, 2).cuda()  # [n_views, 3, h, w]

        h, w = train_imgs.shape[2:]

        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front, project_depth = self.compute_projections(xyz, train_cameras)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_rays, n_samples, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=True)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]

        ## depth sampling (occulsion judge)
        if train_depths is not None:
            train_depths = train_depths.unsqueeze(-1).permute(0, 3, 1, 2).cuda() # [n_views, 3, h, w]
            depths_sampled = F.grid_sample(train_depths, normalized_pixel_locations, align_corners=True)
            depths_sampled = depths_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]   # [n_rays, n_samples, n_views, 1]
        rgb = rgb_sampled.masked_fill(mask==0,0)

        if train_depths is not None:
            depth = depths_sampled.masked_fill(mask==0,0)
            # self.verify_projection(xyz=xyz,rgb=rgb,img=img,camera=train_cameras)
            rgb = self.occlusion_check(rgbs=rgb,
                                        sampled_depth=depth,
                                        projection_depth= (project_depth*mask_in_front).permute(1, 2, 0)[..., None],
                                        train_imgs = train_imgs,
                                        pixel_locations=pixel_locations,
                                        )
        return rgb
    
    def occlusion_check(self,rgbs,sampled_depth,projection_depth,train_imgs,pixel_locations):
        depth_delta = 0.1
        # print("occlusion")
        no_occlusion = torch.logical_not(projection_depth - sampled_depth> depth_delta) 
        all_false = ~no_occlusion.any(dim=2, keepdim=True).expand(-1, -1, 3, -1)
        no_occlusion[all_false] = True

        modify_rgbs = no_occlusion * rgbs


        # true_colors_sum = torch.sum(modify_rgbs, dim=2)
        # true_counts = torch.sum(no_occlusion,dim=2)
        # avg_true_colors = true_colors_sum / torch.maximum(true_counts, torch.tensor(1))
        # modify_rgbs = torch.where(~no_occlusion, avg_true_colors.unsqueeze(-1).repeat(1,1,1,3), rgbs)

        # exit()

        """ debug 投影点并且可视化"""
        # smapled_id = 50
        # sampled_uv = pixel_locations[:,0,smapled_id,:]
        # for i in range(3):
        #     cur_img = train_imgs[i].permute(1,2,0).detach().cpu().numpy()*255
        #     image = Image.fromarray(cur_img.astype(np.uint8))
        #     cur_location = sampled_uv[i]
        #     draw = ImageDraw.Draw(image)
        #     draw.ellipse((cur_location[0] - 5, cur_location[1] - 5, cur_location[0] + 5, cur_location[1] + 5), outline='red', width=3)
        #     image.save(f'projiect{i}.png')


            

        return modify_rgbs
    

    # def verify_projection(self,xyz,rgb,img,camera):
    #     save_dir = "project"
    #     os.makedirs(save_dir,exist_ok=True)
    #     np.save(os.path.join(save_dir,"xyz.npy"),xyz.detach().cpu().numpy())
    #     np.save(os.path.join(save_dir,"pose.npy"),camera.squeeze().detach().cpu().numpy())
    #     np.save(os.path.join(save_dir,"sample_rgb.npy"),rgb.detach().cpu().numpy())
    #     import imageio
    #     imageio.imwrite(os.path.join(save_dir,"source.png"), img.squeeze().detach().cpu().numpy())
    #     exit()
    #     return







