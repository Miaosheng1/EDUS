import torch 
import numpy as np 
import os
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import Encoding, NeRFEncoding
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from torch.utils.tensorboard import SummaryWriter
from nerfstudio.cameras import camera_utils
import matplotlib.pyplot as plt
from torch import nn
from nerfstudio.cameras.rays import RayBundle
import open3d as o3d
from rich.progress import track
from tqdm import tqdm, trange
from nerfstudio.model_components.scene_colliders import LidarNearFarCollider
from nerfstudio.model_components.losses import urban_radiance_field_depth_loss

class LidarRay():
    def __init__(self, LidarRays = None,scale_factor = 1,device=None,bbx_min=None ) -> None:
        super().__init__()
        self.device = device
        self.scale_factor = scale_factor
        self.LidarRays = LidarRays

        self.bounding_box_min = bbx_min
        self.uniform_sampler = UniformSampler(single_jitter=True)
        self.lidar_numsamples = 128
        self.chunk_size = 512
        self.termination_depth = 0
        self.sigma_decay_rate = torch.tensor(0.99985)
        self.depth_sigma = 0.5 * scale_factor
        self.depth_sigma_min = 0.2 * scale_factor
       
    def cal_termination_depth(self,start_pnt,end_pnt):
        differences = end_pnt - start_pnt
        depth = torch.sqrt(torch.sum(differences ** 2, dim=1))
        return depth.unsqueeze(-1)    
    
    def _get_sigma(self):
        self.depth_sigma = torch.maximum(
            self.sigma_decay_rate * self.depth_sigma, torch.tensor([self.depth_sigma_min])
        )
        return self.depth_sigma

    
    def get_lidar_samples(self,scene_id = 0):
        chunk_size = self.chunk_size
        cur_lidarays = torch.from_numpy(self.LidarRays[scene_id.item()]).to(self.device) * self.scale_factor
        lidar_rayo = cur_lidarays[:,:3]
        lidar_ray_xyz = cur_lidarays[:,3:6]
        selected_index = torch.randint(low=0, high=lidar_rayo.shape[0], size=[chunk_size])
        lidar_rayo = lidar_rayo[selected_index]
        lidar_ray_xyz = lidar_ray_xyz[selected_index]
        self.termination_depth = self.cal_termination_depth(start_pnt=lidar_rayo,end_pnt=lidar_ray_xyz)

        self.collider = LidarNearFarCollider(
                near_plane=0.01, far_plane=self.termination_depth + 0.05
            )

        directions_stack = lidar_ray_xyz - lidar_rayo
        directions_stack, _ = camera_utils.normalize_with_norm(directions_stack, -1)
        
        lidar_raysample = self.generate_lidar_raysample(ray_o=lidar_rayo,ray_d=directions_stack)
        # self.visRaysamplesAndLidarPoints(raysamples=lidar_raysample,lidar_voxel = cur_lidarays[:,3:6] )
        # exit()
        # print("scene ID......................:",scene_id)
        return lidar_raysample
    
    def cal_urf_loss(self,lidar_samples=None,weights=None,predicted_depth=None):

        steps = (lidar_samples.frustums.starts + lidar_samples.frustums.ends) / 2
        sigma = self._get_sigma().to(self.device)
        depth_loss = urban_radiance_field_depth_loss(weights= weights,
                                        termination_depth= self.termination_depth, 
                                        predicted_depth=predicted_depth,
                                        steps=steps,
                                        sigma=sigma,
                                        )
        # self.debug(steps=steps,termination_depth= self.termination_depth,weights= weights,predicted_depth=predicted_depth,sigma=sigma)
        return depth_loss

    def generate_lidar_raysample(self,ray_o,ray_d):
        lidar_raybundle = RayBundle(
            origins=ray_o,
            directions=ray_d,
            pixel_area=1,
            directions_norm=ray_d,
        )
        lidar_raybundle = self.collider(lidar_raybundle)
        lidar_samples = self.uniform_sampler(lidar_raybundle,num_samples=self.lidar_numsamples)
        return lidar_samples
    
    
    """ Visuailize the lidarsamples"""
    def visRaysamplesAndLidarPoints(self,raysamples,lidar_voxel):
        lidar_pnt = raysamples.frustums.get_positions()
        selected_index = torch.randint(low=0, high=self.chunk_size, size=[20]).to(self.device)

        vis_lidar_pnt = lidar_pnt[selected_index].reshape(-1,3)

        pointcloud = o3d.geometry.PointCloud()
        point = torch.cat([vis_lidar_pnt,lidar_voxel],dim=0)
        pointcloud.points = o3d.utility.Vector3dVector(point.detach().cpu().numpy())

        point = point.detach().cpu().numpy()
        z_coords = np.array(point[:, 2])
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        colors = plt.cm.jet((z_coords - z_min) / (z_max - z_min))[:, :3]  # 使用 matplotlib 颜色映射
        pointcloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud("debug.ply", pointcloud)
        
        return
    
    """ Visuailize the color pointcloud and lidarsamples"""
    def vis_point(self, lidar_samples, color_volume):
        lidar_pnt = lidar_samples.frustums.get_positions()
        selected_index = torch.randint(low=0, high=self.chunk_size, size=[20]).to(self.device)
        Voxel_size = 0.2
        vis_lidar_pnt = lidar_pnt[selected_index].reshape(-1,3)

        color_volume = color_volume.permute(2,3,4,1,0).squeeze()
        non_zero_color = torch.any(color_volume > 0, dim=-1)
        non_zero_indices = torch.nonzero(non_zero_color, as_tuple=False)
        volume_colors = color_volume[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]]

        non_zero_indices = non_zero_indices.detach().cpu().numpy()
        volume_pnts = np.stack((non_zero_indices[:, 0] * Voxel_size, \
                                non_zero_indices[:, 1] * Voxel_size, \
                                non_zero_indices[:, 2] * Voxel_size), axis=1)
        volume_pnts += self.bounding_box_min

        volume_pnts = volume_pnts * self.scale_factor.numpy()

        pointcloud = o3d.geometry.PointCloud()
        points = np.concatenate([volume_pnts,vis_lidar_pnt.detach().cpu().numpy()],axis=0)
        colors = np.concatenate([volume_colors.detach().cpu().numpy(),np.ones_like(vis_lidar_pnt.detach().cpu().numpy())*0.5],axis=0)
        pointcloud.points = o3d.utility.Vector3dVector(points)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud("debug_urf.ply", pointcloud)
        exit()

    def debug(self,steps=0,termination_depth= 0,weights= 0,predicted_depth=0,sigma=0):
        np.save("steps.npy",steps.detach().cpu().numpy())
        np.save("predicted_depth.npy",predicted_depth.detach().cpu().numpy())
        np.save("weights.npy",weights.detach().cpu().numpy())
        np.save("termination_depth.npy",termination_depth.detach().cpu().numpy())
        np.save("sigma.npy",sigma.detach().cpu().numpy())
        exit()
        
        

