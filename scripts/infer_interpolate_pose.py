from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

import mediapy as media
import numpy as np
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from skimage.metrics import structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import cv2
from torchvision.transforms.functional import resize as gpu_resize
from nerfstudio.utils import colormaps
from nerfstudio.data.utils.label import assigncolor
from nerfstudio.cameras.camera_paths import get_interpolated_camera_path
from tqdm import tqdm
CONSOLE = Console(width=120)

class RenderDatasets():
    """Load a checkpoint, render the trainset and testset rgb,normal,depth, and save to the picture"""
    def __init__(self,parser_path):
        self.load_config = Path(parser_path.config)
        seq_id = str(self.load_config).split('/')[1].split('_')[-2]
        exp_method = seq_id+ "_" + str(self.load_config).split('/')[-2]
        if exp_method == 'nerfacto':
            self.rendered_output_names = ['rgb', 'depth','accumulation',"semantics"]
        else:
            self.rendered_output_names = ['rgb', 'depth','accumulation','bg_rgb',"sky",'bg_depth','bg_accumulation']
        self.root_dir = Path('exp_' + exp_method)
        if self.root_dir.is_dir():
            os.system(f"rm -rf {self.root_dir}")
        self.task = parser_path.task
        self.is_leaderboard = parser_path.is_leaderboard
        self.ssim = structural_similarity
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.ssim = structural_similarity_index_measure
        self.downscale = parser_path.downscale

    def generate_MSE_map(self,redner_img,gt_img,index):
        mse = np.mean((redner_img - gt_img) ** 2,axis=-1)
        plt.close('all')
        plt.figure(figsize=(15, 5))  ## figure 的宽度:1500 height: 500
        ax = plt.subplot()
        sc = ax.imshow((mse), cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax)
        plt.savefig(os.path.join(str(self.root_dir)+"/error_map"+ f'/{self.task}_{index:02d}_mse.png'), bbox_inches='tight')
        return

    def search_Camera_index(self,train_names,test_list):
        train_idx =[]
        test_idx = []
        for name in train_names:
            name = str(name).split('/')[-1][:-4]
            train_idx.append(name)
        for name in test_list:
            name = str(name).split('/')[-1][:-4]
            test_idx.append(name)
        result = []
        i = 0
        for element in test_idx:
            while i < len(train_idx) and train_idx[i] < element:
                i += 1
            result.append(i)

        return result
    

    def main(self):
        config, pipeline, _ = eval_setup(
            self.load_config,
            test_mode= "test",
        )
        
        
        ## 在few_show 场景上记得开区 average embedding
        pipeline.model.field.use_average_appearance_embedding = True

        
        # os.makedirs(self.root_dir / "error_map", exist_ok=True)
        # os.makedirs(self.root_dir / "gt_rgb", exist_ok=True)
        # os.makedirs(self.root_dir / "depth", exist_ok=True)

      

        # config.print_to_terminal()
        # 'Read the image and save in target directory'
        # os.makedirs(self.root_dir / "render_rgb",exist_ok=True)

        # CONSOLE.print(f"[bold yellow]Rendering {len(DataCache.image_cache)} Images")
        cameras = pipeline.datamanager.eval_dataset.cameras.to(pipeline.device)
       


        progress = Progress(
            TextColumn(":movie_camera: Rendering :movie_camera:"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            ItersPerSecColumn(suffix="fps"),
            TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        )
        render_image = []
        render_depth = []
        render_accumulation = []
        render_semantics = []
        render_bg_image = []
        render_bg_depth = []
        render_bg_acc = []
        render_sky = []

        save_dir = "interpolate_video/"+config.experiment_name
        os.makedirs(save_dir,exist_ok=True)

        new_cameras = get_interpolated_camera_path(cameras,steps=5)
        pipeline.eval()
        for camera_idx in tqdm(range(new_cameras.size)):
            camera_ray_bundle = new_cameras.generate_rays(camera_indices=camera_idx)
            camera_ray_bundle.scene_id = torch.tensor([0],device='cuda')
            pipeline.datamanager.set_refrence_image_for_interpolate_image(camera_ray_bundle,c2w=new_cameras[camera_idx].camera_to_worlds)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            render_img = outputs['rgb']
            render_depth = outputs['depth']
            render_bg_depth = outputs['bg_depth']
            render_accumulation = outputs['bg_accumulation']
            render_bg_acc = outputs['accumulation']
            render_depth = render_depth + render_bg_depth
            render_depth = render_depth * (render_accumulation + render_bg_acc).clip(0,1)

            depth_map = colormaps.apply_XDLab_color_depth_map(render_depth.detach().cpu().numpy())
            media.write_image(os.path.join(save_dir,"render_{:03d}.png".format(camera_idx)), render_img.detach().cpu().numpy())
            media.write_image(os.path.join(save_dir,"depth_{:03d}.png".format(camera_idx)), depth_map)



        exit()

        with progress:
            for camera_idx in progress.track(range(start_camera_idx,cameras.size ), description=""):
                image_id,camera_ray_bundle, _ , scene_id=pipeline.datamanager.next_fixed_eval_image(step=camera_idx)
                camera_ray_bundle.scene_id =  torch.tensor([scene_id]).to('cuda')
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                for rendered_output_name in self.rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}",
                                      justify="center")
                        sys.exit(1)
                    output_image = outputs[rendered_output_name].cpu().numpy()
                    if rendered_output_name == 'rgb':
                        render_image.append(output_image)
                    elif rendered_output_name == 'depth':
                        render_depth.append(output_image)
                    elif rendered_output_name == 'accumulation':
                        render_accumulation.append(output_image)
                    elif rendered_output_name == 'bg_accumulation':
                        render_bg_acc.append(output_image)    
                    elif rendered_output_name == 'sky':
                        render_sky.append(output_image)
                    elif rendered_output_name == 'bg_rgb':
                        render_bg_image.append(output_image)
                    elif rendered_output_name == 'bg_semantics':
                        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["bg_semantics"], dim=-1),dim=-1)
                        h, w = semantic_labels.shape[0], semantic_labels.shape[1]
                        ## waymo dataset or kitti360 have different colormap
                        semantic_color_map = assigncolor(semantic_labels.reshape(-1)).reshape(h, w, 3)
                        render_bg_semantics.append(semantic_color_map)
                    elif rendered_output_name == 'bg_depth':
                        render_bg_depth.append(output_image)
                    elif rendered_output_name == "semantics":
                        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1),dim=-1)
                        h, w = semantic_labels.shape[0], semantic_labels.shape[1]
                        ## waymo dataset or kitti360 have different colormap
                        semantic_color_map = assigncolor(semantic_labels.reshape(-1)).reshape(h, w, 3)
                        render_semantics.append(semantic_color_map)
        CONSOLE.print("[bold green]Rendering Images Finished")


        ''' Output rgb depth and normal image'''
        sum_psnr = 0
        sum_lpips = 0
        sum_ssim = 0
        for i,image in sorted(DataCache.image_cache.items()):
            if self.downscale != 1:
                H_ = int(image.shape[0] // self.downscale)
                W_ = int(image.shape[1] // self.downscale)
                image = gpu_resize(image.movedim(-1, 0),(H_, W_)).movedim(0, -1)

            if self.is_leaderboard and self.task == 'testset':
                media.write_image(self.root_dir /"render_rgb"/ test_file[i], render_image[i])
            else:
                if "bg_rgb" in self.rendered_output_names:     
                    media.write_image(self.root_dir / "render_rgb" / f'{self.task}_{i:02d}_redner_dv.png', render_bg_image[i])
                    media.write_image(self.root_dir / "render_rgb" / f'{self.task}_{i:02d}_sky.png', render_sky[i])
                    # media.write_image(self.root_dir/"gt_rgb" / f'{self.task}_{i:02d}_bg.png', (bg_acc.detach().cpu().numpy()))
                    acc = colormaps.apply_colormap(torch.from_numpy(render_accumulation[i]))
                    media.write_image(self.root_dir/"gt_rgb" / f'{self.task}_{i:02d}_acc.png', (acc.detach().cpu().numpy()))
                    render_depth[i] = render_depth[i] + render_bg_depth[i]
                    render_depth[i] = render_depth[i] * (render_accumulation[i] + render_bg_acc[i]).clip(0,1)

                # depth_map = colormaps.apply_depth_colormap(torch.from_numpy(render_depth[i]))
                depth_map = colormaps.apply_XDLab_color_depth_map(render_depth[i])
                # error_map = torch.mean((torch.from_numpy(render_image[i]) - image) ** 2,dim=-1,keepdim=True)
                # error_map = colormaps.apply_colormap(error_map)

                media.write_image(self.root_dir / "render_rgb" / f'{self.task}_{i:02d}_redner_rgb.png', render_image[i])
                media.write_image(self.root_dir/"gt_rgb" / f'{self.task}_{i:02d}_gtrgb.png', (image.detach().cpu().numpy()))
                media.write_image(self.root_dir/"depth" / f'{self.task}_{i:02d}_depth.png', depth_map)
                np.save(self.root_dir/"depth" / f'{self.task}_{i:02d}_depth.npy', render_depth[i])


                psnr = -10. * np.log10(np.mean(np.square(image.detach().cpu().numpy() - render_image[i])))
                lpips = self.lpips(image.unsqueeze(0).permute(0,3,1,2),torch.from_numpy(render_image[i]).unsqueeze(0).permute(0,3,1,2))
                SSIM = self.ssim(image.unsqueeze(0).permute(0,3,1,2),torch.from_numpy(render_image[i]).unsqueeze(0).permute(0,3,1,2))

                sum_psnr += psnr
                sum_lpips += lpips
                sum_ssim += SSIM
                print("image {} PSNR:{} ,SSIM: {}, LPIPS: {}".format(i,psnr,SSIM,lpips))
                if "semantics" in self.rendered_output_names:
                    render_semantics[i] *= render_accumulation[i]
                    media.write_image(self.root_dir / "semantics" / f'{self.task}_{i:02d}_pred.png', render_semantics[i])

        CONSOLE.print(f"[bold green]Average PSNR:{sum_psnr/len(DataCache.image_cache)}",justify="center")
        CONSOLE.print(f"[bold green]Average PSNR:{sum_ssim / len(DataCache.image_cache)}",justify="center")
        CONSOLE.print(f"[bold green]Average PSNR:{sum_lpips/len(DataCache.image_cache)}",justify="center")

        # print(f"Average PSNR:{sum_psnr/len(DataCache.image_cache)}")
        # print(f"Average SSIM: {sum_ssim / len(DataCache.image_cache) }")
        # print(f"Average LPIPS: {sum_lpips / len(DataCache.image_cache) }")
        
        CONSOLE.print(f" [bold blue] Store image to {self.root_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='testset or trainset')
    parser.add_argument('--config',type=str,help='Config Path')
    parser.add_argument('--is_leaderboard',action='store_true')
    parser.add_argument("--downscale",type=int,help= "downscale the image",default=1)
    config = parser.parse_args()

    RenderDatasets(config).main()