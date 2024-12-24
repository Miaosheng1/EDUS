from __future__ import annotations

import traceback
from datetime import timedelta
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import tyro
import yaml
from rich.console import Console
import os
from pathlib import Path

from nerfstudio.configs import base_config as cfg
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import Trainer
import mediapy as media
import numpy as np
from tqdm import tqdm
from nerfstudio.utils import colormaps
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure
from nerfstudio.cameras.camera_paths import get_interpolated_camera_path
CONSOLE = Console(width=120)
DEFAULT_TIMEOUT = timedelta(minutes=30)
# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def infer_loop(local_rank: int, world_size: int, config: cfg.Config, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    trainer = Trainer(config, local_rank, world_size)
    trainer.setup(config_path=config.config_file)
    num_eval_images = len(trainer.pipeline.datamanager.eval_dataset.filenames)

    save_dir = None
    if config.pipeline.datamanager.dataparser.drop50:
        save_dir = "zeroshot_Drop50/"+config.experiment_name
    elif config.pipeline.datamanager.dataparser.drop80:
        save_dir = "zeroshot_Drop80/"+config.experiment_name
    elif config.pipeline.datamanager.dataparser.drop90:
        save_dir = "zeroshot_Drop90/"+config.experiment_name
    else:
        raise ValueError("Error")

    os.makedirs(save_dir,exist_ok=True)
    
    cpu_or_cuda_str = trainer.device.split(":")[0]
    lpips_fn = LearnedPerceptualImagePatchSimilarity().to('cuda')
    ssim = structural_similarity_index_measure
    
    sum_psnr = 0
    sum_lpips = 0
    sum_ssim = 0
    with torch.autocast(device_type=cpu_or_cuda_str, enabled=trainer.mixed_precision):
         for i in tqdm(range(num_eval_images)):
                outputs,gt_img = trainer.pipeline.get_eval_zeroshot(step=i)
                render_img = outputs['rgb']
                render_depth = outputs['depth']
                render_bg_depth = outputs['bg_depth']
                render_accumulation = outputs['bg_accumulation']
                render_bg_acc = outputs['accumulation']
                render_depth = render_depth + render_bg_depth
                render_depth = render_depth * (render_accumulation + render_bg_acc).clip(0,1)
                depth_map = colormaps.apply_XDLab_color_depth_map(render_depth.detach().cpu().numpy())
                gt_img =  torch.tensor(gt_img).to(outputs['accumulation'])
           
                media.write_image(os.path.join(save_dir,"render_{:03d}.png".format(i)), render_img.detach().cpu().numpy())
                media.write_image(os.path.join(save_dir,"depth_{:03d}.png".format(i)), depth_map)

                psnr = -10. * np.log10(np.mean(np.square(gt_img.detach().cpu().numpy() - render_img.detach().cpu().numpy())))
                lpips = lpips_fn(gt_img.unsqueeze(0).permute(0,3,1,2),render_img.unsqueeze(0).permute(0,3,1,2))
                SSIM = ssim(render_img.unsqueeze(0).permute(0,3,1,2),gt_img.unsqueeze(0).permute(0,3,1,2))

                sum_psnr += psnr
                sum_lpips += lpips
                sum_ssim += SSIM
                print("image {} PSNR:{} ,SSIM: {}, LPIPS: {}".format(i,psnr,SSIM,lpips))
   
    CONSOLE.print(f"[bold green]Average PSNR:{sum_psnr/ num_eval_images}",justify="center")
    CONSOLE.print(f"[bold green]Average SSIM:{sum_ssim / num_eval_images}",justify="center")
    CONSOLE.print(f"[bold green]Average Lpips:{sum_lpips/num_eval_images}",justify="center")
    CONSOLE.print(f"[bold yellow] Results are saved in {save_dir}.")
   

def render_interpolate_camera(camera,trainder,save_dir=None):
    save_dir = "inter_video/"+save_dir
    if Path(save_dir).is_dir():
        os.system(f"rm -rf {save_dir}")
    os.makedirs(save_dir,exist_ok=True)

    new_cameras = get_interpolated_camera_path(camera,steps=100)
    trainder.pipeline.eval()
    for camera_idx in tqdm(range(new_cameras.size)):
        camera_ray_bundle = new_cameras.generate_rays(camera_indices=camera_idx)
        camera_ray_bundle.scene_id = torch.tensor([0],device='cuda')
        trainder.pipeline.datamanager.set_refrence_image_for_interpolate_image(camera_ray_bundle,c2w=new_cameras[camera_idx].camera_to_worlds)
        with torch.no_grad():
            outputs = trainder.pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
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



def launch(
    main_func: Callable,
    num_gpus_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[cfg.Config] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
) -> None:
    """Function that spawns muliple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (Config, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
    """
    assert config is not None
    world_size = num_machines * num_gpus_per_machine
    if world_size <= 1:
        # world_size=0 uses one CPU in one process.
        # world_size=1 uses one GPU in one process.
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())
 
    elif world_size > 1:
        # Using multiple gpus with multiple processes.
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            join=False,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                config,
                timeout,
            ),
        )
        # process_context won't be None because join=False, so it's okay to assert this
        # for Pylance reasons
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")


def main(config: cfg.Config) -> None:
    """Main function."""
    
    config.set_timestamp()
    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.dataparser.data")
        config.pipeline.datamanager.dataparser.data = config.data

    if config.trainer.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.trainer.load_config}")
        config = yaml.load(config.trainer.load_config.read_text(), Loader=yaml.Loader)

    # print and save config
    config.print_to_terminal()
    config.save_config()

    launch(
        main_func=infer_loop,
        num_gpus_per_machine=config.machine.num_gpus,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )



def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()