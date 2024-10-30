# Welcome to our [ EDUS Project Page.](https://xdimlab.github.io/EDUS/) 
We provide the preprocessed data for inference on KITTI-360, which contains 5 validation scenes. For those who want to run EDUS quickly, you can download the data directly from [Hugging Face](https://huggingface.co/datasets/cookiemiao/EDUS_infer_dataset/tree/main).
## Data Preparation
We evaluate our model on [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/) and [Waymo](https://waymo.com/open/download/). Here we show the structure of a test dataset as follow. You can download them from the official website and then put it into `$ROOT`. 
Before using the following script, please download model of `metric3d` from [here](https://drive.google.com/file/d/1KVINiBkVpJylx_6z1lAC7CQ4kmn-RJRN/view) and put it into `preprocess_dataset/metric3d/models`. In addition, please download `cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth` and `hrnetv2_w48_imagenet_pretrained.pth` for sky segmentation from [here](https://drive.google.com/drive/folders/1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U) and put them into `preprocess_dataset/nvi_sem/checkpoints`.

### KITTI-360 Dataset
1. The original KITTI-360 Dataset have a structure as follows:
    ```
    ├── KITTI-360
        
        ├── 2013_05_28_drive_0000_sync
            ├── image_00
            ├── image_01
        
        ├── calibration
            ├── calib_cam_to_pose.txt
            ├── perspective.txt
        
        ├── data_poses
            ├── cam0_to_world.txt
            ├── poses.txt
    ```
2. The generated dataset should have a structure as follows:
    ```
    ├── $PATH_TO_YOUR_DATASET
        
        ├── $SCENE_0
            ├── depth
            ├── semantic
            ├── mask
            ├── voxel
            ├── *.png
            ...
            ├── transfroms.json
        
        ├── SCENE_1
            ├── depth
            ├── semantic
            ├── mask
            ├── voxel
            ├── *.png
            ...
            ├── transfroms.json
        ...
        
        ├── SCENE_N
            ├── depth
            ├── semantic
            ├── mask
            ├── voxel
            ├── *.png
            ...
            ├── transfroms.json
    ```

### Waymo Dataset
The generated dataset should have a structure as follows:

    ```
    ├── $PATH_TO_YOUR_DATASET
        
        ├── $SCENE_0
            ├── depth
            ├── semantic
            ├── mask
            ├── voxel
            ├── *.png
            ...
            ├── transfroms.json
            ├── transfroms_all.json  # not used
        
        ├── SCENE_1
            ├── depth
            ├── semantic
            ├── mask
            ├── voxel
            ├── *.png
            ...
            ├── transfroms.json
            ├── transfroms_all.json  # not used
        ...
        
        ├── SCENE_N
            ├── depth
            ├── semantic
            ├── mask
            ├── voxel
            ├── *.png
            ...
            ├── transfroms.json
            ├── transfroms_all.json  # not used
    ```

## Install
Our EDUS is built on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio). You can follow the nerfstudio webpage to install our code.  You must have an NVIDIA video card with CUDA installed on the system. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

### Create environment

```bash
conda create --name EDUS -y python=3.8
conda activate EDUS
pip install --upgrade pip
```
### Dependencies
Install PyTorch with CUDA (this repo has been tested with CUDA 11.7) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
```
After pytorch, install the torch bindings for tiny-cuda-nn:
```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
### Install EDUS
Install EDUS form source code
```bash
git clone https://github.com/XDimLab/EDUS.git
cd EDUS
pip install --upgrade pip setuptools
pip install -e .
```
## Experiments on KITTI-360 Datasets

### Run EDUS:
We currently only provide the code for the inference section. You can use the checkpoint to perform inference on KITTI-360.

## Per-trained Model
We also provide the pretrained model trained on `KITTI-360` and `Waymo` and you can download the pre-trained models from  [here](https://drive.google.com/drive/folders/19TfuF-TCNz31rqsMDlI7ghC1i0vYy01c). Place the downloaded checkpoints in `checkpoint` in order to test it later.

### Feed-forward Inference
We further provide the different sparsity levels (50%, 80% or 90%) to validate our methods, where a higher drop rate signifies a more sparsely populated set of reference images.
```
python scripts/infere_zeroshot.py neuralpnt  --config_file config/test_GVS_nerf.yaml --pipeline.model.mode=val zeronpt-data --data $Data_Dir$seq_04_nerfacto_0382_40 --drop80=True --include_depth_map=True
```
If you want to test on other sparsity setting, replace the `--drop80=True` with `--drop50=True` or `--drop90=True`.


