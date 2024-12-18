<h1 align="center">EDUS: Efficient Depth-Guided Urban View Synthesis (ECCV 2024)</h1>

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2407.12395)

Sheng Miao*, [Jiaxin Huang*](https://jaceyhuang.github.io/), Dongfeng Bai, Weichao Qiu, Bingbing Liu, [Andreas Geiger](https://www.cvlibs.net/) and [Yiyi Liao](https://yiyiliao.github.io/) 

Our project page can be seen [here](https://xdimlab.github.io/EDUS/).
<img src="./docs/teaser.png" height="200">>
## :book: Datasets
We evaluate our model on [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/) and [Waymo](https://waymo.com/open/download/). Here we show the structure of a test dataset as follow. We provide the preprocessed data for inference on KITTI-360, which contains 5 validation scenes. We exploit the `Metric3d` for metric depth predictions and `HRNet` for sky mask segmentation.

You can download validation data directly from 🤗 [Hugging Face](https://huggingface.co/datasets/cookiemiao/EDUS_infer_dataset/tree/main). 

The dataset should have a structure as follows:
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

## :house: Install EDUS:  Setup the environment
Our EDUS is built on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio). You can follow the nerfstudio webpage to install our code.  

#### Create environment

```bash
conda create --name EDUS -y python=3.8
conda activate EDUS
pip install --upgrade pip
```
#### Dependencies
Install PyTorch with CUDA (this repo has been tested with CUDA 11.7).
```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
```
After pytorch, install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):
```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
#### Installing EDUS
Install EDUS form source code
```bash
git clone https://github.com/XDimLab/EDUS.git
cd EDUS
pip install --upgrade pip setuptools
pip install -e .
```
## :chart_with_upwards_trend: Experiments on KITTI-360 Datasets
Download the checkpoint to perform inference on KITTI-360. We provide the pretrained model trained on `KITTI-360` and `Waymo` and you can download the pre-trained models from  [here](https://drive.google.com/drive/folders/19TfuF-TCNz31rqsMDlI7ghC1i0vYy01c). 

Place the downloaded and put checkpoints in `checkpoint` folder in order to test it later.

### Feed-forward Inference
We further provide the different sparsity levels (50%, 80%) to validate our methods, where a higher drop rate corresponds to a more sparsely populated set of reference images. Replace `$Data_Dir$` with your data path.
```
python scripts/infere_zeroshot.py neuralpnt  --config_file config/test_GVS_nerf.yaml --pipeline.model.mode=val zeronpt-data --data $Data_Dir$ --drop50=True --include_depth_map=True
```
If you want to test on other sparsity setting, replace the `--drop50=True` with `--drop80=True`.

## :clipboard: Citation

If our work is useful for your research, please consider citing:

```
@inproceedings{miao2025efficient,
  title={Efficient Depth-Guided Urban View Synthesis},
  author={Miao, Sheng and Huang, Jiaxin and Bai, Dongfeng and Qiu, Weichao and Liu, Bingbing and Geiger, Andreas and Liao, Yiyi},
  booktitle={European Conference on Computer Vision},
  pages={90--107},
  year={2025},
  organization={Springer}
}
```
## :sparkles: Acknowledgement
- This project is based on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- Some codes are brought from [IBRNet](https://github.com/googleinterns/IBRNet) and [UrbanGIRAFFE](https://github.com/freemty/urbanGIRAFFE).