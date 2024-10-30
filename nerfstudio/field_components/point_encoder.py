from typing import Dict, Optional, Tuple
from rich.console import Console
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import ParameterList
from torchtyping import TensorType
from inplace_abn import InPlaceABN


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))


class VoxelEncoder(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(VoxelEncoder, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)
        # self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)
        # self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)
        # self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=True),
            # norm_act(8)
        )
    
        # self.conv12 = nn.Conv3d(8, 16, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # latents = []
        color_volume = x
        conv0 = self.conv0(x)
        # conv2 = self.conv2(self.conv1(conv0))  # 16
        conv_1 = self.conv1(conv0)  # 16
        conv2 = self.conv2(conv_1)  # 16
        conv4 = self.conv4(self.conv3(conv2))  # 32
    

        x = self.conv6(self.conv5(conv4))  # 64
        # latents.append(x)
        x = conv4 + self.conv7(x)  # 32
        del conv4
        x = conv2 + self.conv9(x)
        # latents.append(x)
        del conv2
        del conv_1
        ## 这样 sum 之后的效果更好
        x = self.conv11(x) + conv0
        # latents.append(x)
        # latent = self.aggressive_feature_volume(latents)
        res = torch.cat([color_volume,x],dim=1)
        return res
    
    # def aggressive_feature_volume(self,latents):
    #     scale = [8,2]
    #     for i in range(len(latents) -1 ):
    #             latents[i] = F.interpolate(
    #                 latents[i],
    #                 scale_factor = (scale[i],scale[i],scale[i]),
    #                 mode="trilinear",
    #                 align_corners=True,
    #             )
    #     latent = torch.cat(latents, dim=1)
    #     return latent



if __name__ == "__main__":
    # voxel = np.load("/data1/smiao/multiscene_kitti_seg/few_show/voxel/train_pose_fewshow.npy")
    # voxel = torch.from_numpy(voxel).to('cuda')
    # coord = voxel.permute(3,0,1,2).float()
    coord = torch.ones(3, 128, 256, 384).to('cuda')
    voxel_size = 0.1  # 设置体素大小
    print(coord.shape)
    batch_size = 1
    coor = coord[None, ...]

    net = VoxelEncoder(in_channels=3).to('cuda')
    ans = net.forward(coor)
    print(ans.shape)

    exit()