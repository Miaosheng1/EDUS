import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

from nerfstudio.third_party.gsn_ops import FusedLeakyReLU, fused_leaky_relu


class ConvLayer3d(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        bias=True,
        activate=True,
    ):
        # assert not (upsample and downsample), 'Cannot upsample and downsample simultaneously'
        layers = []

        padding = kernel_size // 2
        layers.append(
            EqualConv3d(in_channel, out_channel, kernel_size, padding=padding, stride=1, bias=bias and not activate)
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)
        

class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate.

    During the forward pass the weights are scaled by the inverse of the He constant (i.e. sqrt(in_dim)) to
    prevent vanishing gradients and accelerate training. This constant only works for ReLU or LeakyReLU
    activation functions.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    bias: bool
        Use bias term.
    bias_init: float
        Initial value for the bias.
    lr_mul: float
        Learning rate multiplier. By scaling weights and the bias we can proportionally scale the magnitude of
        the gradients, effectively increasing/decreasing the learning rate for this layer.
    activate: bool
        Apply leakyReLU activation.

    """

    def __init__(self, in_channel, out_channel, bias=True, bias_init=0, lr_mul=1, activate=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))
        else:
            self.bias = None

        self.activate = activate
        self.scale = (1 / math.sqrt(in_channel)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activate:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"


class EqualConv3d(nn.Module):
    """3D convolution layer with equalized learning rate.

    During the forward pass the weights are scaled by the inverse of the He constant (i.e. sqrt(in_dim)) to
    prevent vanishing gradients and accelerate training. This constant only works for ReLU or LeakyReLU
    activation functions.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    kernel_size: int
        Kernel size.
    stride: int
        Stride of convolutional kernel across the input.
    padding: int
        Amount of zero padding applied to both sides of the input.
    bias: bool
        Use bias term.

    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 3)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv3d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class SPADE3D(nn.Module):
    def __init__(self, norm_nc, label_nc, nhidden = 128, norm_type = 'instance', kernel_size = 3):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)
        elif norm_type == 'syncbatch':
            self.param_free_norm = nn.SyncBatchNorm(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = nhidden
        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            EqualConv3d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = EqualConv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = EqualConv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='trilinear')

        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


class ConvResBlock3d(nn.Module):
    """3D convolutional residual block with equalized learning rate.

    Residual block composed of 3x3 convolutions and leaky ReLUs.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    upsample: bool
        Apply upsampling via strided convolution in the first conv.
    downsample: bool
        Apply downsampling via strided convolution in the second conv.

    """

    def __init__(self, in_channel, out_channel, kernel_size = 3, padding = 1):
        super().__init__()

        # assert not (upsample and downsample), 'Cannot upsample and downsample simultaneously'
        mid_ch = min(in_channel,out_channel)

        self.conv1 = ConvLayer3d(in_channel, mid_ch, kernel_size=kernel_size, bias=True, activate=True)
        self.conv2 = ConvLayer3d(mid_ch, out_channel, kernel_size=kernel_size, bias=True, activate=True)

        if (in_channel != out_channel):
            self.skip = ConvLayer3d(
                in_channel,
                out_channel,
                kernel_size=1,
                activate=False,
                bias=False,
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if hasattr(self, 'skip'):
            skip = self.skip(input)
            out = (out + skip) / math.sqrt(2)
        else:
            out = (out + input) / math.sqrt(2)
        return out


class SPADE3DResnetBlock(nn.Module):
    def __init__(self, 
                # args4resblock
                in_channel, 
                out_channel,
                # weight_norm_type,
                # args4spade
                spade_nc,
                hidden_nc,
                kernel_size = 3,
                ):
        super().__init__()
        # Attributes
        self.learned_shortcut = (in_channel != out_channel)
        middle_channel = min(in_channel, out_channel)

        # create conv layers
        self.conv_0 = ConvLayer3d(in_channel, middle_channel, kernel_size=kernel_size,bias=True, activate=True )
        self.conv_1 = ConvLayer3d(middle_channel, out_channel, kernel_size=kernel_size, bias=True, activate=True)
        
        # define normalization layers
        self.norm_0 = SPADE3D(in_channel, spade_nc, hidden_nc)
        self.norm_1 = SPADE3D(middle_channel, spade_nc,hidden_nc)

        if self.learned_shortcut:
            self.conv_s = ConvLayer3d(
                in_channel,
                out_channel,
                kernel_size=1,
                bias=False,
                activate=False)
            self.norm_s = SPADE3D(in_channel, spade_nc, hidden_nc)
        self.actvn= nn.LeakyReLU(0.2, inplace=True)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, input, seg):
        out = self.conv_0(self.actvn(self.norm_0(input, seg)))
        out = self.conv_1(self.actvn(self.norm_1(out, seg)))

        if self.learned_shortcut:
            skip = self.conv_s(self.norm_s(input, seg))
            out = (out + skip) / math.sqrt(2)
        else:
            out = (out + input) / math.sqrt(2)
        return out


class FeatureVolumeGenerator(nn.Module):
    def __init__(self, 
        init_res = 4,
        volume_res = (64, 64, 64),
        max_channel= 512,
        out_channel = 16,
        input_channels = 3,
        spade_hidden_channel = 128,
        noise_type = 'oasis',
        z_dim = 256,
        z_dim_oasis = 64,
        final_unconditional_layer = True,
        final_tanh = False,
        **kwargs
    ):
        super().__init__()
        self.h, self.w, self.d = volume_res
        out_res = min(volume_res)
        self.out_nc = out_channel
        self.max_nc = max_channel
        self.noise_type = noise_type
        self.z_dim = z_dim
        self.final_tanh = final_tanh
        self.final_unconditional_layer = final_unconditional_layer

        res_log2 =  int(math.log2(out_res) - math.log2(init_res))
        self.block_num = res_log2  ## 根据 输入和输出的 volume size 计算得到 （8 --> 16）
        nc_list = [min(self.out_nc * (2 ** (res_log2 - i)),self.max_nc) for i in range(res_log2)]
        nc_list += [nc_list[-1], out_channel]
        self.nc_list = nc_list
        self.sw, self.sh, self.sd = self.compute_latent_vector_size3D(num_up_layers=self.block_num)
        if noise_type == 'oasis':
            self.z_dim_oasis = z_dim_oasis 
            z_nc = z_dim_oasis
            spade_nc = input_channels + z_nc
            self.fc = EqualConv3d(spade_nc, nc_list[0], 3, padding=1)
        else:
            spade_nc = input_channels
            self.fc = EqualLinear(z_dim, nc_list[0] * self.sw * self.sh * self.sd)

        for i in range(self.block_num):
            block_name = f'spade_block{i}'
            self.add_module(block_name, 
                            SPADE3DResnetBlock(in_channel=nc_list[i],
                                               out_channel=nc_list[i+1],
                                               spade_nc=spade_nc, 
                                               hidden_nc=spade_hidden_channel))

        self.up2x= nn.Upsample(scale_factor=2)

        if final_unconditional_layer:
            self.nospade = ConvResBlock3d(nc_list[-2], nc_list[-2])
        # else:
        #     self.spade = SPADE3DResnetBlock(min(2 * nf, max_nc), min(1 * nf, max_nc), opt)
        self.out = nn.Sequential(
            nn.LeakyReLU(0.2),
            EqualConv3d(nc_list[-2], nc_list[-1], kernel_size=3 ,padding=1)
        )

    def compute_latent_vector_size3D(self, num_up_layers):
         
        sw = self.w // (2**num_up_layers)
        sh = round(sw / (self.w / self.h))
        sd = round(sw / (self.w / self.d))

        return sw, sh, sd

    def forward(self, input, z=None):
        B, C, D, H, W = input.shape
        input = rearrange(input, 'B C D H W -> B C H W D')
        
        if z is None:
            z = torch.randn(B, self.z_dim, dtype=torch.float32, device=input.get_device())
        
        if self.noise_type == 'oasis':
            if self.z_dim_oasis > 0:
                z = z[:, :self.z_dim_oasis].to(input.device) # only use z_oasis part
                z = z.view(B, self.z_dim_oasis, 1, 1, 1)
                z = z.expand(B, self.z_dim_oasis, input.size(2), input.size(3), input.size(4))
                input = torch.cat((z, input), dim=1)
            # we downsample input and run convolution
            x = F.interpolate(input, size=(self.sh, self.sw, self.sd))
            x = self.fc(x)
            
        else:
            # we sample z from unit normal and reshape the tensor
            x = self.fc(z)
            x = x.view(-1, self.nc_list[0], self.sh, self.sw, self.sd)

        for i in range(self.block_num):
            block = getattr(self, f'spade_block{i}')
            x = block(x, input)
            x = self.up2x(x)
        ### x.shape (1,32,64,128,256)
        if self.final_unconditional_layer:  ### x.shape (1,32,64,128,256)
            x = self.nospade(x)
        x = self.out(x)  ### x.shape (1,16,64,128,256)
        if self.final_tanh:
            x = torch.tanh(x)
        return x
    
if __name__ == "__main__":
    coord = torch.ones(3, 128, 128, 256).float().to('cuda')  ## (C,X,Y,Z)
    print(coord.shape)
    batch_size = 1
    coor = coord[None, ...]  ## (B,C,W,H,D)
    print(coor.shape)
    coor = coor.permute(0,1,4,2,3)
    print(coor.shape)
  
    net = FeatureVolumeGenerator(init_res=8,volume_res=(128,128,256),out_channel=16,input_channels=3,z_dim_oasis=0).to('cuda')
    ##  X -> W   Y -> H  Z -> D
    ans = net.forward(coor)   ## 输入 D H W
    print(ans.shape)

    exit()