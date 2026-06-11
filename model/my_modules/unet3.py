import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import pytorch_wavelets as pw
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torchvision import transforms
from torch.nn.modules.utils import _single, _pair, _triple
from torch.fft import fft2, ifft2

from numpy import *


def show_img(x):
    image = (x[0] + 1) / 2
    plt.imshow(image.cpu().detach().numpy().transpose(2, 1, 0))
    plt.show()


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        # encoding shape: [1, 1, dim] (dim=32)
        return encoding


# Integration of x and noise feature
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# ResSE module
class ResSE(nn.Module):
    def __init__(self, ch_in, reduction=2):
        super(ResSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        tmp = x
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + tmp


# FD Info Spliter
class FD_Info_Spliter(nn.Module):
    def __init__(self, dim, in_channels, out_channels, image_size):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.noise_func = nn.Linear(dim, image_size)
        self.noise_resSE = ResSE(in_channels)
        self.sigma_resSE = ResSE(in_channels * 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.HF_guided_resSE = ResSE(in_channels * 2)
        self.channel_transform = nn.Conv2d(6, 3, 1)

    def forward(self, x, noise_embed):
        cnn_x, x = torch.split(x, 3, dim=1)

        assert x.shape == cnn_x.shape
        # Noise image suppression
        b, c, h, w = x.shape
        noise_embed = self.noise_func(noise_embed.view(b, -1))
        noise_embed = noise_embed.unsqueeze(1).unsqueeze(2).repeat(1, 3, self.image_size, 1)
        noise_atten = self.noise_resSE(noise_embed)
        denoise_x = x * noise_atten

        # High and low frequency information separation
        n, m = x.shape[-2:]
        device = x.device

        # create frequency grid
        xx = torch.arange(n, dtype=torch.float, device=device)
        yy = torch.arange(m, dtype=torch.float, device=device)
        u, v = torch.meshgrid(xx, yy)
        u = u - n / 2
        v = v - m / 2

        # convert tensor to complex tensor and apply FFT
        tensor_complex = torch.stack([cnn_x, torch.zeros_like(cnn_x)], dim=-1)
        tensor_complex = torch.view_as_complex(tensor_complex)
        tensor_fft = torch.fft.fftn(tensor_complex)

        # Concat the real and imaginary parts
        x_real, x_imag = torch.real(tensor_fft), torch.imag(tensor_fft)
        x_fd = torch.cat([x_real, x_imag], dim=1)

        # get sigma, numerical stabilization was performed
        sigma_pre = torch.abs(torch.mean(self.avg_pool(self.sigma_resSE(x_fd)), dim=1)) + self.image_size/2
        sigma_min = torch.tensor(self.image_size-10, device=device).view(1, 1, 1).expand_as(sigma_pre)
        sigma = torch.minimum(sigma_pre, sigma_min)

        # calculate Gaussian high-pass filter
        D = torch.sqrt(u ** 2 + v ** 2).to(device)
        H = 1 - torch.exp(-D ** 2 / (2 * sigma ** 2))
        H = H.to(device).unsqueeze(1)
        H = torch.cat([H, H, H], dim=1)

        # apply Gaussian high-pass filter to FFT
        tensor_filtered_fft = tensor_fft * H

        # get Frequency-domain guided attention weight,thus obtain Low-frequency feature map
        x_real_filterd, x_imag_filterd = torch.real(tensor_filtered_fft), torch.imag(tensor_filtered_fft)
        x_fd_filterd = torch.cat([x_real_filterd, x_imag_filterd], dim=1)
        x_hf_guided_atten = self.HF_guided_resSE(x_fd_filterd)

        x_lf_feature = cnn_x * self.channel_transform(x_hf_guided_atten)

        # IFFT，get High-frequency feature map
        tensor_filtered = torch.fft.ifftn(tensor_filtered_fft)
        x_hf_feature = torch.abs(tensor_filtered)

        return torch.cat([x, cnn_x, denoise_x, x_lf_feature, x_hf_feature], dim=1)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


# HF_guided_CA
class HF_guided_CA(nn.Module):
    def __init__(self,image_size, in_channel, norm_groups=32):
        super().__init__()

        self.norm = nn.GroupNorm(norm_groups, in_channel).to('cuda')
        self.q = nn.Conv2d(3, in_channel, 1, bias=False)

        # self.ff_parser_attn_conv = nn.Parameter(torch.ones(in_channel, image_size, image_size))
        self.ff_parser_attn_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1,padding=1,bias=False)

        self.kv = nn.Conv2d(in_channel, in_channel * 2, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input, quary): # input 是UNet中间tensor，quary是DWT的特征
        batch, channel, height, width = input.shape
        head_dim = channel

        norm = self.norm(input) # (1,64,64,64)input经过卷积之后的张量

        ff_parser_attn_map = self.ff_parser_attn_conv(norm)

        dtype = norm.dtype
        norm = fft2(norm)
        norm = norm * ff_parser_attn_map
        norm = ifft2(norm).real
        norm = norm.type(dtype)

        kv = self.kv(norm).view(batch, 1, head_dim * 2, height, width)
        key, value = kv.chunk(2, dim=2)  # bhdyx 把kv在dim=2上分成两个

        quary = self.q(quary).unsqueeze(1) # quary先经过卷积  在quary的dim=1上添加一个维度

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", quary, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, 1, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, 1, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

# MF Block
def fspecial(func_name, kernel_size=3, sigma=1):
    if func_name == 'gaussian':
        m = n = (kernel_size - 1.) / 2.
        y, x = ogrid[-m:m + 1, -n:n + 1]
        h = exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

class LDEB(nn.Module):
    def __init__(self, channel=3, nlv_dens=6):
        # nlv_dens代表卷积核的个数
        super(LDEB, self).__init__()
        self.channel = channel
        self.nlv_dens = nlv_dens

        self.init_conv = nn.Conv2d(in_channels=3, out_channels=self.channel, kernel_size=1)
        self.init_bn = nn.BatchNorm2d(self.channel)
        self.init_gelu = nn.GELU()

        self.conv = torch.nn.Sequential()


        for i in range(self.nlv_dens):
            self.conv.add_module(str(i), Conv2d(self.channel, 1, kernel_size=i + 1,bias=False))
            self.conv[i].weight = nn.Parameter(
                torch.FloatTensor(fspecial('gaussian', kernel_size=i + 1, sigma=(i + 1) / 2)).repeat(1, self.channel, 1,
                                                                                                     1),
                requires_grad=False)
        self.relu = nn.ReLU()


    def forward(self, input):
        input = self.init_conv(input)
        input = self.init_bn(input)
        input = self.init_gelu(input)
        for i in range(self.nlv_dens):
            tmp = self.conv[i](input) * ((i + 1) ** 2)
            if i == 0:
                output = tmp
            else:
                output = torch.cat((output, tmp), 1)
        output = torch.log2(self.relu(output) + 1)
        X = [math.log(i + 1, 2) for i in range(self.nlv_dens)]
        X = torch.tensor(X).to(output.device)
        X = X.view(1, X.shape[0], 1, 1)
        meanX = torch.mean(X, 1, True)
        meanY = torch.mean(output, 1, True)
        eps = 0.1
        Densemap = torch.div(torch.sum((output - meanY) * (X - meanX), 1, True), torch.sum((X - meanX) ** 2 + eps, 1, True))
        return Densemap

def scaled_l2(X, C, S):
    X_minus_C = X.unsqueeze(2) - C.unsqueeze(0)  # 计算 X - C
    squared_distances = torch.norm(X_minus_C, dim=3, p=2)**2  # 计算 \|X - C\|^2
    scaled_distances = S.unsqueeze(0) * squared_distances  # 计算 s_k \|X - C\|^2
    return scaled_distances

class PGB(Module):
    def __init__(self, D=1, K=36):
        super(PGB, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        # a = X.size(1)
        # b = self.D
        assert (X.size(1) == self.D)
        B, D = X.size(0), self.D
        H, W = X.size(2), X.size(3)
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)

        A = A.permute(0, 2, 1).contiguous()
        A = A.view(B, self.K, H, W)
        return A

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
               + str(self.D) + ')'

class MFFB(nn.Module):
    def __init__(self, input_dim=1, k=64):
        super(MFFB, self).__init__()
        self.D = input_dim
        self.K = k

        self.deepmfs = nn.Sequential(
            LDEB(channel=3, nlv_dens=6),
            nn.BatchNorm2d(1), )
        self.deephist = nn.Sequential(
            PGB(D=self.D, K=self.K),
            nn.BatchNorm2d(self.K), )

    def forward(self, input):
        minsample = torch.min(input)
        maxsample = torch.max(input)
        input = 1 + (input - minsample) / (torch.clamp(maxsample - minsample, 10e-8, 10e8)) * 255
        Densemap = self.deepmfs(input)
        histmap = self.deephist(Densemap)
        return histmap

class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels  # 表示通道分开的组数
        chunk_dim = dim // n_levels  # 通道分开后每一组的通道数

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)  # 2、3、4的最大池化下采样
                s = self.mfr[i](s)  # 中间的3×3的DW卷积
                s = F.interpolate(s, size=(h, w), mode='nearest')  # 2、3、4之后的近邻插值上采样
            else:
                s = self.mfr[i](xc[i])  # 1的DW卷积
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x

        return out

class MFatt(nn.Module):
    def __init__(self, final_dim = 3):
        super().__init__()
        self.mf_block = MFFB(input_dim=1, k=64)
        self.att_block = SAFM(dim = 64, n_levels=4)
        self.final_conv = nn.Conv2d(64, final_dim, 3, 1, 1)

    def forward(self, x):
        x = self.mf_block(x)
        x = self.att_block(x)
        x = self.final_conv(x)
        return x


class UNet(nn.Module):
    def __init__(
            self,
            in_channel=9,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(8,),
            res_blocks=3,
            dropout=0,
            with_noise_level_emb=True,
            image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        current_image_size = image_size
        self.fd_spliter = FD_Info_Spliter(dim=inner_channel, in_channels=3, out_channels=3, image_size=image_size)
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]

        self.hf_ca_list = []

        # DWT downsampling of the number of layers, note to preserve equal depth with the unet
        self.J = 4
        for i in range(self.J):
            current_image_size = current_image_size // 2
            self.hf_ca_list.append(HF_guided_CA(current_image_size, inner_channel * (2 ** i)))
        self.hf_ca_list = nn.ModuleList(self.hf_ca_list)

        self.mf_block = MFatt(final_dim=3)

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        # UNet的输入：concat[cnn_prediction, x_noisy] 每个量的channel都是3
        # Images of each layer obtained by DWT
        dwt_x, _ = torch.split(x, 3, dim=1)

        J = self.J
        dwt_img_list = [] # 在UNet中间concat的连接部分注意力的DWT
        dwt_f = pw.DWTForward(J=J, wave='haar', mode='symmetric')
        dwt_f.cuda()
        x_dwt = dwt_f(dwt_x)[1] # 一个列表 长度为4
        # 把四个张量在第3个维度上相加 得到dwt_img_list是一个长度为4的张量
        for i in range(J):
            dwt_img_list.append(x_dwt[i][:, :, 0, :, :] + x_dwt[i][:, :, 1, :, :] + x_dwt[i][:, :, 2, :, :])
        x_mf = self.mf_block(dwt_x)

        # Performing time-step embedding
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None # t:[1,1,64]

        x = self.fd_spliter(x, t) # 将输入和时间步送入FD spliter 是UNet初始条件的处理 输出得到x:[1,15,128,128] 送入spliter的包括cnn_prediction以及噪声图

        x = torch.cat([x, x_mf], dim=1)
        feats = []
        idx = 0
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            if len(feats) != 0 and feats[-1].shape[2:] != x.shape[2:]:
                hf_ca = self.hf_ca_list[idx]
                idx += 1
                query = dwt_img_list.pop(0)
                feats.append(hf_ca(x, query))
            else:
                feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


if __name__ == '__main__':
    img = torch.randn(2, 6, 128, 128).to('cuda')
    t = torch.tensor([[0.645], [0.545]]).to('cuda')
    net = UNet().to('cuda')
    y = net(img, t)
    print(y.shape)



class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    # 修改这里的实现函数
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = torch.nn.functional.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)