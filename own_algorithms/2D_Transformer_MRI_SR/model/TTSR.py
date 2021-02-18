"""
Reference:
[1] 2020.Learning Texture Transformer Network for Image Super-Resolution
(https://arxiv.org/abs/2006.04139)
"""

from model import MainNet, LTE, SearchTransfer

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWT, IDWT # (or import DWTForward, DWTInverse)


"apply wavelets transform for any image and inverse wavelets transform"
"""
Note: 小波变换（wavelet transform，WT）是一种新的变换分析方法，它继承和发展了短时傅立叶变换局部化的思想，同时又克服了窗口大小不随频率变化等缺
点，能够提供一个随频率改变的“时间-频率”窗口，是进行信号时频分析和处理的理想工具。它的主要特点是通过变换能够充分突出问题某些方面的特征，能对时
间（空间）频率的局部化分析，通过伸缩平移运算对信号（函数)逐步进行多尺度细化，最终达到高频处时间细分，低频处频率细分，能自动适应时频信号分析的
要求，从而可聚焦到信号的任意细节，解决了Fourier变换的困难问题。
小波变换使用的基底函数不像FFT那样是三角函数，而是小波函数。所谓“小波函数”是一类函数，该类型函数需要满足：均值为0并在时域和频域都局部化
（不是蔓延整个坐标轴的），满足这两条的函数就是小波函数。具体来说，就是小波在整个时间范围的幅度平均值是0，具有有限的持续时间和突变的频率和振幅，
可以是不规则，也可以是不对称。
小波有很多，最简单的是Haar Wavelet。所以小波分析或者说小波变换要做的就是将原始信号表示为一组小波基的线性组合，然后通过忽略其中不重要的部分达
到数据压缩或者说降维的目的。另外注意小波变换的结果是2D的时频谱，不是FFT那样的1D频谱。
See https://www.youtube.com/watch?v=ExU0izGXgSI for more detail of wavelet transform technology
"""
def calculate_wavelet_transform(img):
    """
    DWT stand for Discrete Wavelet Transform.
    """
    xfm = DWT(J = 1, mode = 'zero', wave = 'db3').cuda() # J stands for level of wavelet decomposition
    Y_low_frequency, Y_high_frequency = xfm(img)
    print(Y_low_frequency.shape)     # Low frequency information: Approximation coefficients
    print(Y_high_frequency[0].shape) # 1st level High frequency information: Horizontal detail coefficients, Vertical detail coefficients, Diagonal detail coefficients
    # Beware Y_low_frequency has shape (N, C_in, H′, W′) and Y_high_frequency has shape list(N, C_in, 3, H′, W′). Where H′ = ceil(H/2)+2 and W′ = ceil(W/2)+2
    return Y_low_frequency, Y_high_frequency[0]

def calculate_inverse_wavelet_transform(Y_low_frequency, Y_high_frequency):
    """
    IDWT stand for Inverse Discrete Wavelet Transform.
    """
    ifm = IDWT(mode='zero', wave='db3').cuda()
    img = ifm((Y_low_frequency, [Y_high_frequency]))
    print(Y_low_frequency.shape)     # Low frequency information: Approximation coefficients
    print(Y_high_frequency.shape) # High frequency information: Horizontal detail coefficients, Vertical detail coefficients, Diagonal detail coefficients
    # Beware Y_low_frequency has shape (N, C_in, H′, W′) and Y_high_frequency has shape list(N, C_in, 3, H′, W′). Where H′ = ceil(H/2)+2 and W′ = ceil(W/2)+2
    return img


"""
Class TTSR implements the entire framework of TTSR network which is shown in figure 2 of [1].
"""
class TTSR(nn.Module):
    def __init__(self, args):
        super(TTSR, self).__init__()
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )
        if not self.args.wavelet_to_extract_high_freq_components_for_reference:
            # Define the ordinary MainNet for processing MRI image in the image domain.
            self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, n_colors=args.n_colors,
            res_scale=args.res_scale, scale_factor=args.scale_factor, use_channel_attention_in_ResBlock_of_SFE = args.use_channel_attention_in_ResBlock_of_SFE)
        else:
            # Define the freq domain MainNet for processing high freq components of MRI image in the freq domain. Beware here
            # n_colors has been hardcoded as 3 due that we leave 3 high freq components through this class MainNet for freq domain.
            self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, n_colors=3,
            res_scale=args.res_scale, scale_factor=args.scale_factor, use_channel_attention_in_ResBlock_of_SFE = args.use_channel_attention_in_ResBlock_of_SFE)
        self.LTE = LTE.LTE(requires_grad=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.n_colors = args.n_colors

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            if self.args.dataset == 'CUFED':
                sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            elif self.args.dataset == 'MRI_SR':
                sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy(sr)
            return sr_lv1, sr_lv2, sr_lv3

        if self.args.dataset == 'CUFED':
            _, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)   # lrsr_lv3 is the Q in equation (1) of [1].
            _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)  # refsr_lv3 is the K in equation (2) of [1].
            ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)  # ref_lv3 is the V in equation (3) of [1].
            S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
            sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)
            return sr, S, T_lv3, T_lv2, T_lv1
        elif self.args.dataset == 'MRI_SR' and self.args.wavelet_to_extract_high_freq_components_for_reference == True:
            if self.n_colors == 1:
                # Extract high freq components in freq domain by applying wavelet transform.
                N = lr.size(0)
                lr_low_freq, lr_high_freq = calculate_wavelet_transform(lr)
                lrsr_low_freq, lrsr_high_freq = calculate_wavelet_transform(lrsr)
                ref_low_freq, ref_high_freq = calculate_wavelet_transform(ref)
                refsr_low_freq, refsr_high_freq = calculate_wavelet_transform(refsr)
                
                lr_high_freq = lr_high_freq.view(N, 3, lr_high_freq.shape[3], lr_high_freq.shape[4])  # lr_high_freq has only 3 high freq components in freq domain of low resolution MRI image.
                lrsr_high_freq = lrsr_high_freq.view(N, 3, lrsr_high_freq.shape[3], lrsr_high_freq.shape[4])  # lrsr_high_freq has only 3 high freq components in freq domain of bicurbic upsampled low resolution MRI image.
                ref_high_freq = ref_high_freq.view(N, 3, ref_high_freq.shape[3], ref_high_freq.shape[4])  # ref_high_freq has only 3 high freq components in freq domain of reference MRI image.
                refsr_high_freq = refsr_high_freq.view(N, 3, refsr_high_freq.shape[3], refsr_high_freq.shape[4])  # refsr_high_freq has only 3 high freq components in freq domain of downsampled and bicurbic upsampled reference MRI image.
            else:
                raise SystemExit('Error: n_colors must be 1 when using wavelet to extract high freq components for reference data!')
            _, _, lrsr_high_freq_lv3  = self.LTE(lrsr_high_freq.detach())   # lrsr_high_freq_lv3 is comparable to the Q in equation (1) of [1].
            _, _, refsr_high_freq_lv3 = self.LTE(refsr_high_freq.detach())  # refsr_high_freq_lv3 is comparable to the K in equation (2) of [1].
            ref_high_freq_lv1, ref_high_freq_lv2, ref_high_freq_lv3 = self.LTE(ref_high_freq.detach())  # ref_high_freq_lv3 is comparable to the V in equation (3) of [1].
            # Search transfer for all the high freq components in freq domain.
            S_high_freq, T_high_freq_lv3, T_high_freq_lv2, T_high_freq_lv1 = self.SearchTransfer(lrsr_high_freq_lv3, refsr_high_freq_lv3, ref_high_freq_lv1, ref_high_freq_lv2, ref_high_freq_lv3)
            # Crop only center part of lr_high_freq and leave it into MainNet.
            lr_high_freq_center_part = lr_high_freq[:, :, 1:-1, 1:-1]
            # Beware here the MainNet is the MainNet for processing in the freq domain, which is different to the MainNet for processing the MRI image in image domain.
            sr_high_freq_center_part = self.MainNet(lr_high_freq_center_part, S_high_freq, T_high_freq_lv3, T_high_freq_lv2, T_high_freq_lv1)
            # Initialize sr_high_freq using all zero.
            sr_high_freq = torch.zeros(sr_high_freq_center_part.shape[0], sr_high_freq_center_part.shape[1],  sr_high_freq_center_part.shape[2] + 2,  sr_high_freq_center_part.shape[3] + 2).cuda()
            # Copy lr_high_freq_center_part to the center part of sr_high_freq.
            sr_high_freq[:, :, 1:-1,  1:-1] = sr_high_freq_center_part
            # Copy boundary part of lr_high_freq to the boundary part of sr_high_freq.
            """ sr_high_freq[:, :, 0, :] = lr_high_freq[:, :, 0, :]
            sr_high_freq[:, :, -1, :] = lr_high_freq[:, :, -1, :]
            sr_high_freq[:, :, :, 0] = lr_high_freq[:, :, :, 0]
            sr_high_freq[:, :, :, -1] = lr_high_freq[:, :, :, -1] """
            sr_high_freq[:, :, 0, :] = lrsr_high_freq[:, :, 0, :]
            sr_high_freq[:, :, -1, :] = lrsr_high_freq[:, :, -1, :]
            sr_high_freq[:, :, :, 0] = lrsr_high_freq[:, :, :, 0]
            sr_high_freq[:, :, :, -1] = lrsr_high_freq[:, :, :, -1]
            
            # Inverse wavelet transformation from freq domain back to image domain to restore the SR MRI image.
            sr_high_freq = sr_high_freq.unsqueeze(1)
            sr = calculate_inverse_wavelet_transform(lrsr_low_freq, sr_high_freq)
            return sr, S_high_freq, T_high_freq_lv3, T_high_freq_lv2, T_high_freq_lv1
        elif self.args.dataset == 'MRI_SR' and self.args.wavelet_to_extract_high_freq_components_for_reference == False:
            if self.n_colors == 1:  # When loading the normal MRI image with number of channel = 1, we simply copy the same image 3 times and stack them together for putting into LTE(pretrained VGG feature extractor).
                lrsr = torch.cat((lrsr, lrsr, lrsr), 1)
                ref = torch.cat((ref, ref, ref), 1)
                refsr = torch.cat((refsr, refsr, refsr), 1)
            elif not(self.n_colors == 3):
                raise SystemExit('Error: n_colors must be 1 or 3!')
            # When loading the MRI image with number of channel = 3 which we already stacked 3 different MRI image with number of channel = 1 together, we let it into LTE(pretrained VGG feature extractor) without doing anything more.
            _, _, lrsr_lv3  = self.LTE(lrsr.detach())   # lrsr_lv3 is the Q in equation (1) of [1].
            _, _, refsr_lv3 = self.LTE(refsr.detach())  # refsr_lv3 is the K in equation (2) of [1].
            ref_lv1, ref_lv2, ref_lv3 = self.LTE(ref.detach())  # ref_lv3 is the V in equation (3) of [1].
            S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
            sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)
            return sr, S, T_lv3, T_lv2, T_lv1
