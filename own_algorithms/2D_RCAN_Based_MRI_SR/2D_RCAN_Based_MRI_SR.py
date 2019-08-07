# -*- coding: utf-8 -*-
"-------------------------------------------------------------------------------------------------"
"""
2D_RCAN_Based_MRI_SR Reconstruct (2D Deep Residual Channel Attention Network Based MRI Super-Resolution Image Reconstruction)
This is the code for RCAN based MRI SR Reconstruction, RCAN model proposed to generate super resolution(SR) image in the paper: 
2018.Image Super-Resolution Using Very Deep Residual Channel Attention Networks
https://arxiv.org/abs/1807.02758
"""
"""
Author: chisyliu@hotmail.com *
        hao.li@med.uni-heidelberg.de *
        
        * Both authors contribute equally
Version: 1.0.0
"""
"-------------------------------------------------------------------------------------------------"
"""
This is the current version we are working on, in 20190807
This is a demo code of 2D_RCAN_Based_MRI_SR.
    0) This is the 2D version of 2D_RCAN_Based_MRI_SR, in this version we have following items:
        a) Optional module(exist or NOT): a VGG feature extractor is added before the main "CNN based Reconstruct network"(so the input to "CNN based Reconstruct network" is feature map of LR image)
        b) Optional module(either one exist): either Pixel-Wise MSE loss or Pixel-Wise L1 loss
        c) Optional module(exist or NOT): weighted k space loss(fft loss)
        d) Optional module(exist or NOT): weighted VGG loss
        e) Optional module(exist or NOT): L2 Regularization
        f) Optional module(): ssim smooth L1 loss
        g) Optional module(): gradient map smoothL1 loss
	i) add option to use lookahead optimizer
        loss function = Pixel-Wise MSE loss(or Pixel-Wise L1 loss) + weighted VGG loss + weighted k space loss + ssim smoothL1 loss + gradient map smoothL1 loss +L2 Regularization
       
        
       In this 2D version, the data format has been changed. The input data is just 64 x 64 2D matrix rather than 64 x 64 x 64, we already collapse all the 64 layers into only one layer in the data tailing and noise filtering processing
"""
"-------------------------------------------------------------------------------------------------"
"""Reference: 
    [1] 2015 Deep Residual Learning for Image Recognition. download here: https://arxiv.org/pdf/1512.03385.pdf"
        tutorial online: https://icml.cc/2016/tutorials/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf
    [4] 2017 Improving Generalization Performance by Switching from Adam to SGD. https://arxiv.org/pdf/1712.07628.pdf
    [5] 2016 Residual Networks Behave Like Ensembles of Relatively Shallow Networks. 
        https://papers.nips.cc/paper/6556-residual-networks-behave-like-ensembles-of-relatively-shallow-networks.pdf
        https://zhuanlan.zhihu.com/p/37820282
    [6] 2014 Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    [7] 2015 Batch normalization: Accelerating deep network training by reducing internal covariate shift
        https://arxiv.org/pdf/1502.03167.pdf
    [8] 2016 Aggregated Residual Transformations for Deep Neural Networks
        https://arxiv.org/pdf/1611.05431.pdf
    [9] Memory-Efficient Implementation of DenseNets
        https://arxiv.org/pdf/1707.06990.pdf
    [10]2016 Densely Connected Convolutional Networks. https://arxiv.org/pdf/1608.06993.pdf
        source code: https://github.com/chisyliu/DenseNet
    [12]2016 SGDR: Stochastic Gradient Descent with Warm Restarts. https://arxiv.org/abs/1608.03983
    [13]2016 Identity Mappings in Deep Residual Networks. https://arxiv.org/abs/1603.05027
    [16]2017 Multi-scale brain MRI super-resolution using deep 3D convolutional networks. 
    [17]2018 Brain MRI super resolution using 3D deep densely connected neural networks.
    [19]2018.Image Super-Resolution Using Very Deep Residual Channel Attention Networks. https://arxiv.org/abs/1807.02758
"""
"""
Note:
    a) A possible error about "Broken pips" and "multi-processing" can be happened, see this blog
    https://medium.com/@mackie__m/running-a-cifar-10-image-classifier-on-windows-with-pytorch-9094e29089cd and this
    https://discuss.pytorch.org/t/brokenpipeerror-errno-32-broken-pipe-when-i-run-cifar10-tutorial-py/6224 how to solve it
    b) Error 'Can't pickle <class>: it's not the same object, see this blog 
    https://stackoverflow.com/questions/1412787/picklingerror-cant-pickle-class-decimal-decimal-its-not-the-same-object
    how to solve it
    c) in a function that expects a list of items, how can I pass a Python list item iteratively without getting an error? e.g. 
    my_list = ['red', 'blue', 'orange']
    function_that_needs_strings('red', 'blue', 'orange') # works!
    function_that_needs_strings(my_list) # error!
    answer: function_that_needs_strings(*my_list) # works!
    see more information: https://stackoverflow.com/questions/3480184/unpack-a-list-in-python
    
"""

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
import torchvision as tv
import torchvision.transforms as transforms
# from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

import numpy as np
import h5py
import math
import os
import time
import scipy.io
from torchvision.models import vgg19

import pytorch_ssim_l1
from optimizer import lookahead

"-------------------------------------------------------------------------------------------------"
print('boolean value to see if GPU is ready:', tc.cuda.is_available())
print('number of GPU is', tc.cuda.device_count())
print(tc.cuda.get_device_name(0))
use_cuda = True #-- boolean to choose GPU
since = time.clock()


batch_size = 32
EPOCH_NUM = 20
SELECTED_BATCH_FOR_PLOT_AND_SAVE_MAT_FILE = 10
Feature_Extractor_in_Front_of_Network = False
Use_Lookahead_Optimizer = False
Maintain_Same_Size = True # stand for whether we want the output SR Simage has same size or NOT(e.g. larger size) as input LR image


# --------------------------- configuration of parameters for RCAN --------------------------- #
args = {'n_resgroups': 3, 'n_rcablocks': 3, 'n_feats': 128, 'reduction': 16, 'scale': 4, }
# args['n_resgroups'] = 3, stands for number of RGs in RIR/RCAN
# args['n_rcablocks'] = 3, stands for number of RCABs in one RG
# args['n_feats'] = 128, stands for how many "number of channels" for feature map going through model
# args['reduction'] = 16, stands for reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
# args['scale'] = 4, stands for resize factor, e.g. 2, 4


"""""""""""""""""""""""""""""""""""""""""""""
1. MRI HR and LR Data pair preprocessing part
"""""""""""""""""""""""""""""""""""""""""""""
# =============================================================================
# h5py.version
# =============================================================================

"Note folder log path is changed for 2D_Gated_Dilated_ResNeXt_Based_MRI_SR_v2"
folder_log_path = '/home/HaoLi/SR/Data/'
file_names = os.listdir(folder_log_path)

num_low_resolution_mat_file = 0
num_high_resolution_groundtruth_mat_file = 0

for idx_file in file_names:
    print(idx_file)
    if 'LR' in os.path.join(folder_log_path, idx_file):
        print('One more low resolution image set exist')
        num_low_resolution_mat_file = num_low_resolution_mat_file + 1
        print(os.path.join(folder_log_path, idx_file))
        file_data_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
        data_low_resolution = file_data_low_resolution['LR'][:] #----- numpy array
        print(np.shape(data_low_resolution))
        torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_low_resolution = torch_data_low_resolution.permute(0, 2, 1)
# =============================================================================
#         torch_data_low_resolution = tc.t(torch_data_low_resolution)
# =============================================================================
# =============================================================================
#         torch_data_low_resolution = torch_data_low_resolution.type(tc.DoubleTensor)
# =============================================================================
        print(np.shape(torch_data_low_resolution))
        if num_low_resolution_mat_file == 1:
            torch_data_low_resolution_sequence = torch_data_low_resolution
        elif num_low_resolution_mat_file > 1:
            print(num_low_resolution_mat_file)
            torch_data_low_resolution_sequence = tc.cat((torch_data_low_resolution_sequence, torch_data_low_resolution), 0)
        print(np.shape(torch_data_low_resolution_sequence))
    elif 'HRGT' in os.path.join(folder_log_path, idx_file):
        print('One more high resolution groundtruth image set exist')
        num_high_resolution_groundtruth_mat_file = num_high_resolution_groundtruth_mat_file + 1
        print(os.path.join(folder_log_path, idx_file))
        file_data_high_resolution_groundtruth = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
        data_high_resolution_groundtruth = file_data_high_resolution_groundtruth['HRGT'][:] #----- numpy array
        print(np.shape(data_high_resolution_groundtruth))
        torch_data_high_resolution_groundtruth = tc.from_numpy(data_high_resolution_groundtruth) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.permute(0, 2, 1)
# =============================================================================
#         torch_data_high_resolution_groundtruth = tc.t(torch_data_high_resolution_groundtruth)
# =============================================================================
# =============================================================================
#         torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.type(tc.DoubleTensor)
# =============================================================================
        print(np.shape(torch_data_high_resolution_groundtruth))
        if num_high_resolution_groundtruth_mat_file == 1:
            torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth
        if num_high_resolution_groundtruth_mat_file > 1:
            print(num_high_resolution_groundtruth_mat_file)
            torch_data_high_resolution_groundtruth_sequence = tc.cat((torch_data_high_resolution_groundtruth_sequence, torch_data_high_resolution_groundtruth), 0)
        print(np.shape(torch_data_high_resolution_groundtruth_sequence))
    else:
        print('other type NOT support for now')
        

"Note for 2D matrix data with dimension H x W, everytime before loading into trainset and testset, we have to adapt the dimension into format N x C X H x W. Cause all\
the dimension of input/output are using N x C x H x W. N denotes number of data, C denotes number of channels, H means height, W stays width"        
torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.unsqueeze(1)    
print(np.shape(torch_data_low_resolution_sequence))
torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.unsqueeze(1)  
print(np.shape(torch_data_high_resolution_groundtruth_sequence))


num_training_samples = math.floor(torch_data_low_resolution_sequence.size(0)*0.8)
print(num_training_samples)
torch_data_low_resolution_training_sequence = torch_data_low_resolution_sequence[0: num_training_samples, :, :, :]
torch_data_high_resolution_groundtruth_training_sequence = torch_data_high_resolution_groundtruth_sequence[0: num_training_samples, :, :, :]
print(np.shape(torch_data_low_resolution_training_sequence))
print(np.shape(torch_data_high_resolution_groundtruth_training_sequence))


torch_data_low_resolution_test_sequence = torch_data_low_resolution_sequence[num_training_samples: -1, :, :, :]     
torch_data_high_resolution_groundtruth_test_sequence = torch_data_high_resolution_groundtruth_sequence[num_training_samples: -1, :, :, :]
print(np.shape(torch_data_low_resolution_test_sequence))
print(np.shape(torch_data_high_resolution_groundtruth_test_sequence))

        
print('All mat files have been concatenated into one tensor for each type, data is ready to be loaded!')

"""""""""""""""""""""""""""""""""""""""""""""
2. Load MRI HR and LR Data pair part
"""""""""""""""""""""""""""""""""""""""""""""
torch_data_low_resolution_training_sequence = torch_data_low_resolution_training_sequence.float()
torch_data_high_resolution_groundtruth_training_sequence = torch_data_high_resolution_groundtruth_training_sequence.float()

trainset = tc.utils.data.TensorDataset(torch_data_low_resolution_training_sequence, torch_data_high_resolution_groundtruth_training_sequence)

trainloader = tc.utils.data.DataLoader(
                    trainset, 
                    batch_size = batch_size,
                    shuffle = True, 
                    num_workers = 0)



testset = tc.utils.data.TensorDataset(torch_data_low_resolution_test_sequence, torch_data_high_resolution_groundtruth_test_sequence)

testloader = tc.utils.data.DataLoader(
                    testset, 
                    batch_size = batch_size,
                    shuffle = True, 
                    num_workers = 0)



"""
In the original paper which proposed RCAN(2018. Image Super-Resolution Using Very Deep Residual Channel Attention Networks, mentioned as "original RCAN paper" in following),
The general, relationship between module and sub-module, sub-sub.module, etc, is something like:
RCAN(Deep Residual Channel Attention Network) include "RIR(Residual in Residual module) + upsampling module"; RIR consists of several RG(Residual Group);
each RG consists of several RCAB(Residual Channel Attention Block)s; each RCAB include CA(Channel Attention Layer).
"""

"""""""""""""""""""""""""""""""""""""""
3. Define RCAN architecture part
"""""""""""""""""""""""""""""""""""""""

"calculate gradient map for any input image"
def calculate_gradient_map(img):
    vertical_edge_mask = tc.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_edge_mask = tc.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    vertical_edge_mask = vertical_edge_mask.float().unsqueeze(0).unsqueeze(0).to(device)
    horizontal_edge_mask = horizontal_edge_mask.float().unsqueeze(0).unsqueeze(0).to(device)
    
    gradient_vertical_map = F.conv2d(img, vertical_edge_mask, padding = 1, stride = 1,groups = 1)
    gradient_horizontal_map = F.conv2d(img, horizontal_edge_mask, padding = 1, stride = 1, groups = 1)

    gradient_map = abs(gradient_vertical_map) + abs(gradient_horizontal_map)

    return gradient_map


"default conv layer"
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


"FeatureExtractor"
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out
    
    
"Calculate the k space result" 
class FFT_K_SPACE(nn.Module):
    def __init__(self):
        super(FFT_K_SPACE, self).__init__()

    def forward(self, x):
        x = tc.unsqueeze(x, -1) #----- create the additional last dimension for input matrix with (N, C, H, W)
        x_complex = tc.cat((x, tc.zeros_like(x)), -1)
        k_space_result = tc.fft(x_complex, 2)
        # print(k_space_result.size())
#        out = tc.sqrt(tc.mul(k_space_result[:, :, :, :, 0], k_space_result[:, :, :, :, 0]) + tc.mul(k_space_result[:, :, :, :, 1], k_space_result[:, :, :, :, 1]))
#        return out
        return k_space_result
		

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = tc.Tensor(rgb_std)
        self.weight.data = tc.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * tc.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


"Channel Attention (CA) Layer"
class CALayer(nn.Module):
    """
    Channel Attention (CA) Layer, is basical block in RCAN. One CA forms one RCAB(Residual Channel Attention Block).
    See figure 3 of original RCAN paper 
    """
    def __init__(self, channel, reduction=16):
        """
        reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        """
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # global average pooling, output size is 1 for each channel
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), # W_d in CA
                nn.ReLU(inplace=True), # ReLU in CA
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), # W_u in CA
                nn.Sigmoid() # sigmoid in CA
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        # the x is the "feature maps over channels" in size C x H x W. The y now is actual the weights in size C x 1 x 1 which represents "channel statistics", 
        # it stands for how much "attention" expected to pay for each channel's feature map 
        return x * y


"Residual Channel Attention Block (RCAB)"
class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB): There are several RCABs belong to one RG(Residual Group).
    See figure 4 of original RCAN paper
    """
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2): # conv --> ReLU --> conv
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction)) # CA
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) # conv --> ReLU --> conv --> CA
        #res = self.body(x).mul(self.res_scale)
        res += x # local skip link of RCAB
        return res


"Residual Group (RG)"
class ResidualGroup(nn.Module):
    """
    RG(Residual Group): There are several RGs belong to one RIR(Residual in Residual module).
    See upper figure in figure 2 of original RCAN paper
    """
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_rcablocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_rcablocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size)) # last conv after several RCABs as show in figure 2
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) # RCAB_1 --> RCAB_2 --> ... --> RCAB_(n_rcablocks) --> conv
        res += x # short skip connection in RG
        return res


"Upsampler Module, implemented by employeed of sub-pixel conv"
class Upsampler(nn.Sequential):
    """
    Upsampling/Upscale module, used as last part of "SR reconstruction network model" if the network model employ the "post-upsampling mode".
    Beware the actual upsampling approach is "sub-pixel conv" (which is nn.PixelShuffle() in Pytorch) which was proposed in
    paper: "2016. Real-Time single image and video super-resolution using an efficient sub-pixel convolutional neural network"
    """
    def __init__(self, conv, scale, n_feats):
        super(Upsampler, self).__init__()
        if scale == 2:
            self.upsampler = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                nn.PixelShuffle(scale)
            ])
        elif scale == 4:
            self.upsampler = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                nn.PixelShuffle(2),
            ])
        else:
            raise ValueError("scale must be 2 or 4.")

    def forward(self, x):
        upsampled_x = self.upsampler(x)
        return upsampled_x


"Residual Channel Attention Network (RCAN)"
class RCAN_MRI_SR_2D(nn.Module):
    """
    RCAN(Deep Residual Channel Attention Network) = RIR(Residual in Residual module) + Upsampler Module.
    See bottom figure in figure 2 of original RCAN paper
    """
    def __init__(self, args, conv=default_conv):
        super(RCAN_MRI_SR_2D, self).__init__()

        n_resgroups = args['n_resgroups'] # number of RGs in RIR/RCAN
        n_rcablocks = args['n_rcablocks'] # number of RCABs in one RG
        n_colors = 3 # number of channels going of input of entire model
        n_feats = args['n_feats'] # number of feature maps/channels going through the entire model
        kernel_size = 3 # conv filter size used for all conv in RCAN
        reduction = args['reduction'] # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        scale = args['scale'] # resize factor, e.g. 2, 4
        act = nn.ReLU(True)
        
        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std) """
        # ----------------------------------------------------------------------------------------------------------------------- #
        
        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)] # the first conv layer in RCAN, show in figure 2 of RCAN paper

        # define body module. The RIR(Residual in Residual) which consists of n_resgroups RGs, show in figure 2 of RCAN paper
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_rcablocks=n_rcablocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module. The last stage is upsampling module and one more conv layer, show in figure 2 of RCAN paper
        modules_tail = [
            Upsampler(conv, scale, n_feats),
            conv(n_feats, args.n_colors, kernel_size)]

        # Add a downsize converter by using conv layer.
        # The purpose of original RCAN designed in original RCAN paper is to upscale scale times of LR image, the output from RIR has same size as input LR image. However, here in our task
        # we actually want to maintain output(SR) and input(LR) same as, it means we need to downsize the image before leave it into upscale module. That is the reason we need extra conv layer to perform 
        # down size with scale time first before going through upscale with scale time
        if (scale == 2):
            # W2=(W1−F+2P)/S+1, H2=(H1−F+2P)/S+1.
            self.down_size_converter = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2) 
        elif (scale == 4):
            self.down_size_converter = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2),
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
            ])
        else:
            raise ValueError("scale must be 2 or 4.")

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
         # do NOT understand why need this, may NOT be useful for us 
        """ x = self.sub_mean(x) """
        x = self.head(x) # input image goes through first conv layer

        res = self.body(x) # data goes through RIR(Residual in Residual)
        res += x # long skip connection of RIR

        if (Maintain_Same_Size == True):
            x = self.down_size_converter(x) # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output

        x = self.tail(res) # data goes through upsampling module and one more conv layer
         # do NOT understand why need this, may NOT be useful for us
        """ x = self.add_mean(x) """

        return x


        
device=tc.device("cuda" if use_cuda else "cpu")
our_rcan_mri_sr_2d = RCAN_MRI_SR_2D(args)
if tc.cuda.device_count()>1:
    our_rcan_mri_sr_2d=nn.DataParallel(our_rcan_mri_sr_2d)
our_rcan_mri_sr_2d.to(device)

print('this is our RCAN_MRI_SR_2D: ', our_rcan_mri_sr_2d)

feature_extractor = FeatureExtractor().to(device)
print('this is our FeatureExtractor: ', feature_extractor)

fft_k_space = FFT_K_SPACE().to(device)
print('this is our FFT_K_SPACE: ', fft_k_space)


"""""""""""""""""""""""""""""""""""""""
4. Setup optimization algorithm part
"""""""""""""""""""""""""""""""""""""""
"set an optimizer"
if (Use_Lookahead_Optimizer):
    base_opt = opt.Adam(our_rcan_mri_sr_2d.parameters(), lr=1e-3, betas=(0.9, 0.999)) #----- use Adam algorithm as based optimizer A
    optimizer = lookahead.Lookahead(base_opt, k=5, alpha=0.5) # Initialize Lookahead
else:
    # optimizer = opt.SGD(our_resnext.parameters(), lr = 0.0001, momentum=0.9, weight_decay = 1e-9)    #----- use SGD algorithm for all parameters of our_lenet, by learning rate 0.01 and Momentum is 0.9
    optimizer = opt.Adam(our_rcan_mri_sr_2d.parameters(), lr = 0.0001, eps = 1e-08, weight_decay = 1e-5)    #----- use Adam algorithm for all parameters of our_classifier
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

"set a loss function"
print('The loss function is MSE')
loss_function_MSE = nn.MSELoss().to(device)        #----- here use MSE loss

print('The loss function is L1')
loss_function_L1 = nn.SmoothL1Loss().to(device)       #----- smooth L1 loss

# print('The loss function is Cross Entropy')
# loss_function_CE = nn.CrossEntropyLoss().to(device)

print('The loss function is SSIM')
SSIM_function = pytorch_ssim_l1.SSIM().to(device)       #----- ssim loss

# =============================================================================
# print('The loss function is L1Loss')
# loss_function = nn.L1Loss(size_average = False).to(device) 
# =============================================================================
# =============================================================================
# print('The loss function is CrossEntropyLoss')
# loss_function = nn.CrossEntropyLoss().to(device)        #----- here use cross entropy loss
# =============================================================================



"""""""""""""""""""""""""""
5. Train the RCAN_MRI_SR_2D part
"""""""""""""""""""""""""""
"Train the RCAN_MRI_SR_2D"
tc.set_num_threads(10)  #----- Sets the number of OpenMP threads used for parallelizing CPU operations

for epoch in range(EPOCH_NUM):
    "Set training mode"
    our_rcan_mri_sr_2d.train()
# =============================================================================
#     print('This is the ', epoch, ' epoch')
# =============================================================================
    if (Use_Lookahead_Optimizer):
        optimizer.step()
    else:
        scheduler.step()
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
# =============================================================================
#         print('This is the ', i, ' batch for the ', epoch, ' epoch' )
# =============================================================================
        
        "load input data"
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        # print('The data have been loaded' )
        
        "clear all stored gradients if there exist"
        optimizer.zero_grad()
        # print('The optimizer has been cleared' )
        
        "forward prop"
        # outputs = our_resnext(inputs).double() #-- numpy arrays are 64-bit floating point and will be converted to torch.DoubleTensor standardly. Now, if you use them with your model, you'll need to make sure that your model parameters are also Double
        outputs = our_rcan_mri_sr_2d(inputs) #-- or using default float as type, however remember to cast the input from Double to Float
        # print(outputs.size())
        # print('the forward pass has been went')


        
        SR_copies = tc.cat((outputs, outputs, outputs), 1)
        # print(SR_copies.size())
        SR_features = feature_extractor(SR_copies)
        # print(SR_features.size())
                
        HR_copies = tc.cat((labels, labels, labels), 1)
        # print(HR_copies.size())
        HR_features = feature_extractor(HR_copies)
        # print(HR_features.size())
        
        
        
        SR_freq = fft_k_space(outputs)
        
        HR_freq = fft_k_space(labels)
        
        
        "calculate the gradients for all Variables during back prop"
        "vgg loss + pixel MSE loss + fft frequency loss, and we use weight_decay in Adam so that is L2 regularization"
        feature_map_loss = 0.01*loss_function_MSE(SR_features, HR_features)
        # feature_map_loss = 0.000000001*loss_function_CE(SR_features, HR_features)
#        print("feature_map_loss: ", feature_map_loss)
        # pixel_wise_loss = 10*loss_function_MSE(outputs, labels)
        pixel_wise_loss = 100*loss_function_L1(outputs, labels)
        
#        print("pixel_wise_loss: ", pixel_wise_loss)
        k_space_freq_loss = 0.01*(loss_function_MSE(SR_freq[:,:,:,:,0], HR_freq[:,:,:,:,0])+loss_function_MSE(SR_freq[:,:,:,:,1], HR_freq[:,:,:,:,1]))
#        print(loss_function_MSE(SR_freq[:,:,:,:,0], HR_freq[:,:,:,:,0]))
#        print(loss_function_MSE(SR_freq[:,:,:,:,1], HR_freq[:,:,:,:,1]))
#        print("k_space_freq_loss: ", k_space_freq_loss)
#        ssim_loss = 1-SSIM_function(outputs, labels)
        ssim_loss = loss_function_L1(SSIM_function(labels,labels),SSIM_function(outputs, labels))
#        print("ssim_loss: ", ssim_loss)

        gradient_map_loss = 100*loss_function_L1(calculate_gradient_map(outputs), calculate_gradient_map(labels))
        print('gradient_loss: ', gradient_map_loss)
        
#        loss = pixel_wise_loss + ssim_loss
        loss = pixel_wise_loss + feature_map_loss
        
#        if ssim_loss < 0.5:
#            loss = ssim_loss + feature_map_loss + pixel_wise_loss
#            print('ssim_loss')
#        else:
#            loss = pixel_wise_loss + feature_map_loss
            
        if tc.isnan(k_space_freq_loss) != 1:
            loss = loss + k_space_freq_loss
        
        if tc.isnan(ssim_loss) != 1:
            loss = loss + ssim_loss

        if tc.isnan(gradient_map_loss) != 1:
            loss = loss + gradient_map_loss
        
        print('loss: ', loss)
#        loss = feature_map_loss + pixel_wise_loss + k_space_freq_loss
        # print('the loss has been checked')

        
        "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"
#        if tc.isnan(loss) == 1: #- loss == 'NaN':
#            break
        "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"
            
        
        "back prop"
        loss.backward()
        # print('the backward pass has been went')
        
        "update all Variables by using newly fetched gradients"
        optimizer.step()
        # print('the all Variables have been updated')
        
        "print log info"
        running_loss += loss.data
        if i % 50 == 0: #----- print log info every 1000 batch
            if i == 0:
                print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss))
            else:
                print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / 50))
            
            training_loss_for_current_epoch = running_loss / 50
            feature_map_loss_for_current_epoch = feature_map_loss
            pixel_wise_loss_for_current_epoch = pixel_wise_loss
            ssim_loss_for_current_epoch = ssim_loss
            gradient_map_loss_for_current_epoch = gradient_map_loss
            k_space_freq_loss_for_current_epoch = k_space_freq_loss
            
            running_loss = 0.0
    "Set evaluation Mode"    
    our_rcan_mri_sr_2d.eval()
    for i, testing_data in enumerate(testloader, 0):
        
        if i == 10:
            LR_test, HR_test = testing_data
            # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
            LR_test, HR_test = Variable(LR_test).type(tc.FloatTensor).to(device), Variable(HR_test).type(tc.FloatTensor).to(device)
            
            SR_test = our_rcan_mri_sr_2d(LR_test)
            
            SR_test_copies = tc.cat((SR_test, SR_test, SR_test), 1)
            # print(SR_copies.size())
            SR_test_features = feature_extractor(SR_test_copies)
            # print(SR_features.size())
                    
            HR_test_copies = tc.cat((HR_test, HR_test, HR_test), 1)
            # print(HR_copies.size())
            HR_test_features = feature_extractor(HR_test_copies)
            # print(HR_features.size())
        
            SR_test_freq = fft_k_space(SR_test)
            
            HR_test_freq = fft_k_space(HR_test)
        
        
            "calculate the gradients for all Variables during back prop"
            "vgg loss + pixel MSE loss + fft frequency loss, and we use weight_decay in Adam so that is L2 regularization"
            feature_map_loss_test = 0.01*loss_function_MSE(SR_test_features, HR_test_features)
            print("feature_map_loss_test: ", feature_map_loss_test)
            
            pixel_wise_loss_test = 100*loss_function_L1(SR_test, HR_test)
            print("pixel_wise_loss_test: ", pixel_wise_loss_test)
            
            k_space_freq_loss_test = 0.01*(loss_function_MSE(SR_test_freq[:,:,:,:,0], HR_test_freq[:,:,:,:,0])+loss_function_MSE(SR_test_freq[:,:,:,:,1], HR_test_freq[:,:,:,:,1]))
            print("k_space_freq_loss_test: ", k_space_freq_loss_test)
            
            ssim_loss_test = loss_function_L1(SSIM_function(HR_test,HR_test),SSIM_function(SR_test, HR_test))
            print("ssim_loss_test: ", ssim_loss_test)
            
            gradient_map_loss_test = 100*loss_function_L1(calculate_gradient_map(SR_test), calculate_gradient_map(HR_test))
            print('gradient_loss_test: ', gradient_map_loss_test)
        
            loss_test = pixel_wise_loss_test + feature_map_loss_test + k_space_freq_loss_test + ssim_loss_test + gradient_map_loss_test
            print('loss_test: ', loss_test)
        
                
    "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"    
#    if tc.isnan(loss) == 1: #- loss == 'NaN':
#        break
    "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"
    
    
    "Save the training loss for each epoch"
    if (epoch == 0):
        f = open('result_UResNeXt_l1_4msssim_gradient_bnf_tcf_laf_150_32_4folds_2d_test.txt', 'w')
    f.write('Training Loss:')
    f.write('\n')    
    f.write('The feature_map_loss for epoch %d  is : %f' % (epoch, feature_map_loss_for_current_epoch))
    f.write('\n')
    f.write('The pixel_wise_loss for epoch %d  is : %f' % (epoch, pixel_wise_loss_for_current_epoch))
    f.write('\n')
    f.write('The k_space_freq_loss for epoch %d  is : %f' % (epoch, k_space_freq_loss_for_current_epoch))
    f.write('\n')
    f.write('The ssim_loss for epoch %d  is : %f' % (epoch, ssim_loss_for_current_epoch))
    f.write('\n')
    f.write('The gradient_map_loss for epoch %d  is : %f' % (epoch, gradient_map_loss_for_current_epoch))
    f.write('\n')
    f.write('The training loss for epoch %d  is : %f' % (epoch, training_loss_for_current_epoch))
    f.write('\n')
    f.write(' \n')
    f.write('Test Loss:')
    f.write('\n')    
    f.write('The feature_map_loss_test for epoch %d  is : %f' % (epoch, feature_map_loss_test))
    f.write('\n')
    f.write('The pixel_wise_loss_test for epoch %d  is : %f' % (epoch, pixel_wise_loss_test))
    f.write('\n')
    f.write('The k_space_freq_loss_test for epoch %d  is : %f' % (epoch, k_space_freq_loss_test))
    f.write('\n')
    f.write('The ssim_loss_test for epoch %d  is : %f' % (epoch, ssim_loss_test))
    f.write('\n')
    f.write('The gradient_map_loss_test for epoch %d  is : %f' % (epoch, gradient_map_loss_test))
    f.write('\n')
    f.write('The training_loss_test for epoch %d  is : %f' % (epoch, loss_test))
    f.write('\n')
    f.write(' \n')
    f.write(' \n')
    if (epoch == EPOCH_NUM - 1):
        f.close()
            
print("training complete")


"Set evaluation mode"
our_rcan_mri_sr_2d.eval()
"Resetup the batch size for train and test data, to avoid the errorCUDA out of memory"
new_batch_size_for_checking = 16
trainloader = tc.utils.data.DataLoader(
                    trainset, 
                    batch_size = new_batch_size_for_checking,
                    shuffle = True, 
                    num_workers = 0)

testloader = tc.utils.data.DataLoader(
                    testset, 
                    batch_size = new_batch_size_for_checking,
                    shuffle = True, 
                    num_workers = 0)


"exam the generated SR MRI image by using training LR image data and save them"
# =============================================================================
# for data in testloader:
# =============================================================================
for i, training_data_2 in enumerate(trainloader, 0):
    
    LR_images_training, HR_images_training = training_data_2
    
    # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
    outputs = our_rcan_mri_sr_2d(Variable(LR_images_training).type(tc.FloatTensor).to(device))
# =============================================================================
#     print(outputs.data.size())
# =============================================================================
    
    if (i == math.floor((torch_data_low_resolution_training_sequence.size(0)/new_batch_size_for_checking)/2)): 
        
        LR_images_training, HR_images_training = training_data_2
        # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
        outputs = our_rcan_mri_sr_2d(Variable(LR_images_training).type(tc.FloatTensor).to(device))
        
        #----- skip display "the last batch for one epoch test data" and skip "all the batches expect the batch in the middle"
        SR_images_tensor_training = outputs.data.cpu().squeeze(1)
        # HR_images_tensor = HR_images_temp.cpu().squeeze(1)
# =============================================================================
#         print(SR_images_tensor.size())
# =============================================================================
        # print(HR_images_tensor.size())    
        SR_images_exam_train = SR_images_tensor_training.numpy()
        # HR_images_test = HR_images_tensor.numpy()
        
        "save the .mat files for SR LR, HR training images"
        scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4msssim_gradient_bnf_tcf_laf_150_32_4folds_2d_test/HR_training_image.mat', mdict = {'HR_training_image' : HR_images_training.numpy()})
        scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4msssim_gradient_bnf_tcf_laf_150_32_4folds_2d_test/LR_training_image.mat', mdict = {'LR_training_image' : LR_images_training.numpy()})
        scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4msssim_gradient_bnf_tcf_laf_150_32_4folds_2d_test/SR_training_image.mat', mdict = {'SR_training_image' : SR_images_exam_train})
        
        
    
#        for j in range(new_batch_size_for_checking):
#            plt.imshow(SR_images_exam_train[j, :, :])
#            plt.savefig('/home/HaoLi/SR/Results/training_' + str(j) + '_SR_image.png')
#            plt.show()
#            plt.imshow(HR_images_training[j, 0, :, :])
#            plt.savefig('/home/HaoLi/SR/Results/training_' + str(j) + '_HR_image.png')
#            plt.show()
#            plt.imshow(LR_images_training[j, 0, :, :])
#            plt.savefig('/home/HaoLi/SR/Results/training_' + str(j) + '_LR_image.png')
#            plt.show()

print("examination of generated SR image by using training samples complete")




"predict the SR MRI image by using testing LR image data and save them"
# =============================================================================
# for data in testloader:
# =============================================================================
for i, testing_data_2 in enumerate(testloader, 0):
    
    LR_images_test, HR_images_test = testing_data_2
    # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
    outputs = our_rcan_mri_sr_2d(Variable(LR_images_test).type(tc.FloatTensor).to(device))
# =============================================================================
#     print(outputs.data.size())
# =============================================================================
    
    if (i == math.floor((torch_data_low_resolution_test_sequence.size(0)/new_batch_size_for_checking)/2)): 
        
        LR_images_test, HR_images_test = testing_data_2
        # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
        outputs = our_rcan_mri_sr_2d(Variable(LR_images_test).type(tc.FloatTensor).to(device))
        
        #----- skip display "the last batch for one epoch test data" and skip "all the batches expect the batch in the middle"
        SR_images_tensor_test = outputs.data.cpu().squeeze(1)
        # HR_images_tensor = HR_images_temp.cpu().squeeze(1)
# =============================================================================
#         print(SR_images_tensor.size())
# =============================================================================
        # print(HR_images_tensor.size())    
        SR_images_test = SR_images_tensor_test.numpy()
        # HR_images_test = HR_images_tensor.numpy()
        
        "save the .mat files for SR, HR and LR training images"
        scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4msssim_gradient_bnf_tcf_laf_150_32_4folds_2d_test/SR_test_image.mat', mdict = {'SR_test_image' : SR_images_test})
        scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4msssim_gradient_bnf_tcf_laf_150_32_4folds_2d_test/HR_test_image.mat', mdict = {'HR_test_image' : HR_images_test.numpy()})
        scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4msssim_gradient_bnf_tcf_laf_150_32_4folds_2d_test/LR_test_image.mat', mdict = {'LR_test_image' : LR_images_test.numpy()})
        
    
#        for j in range(new_batch_size_for_checking):
#            plt.imshow(SR_images_test[j, :, :])
#            plt.savefig('/home/HaoLi/SR/Results/testing_' + str(j) + '_SR_image.png')
#            plt.show()            
#            plt.imshow(HR_images_test[j, 0, :, :])
#            plt.savefig('/home/HaoLi/SR/Results/testing_' + str(j) + '_HR_image.png')
#            plt.show()            
#            plt.imshow(LR_images_test[j, 0, :, :])
#            plt.savefig('/home/HaoLi/SR/Results/testing_' + str(j) + '_LR_image.png')
#            plt.show()            

print("the predicting of generated SR image by using testing samples complete")
