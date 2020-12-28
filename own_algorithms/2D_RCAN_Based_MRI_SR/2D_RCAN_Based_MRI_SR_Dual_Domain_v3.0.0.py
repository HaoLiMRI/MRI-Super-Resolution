# -*- coding: utf-8 -*-
"-------------------------------------------------------------------------------------------------"
"""
2D_RCAN_Based_MRI_SR_Dual_Domain Reconstruct 
(2D Deep Residual Channel Attention Network Based Dual Domain Fusion Network for MRI Super-Resolution Image Reconstruction)
This is the code for RCAN based Dual Domain Fusion Network for MRI SR Reconstruction, RCAN model proposed to generate super
resolution(SR) image in the paper: 
2018.Image Super-Resolution Using Very Deep Residual Channel Attention Networks
https://arxiv.org/abs/1807.02758
"""
"""
Author: chisyliu@hotmail.com *
        hao.li@med.uni-heidelberg.de *
        
        * Both authors contribute equally
Version: 3.0.0(Stable Version, even deformable conv works at least for RCAN network)
"""
"-------------------------------------------------------------------------------------------------"
"""
This is the current version we are working on, in 20201228
This is a demo code of 2D_RCAN_Based_MRI_SR_Dual_Domain. in this version we have already support following items:
    0)  Dual Domain Fusion Network Achitecture, where we already support:
        a) RCAN as single branch
        b) gradient map branch as secondary branch, together with RCAN as main(image) branch
        b) k space branch as secondary branch, together with RCAN as main(image) branch
        c) high frequency component extracted using wavelet transformation and wavelet branch as 
            secondary branch, together with RCAN as main(image) branch
        d) interleaving fusion between image branch and  as secondary branch at intermedian level
        e) late stage fusion of outcome from image branch and outcome from secondary branchbranch
        we will plan to support:
        a) dual regression loss(See paper: 2020.Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution)
    Besides, we already support other features:
    1)  option to add a VGG feature extractor in front of the main "CNN based Reconstruct network"(so the input to "CNN based Reconstruct network" is feature map of LR image)
    2)  either Pixel-Wise MSE loss or Pixel-Wise smooth L1 loss
    3)  weighted k space loss(fft loss)
    4)  weighted VGG loss
    5)  L2 Regularization
    6)  option to add ssim smooth L1 loss
    7)  option to add gradient map smoothL1 loss
    8） option to add gram matrix smoothL1 loss(for increasing texture similarity between SR and HR)
	9)  option to use lookahead optimizer
    10) option to use CosineAnnealingLR learning rate decay. See https://blog.zhujian.life/posts/6eb7f24f.html for more info 
    11) option to choose different gradient_operator
    12) option to use coord_conv(See [20])
    13) option to use deformable_conv(However we are running in CUDA out of memory, although already used tc.cuda.empty_cache() for deformable_conv)
    14) option to use py_conv(See [24])
    15) option to use Sine(SIREN) activation function
    16) option to use FReLU activation function
    17) option to use channel attention for cross branch fusion
    18) option to use "1 - exp(-alpha*gradient_map)" to amplify small values in gradient map to emphasize the information from 
        gradient values which stand for texture
    19) option to amplify high frequency loss value in k space loss. We can use weight for "square of difference between one element of SR k space data matrix and corresponding element
        of HR k space data matrix" which follows the 2D Gaussian distribution. Cause the center part of k space data matrix
        stands for high frequency part where we observed the most significant mismatch happened between SR and HR k space 
        data, we expect to give higher weight on center part of difference and lower weight on edge part of difference in the
        k space data matrix for the k space loss.
    20) option to use "warm up learning rate" to set up the relative small learning rate in the beginning of training, then change to normal scheduler
        to apply the large learning rate and lower the learning rate graduately. We could avoid using large learning rate in the very beginning by doing
        such, thus avoid "unstable" training(e.g. The loss becomes very large suddenly in the first several epoch).
        Warm up指的是用一个小的学习率先训练几个epoch，这是因为网络的参数是随机初始化的，假如一开始就采用较大的学习率容易出现数值不稳定，这也是为什么要使用Warm up。
        然后等到训练过程基本上稳定了就可以使用原始的初始学习率进行训练了。
    21) Re-implement deformable conv filter(search ConvOffset2D), to make it runnable now without memory problem. Only tested with RCAN network, defaul conv, ReLU.
    22) option to use Dynamic ReLU Type A and Type B activation function.
    
    23) option to set up args['number_of_progressive_stage'] as 1, 2, 3, to support args['scale']^args['number_of_progressive_stage'] progressive 4x and 8x upsampling reconstruction.
    24) option to use "end to end channel and spatial attention block" for upsampler (inside upsampler, after first conv and before pixel shuffle).
    25) For all the "end to end channel and spatial attention block", either "sequential_mode" or "parallel_mode" could be selected。
    
    we will plan to support other features:
    1) multi-kernel size deformable conv in different paths and fuse together, see [26] for similar idea
    2) kernel size wise attention[25] for multi-kernel size deformable conv
    3) multi-kernel size dilated conv in different paths and fuse together[26]
    4) kernel size wise attention[25] for multi-kernel size dilated conv
    5) feature scale wise attention for py_conv
    6) spatial attention(additional to channel attention) for normal processing inside each branch. One simple way of adding spatial attention is use SAM mechanism proposed in CBAM paper.
    7) "end to end spatial and channel attention in one 3D conv format" for normal processing inside each branch.
        Another simple way of ultilizing the "end to end spatial and channel attention" but NOT using 3D conv format, is, make spatial and channel attention in parallel. 
        See BAM mechanism proposed in paper: 2018.BAM: Bottleneck Attention Module.
        Another simple way of ultilizing the "end to end spatial and channel attention" but NOT using 3D conv format, is, also make spatial and channel attention in parallel.
        See DANet mechanism proposed in paper: 2018.Dual Attention Network for Scene Segmentation.
        类似的双重注意力模式还有scSE注意力，有兴趣的可以自行查看.
    8) set threshold_low and threshold_high for "contrast between each pixel and all the pixels around it", if contrast is
        lower than threshold_high, we have to limit the contrast to let it should be larger than threshold_low. The actual
        threshold_low for contrast between each pixel and all the pixels around it may follow Gaussian distribution(The 
        closer the pixels are the larger threshold_low should be, doing like this lead the contrast between two pixels which
        are close to each other large enough, so they will NOT be samiliar and the super resolution result will NOT be too
        smooth in texture wise.)
    9) consider using channel attenion to assign different weights for different channel for upscaling, for upsmaler module which is commonly used for high resolution task together with . 
        We could see similar idea in paper: 
        2020.Detecting Small Objects Using a Channel-Aware Deconvolutional Network.
        2020.Attention-based Image Upsampling
        https://arxiv.org/abs/2012.09904
    10)consider adding FPN structure into the current framework, to concatenate the feature maps in different sizes from different layer of "encoder" to the 
        corresponding(feature map with same size as the feature map in particular layer if "encoder") layers in "decoder".
We also fixed bugs from previous versions, typical ones like:
    1) After PyTorch version 1.1, call scheduler.step() will overwrite the learning rate used in optimizer to be same as scheduler sets up immediately.
    2) Also beware the scheduler.step() should be called every epoch rather than every batch. It means scheduler.step() should
        only be after and outside of "for loop of batch".
    3) Previsouly the neural network has been initialized properly. So we add xavier and Kaiming initialization in this version of code.(but
        might not be really used since it is NOT common to use weight initialization for super resolution task).
    4) Add the code to always select the weights of network which provides the best value in average SSIM over all batches for validation
        in one epoch, and save the selected weights of network and corresponding LR and SR data.
    5) When accumulate the loss in the training step, only add the value of loss into by using loss_bullet.item(), e.g. ssim_loss_training += ssim_loss.item(), rather than adding the entire
        computational graph into(e.g. ssim_loss_training += ssim_loss). Thus avoid using too much GPU memory which is not necessary.
    6) Replace the mean SSIM (a single value) by using SSIM map (a matrix) in the ssim loss.
       
    
    In this 2D version, the data format has been changed. The input data is just 64 x 64 2D matrix rather than 64 x 64 x 64, we already 
    collapse all the 64 layers into only one layer in the data tailing and noise filtering processing.
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
    [20]2018.An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution. https://arxiv.org/abs/1807.03247
    [21]2020.Implicit Neural Representations with Periodic Activation Functions. https://arxiv.org/abs/2006.09661
    [22]2020.Funnel Activation for Visual Recognition. https://arxiv.org/abs/2007.11824
    [23]2017.Deformable Convolutional Networks. https://arxiv.org/abs/1703.06211
    [24]2020.Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition. https://arxiv.org/abs/2006.11538
    [25]2019.Selective Kernel Networks
    [26]2020.Perceptual Extreme Super Resolution Network with Receptive Field Block
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
    d) After PyTorch version 1.1, call scheduler.step() will overwrite the learning rate used in optimizer to be same as scheduler sets up immediately.
    
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
from math import exp
import numpy as np
import h5py
import math
import os
import time
import scipy.io
from torchvision.models import vgg19
from pytorch_wavelets import DWT, IDWT # (or import DWTForward, DWTInverse)
import copy
import pickle

from ultility import pytorch_ssim_l1
from ultility import pytorch_ssim_map
from ultility.optimizer import lookahead

from ultility.deform_conv import th_batch_map_offsets, th_generate_grid # For supporting deformable conv filter

"-------------------------------------------------------------------------------------------------"
Single_GPU_training = True

if Single_GPU_training == True:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print('boolean value to see if GPU is ready:', tc.cuda.is_available())
print('number of GPU is', tc.cuda.device_count())
print(tc.cuda.get_device_name(0))
use_cuda = True
since = time.perf_counter()

"""""""""""""""""""""""""""""""""""""""""""""
0. Configure all parameter
"""""""""""""""""""""""""""""""""""""""""""""
# --------------------------- configuration of support parameters --------------------------- #
batch_size = 8
EPOCH_NUM = 12
SELECTED_BATCH_FOR_PLOT_AND_SAVE_MAT_FILE = 10
Feature_Extractor_in_Front_of_Network = False # stand for whether we use feature extractor in front of network
Maintain_Same_Size = False # stand for whether we want the output SR Simage has same size or NOT(e.g. larger size) as input LR image
Use_SSIM_L1_Loss = True # stand for whether we want use SSIM L1 loss in the total loss function
Use_Gradient_Map_L1_Loss = True # stand for whether we want use gradient map L1 loss in the total loss function
Use_Gram_Matrix_L1_Loss = False # stand for whether we want use gram matrix L1 loss(between SR and HR, for increasing texture similarity between SR and HR) in the total loss function
Use_Channel_Attention_For_Cross_Branch_Fusion = True # stand for whether we give weight for every channel of feature maps(from both image and secondary branch) before they fuse together
Amplify_Small_Value_In_Gradient_Map = False # stand for whether we want to amplify small values in gradient map to emphasize the information from gradient values which stand for texture
Amplify_High_Frequency_Value_In_K_Space_Loss = False # stand for whether we want to amplify high frequence loss values in k space loss

# If we want to plot some information
plot_the_gradient_map_of_input_image = False
plot_the_k_space_data_of_input_image = False
plot_the_wavelets_transform_data_of_input_image = False

# --------------------------- configuration of parameters for RCAN --------------------------- #
args = {'n_resgroups': 3, 'n_rcablocks': 5, 'n_feats': 64, 'reduction': 16, 'scale': 2, 'number_of_progressive_stage': 2, \
    'use_channel_and_spatial_attention_inside_upsampler': True, 'channel_and_spatial_attention_mode': 'parallel_mode',\
    'conv_layer_type': 'default_conv', 'activation_function_type': 'Dynamic_ReLU_Type_B', 'type_of_network': 'RCAN', \
    'gradient_operator': 'sobel', 'optimizer': 'Adam', 'learning_rate_decay_method': 'cosine_learning_rate_warm_restarts', \
    'use_learning_rate_warm_up': True, 'how_many_epoch_to_be_used_for_warm_up': 10, 'initial_learning_rate_after_warm_up': 0.0001}
args_loss_weight = {'feature_map_weight': 20, 'pixel_wise_weight': 20000, 'k_space_weight': 2, 'ssim_weight': 100, \
                    'gradient_img_weight': 1000, 'gradient_grd_weight': 10, 'k_space_branch_weight': 0.02, \
                    'wavelets_branch_weight': 5, 'gram_similarity_weight': 5, \
                    'ssim_luminance_weight': 2, 'ssim_contrast_weight': 2, 'ssim_structure_weight': 4}
# args['n_resgroups'] = 20, stands for number of RGs in RIR/RCAN (per stage)
# args['n_rcablocks'] = 10, stands for number of RCABs in one RG (per stage)
# args['n_feats'] = 128, stands for how many "number of channels" for feature map going through model
# args['reduction'] = 16, stands for reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
# args['scale'] = 2, stands for scale factor used in one upsampler, e.g. 2, 4
# args['number_of_progressive_stage'] = 2, stands for number of stages(number of "MRI_SR_Dual_Domain_2D network"), e.g. 1, 2, 3, to ultilize progressive upsampling
# args['use_channel_and_spatial_attention_inside_upsampler'] = True, stands for whether we use channel and spatial attention block inside upsampler, e.g. True, False
# args['channel_and_spatial_attention_mode'] = 'sequential_mode', stands for which end to end channel and spatial block to use, e.g. 'sequential_mode', 'parallel_mode'
# args['conv_layer_type'] = 'default_conv', stands for type of conv layer, e.g. 'default_conv', 'coord_conv', 'deformable_conv', 'py_conv'
# args['activation_function_type'] = 'ReLU', stands for type of activation function, e.g. 'ReLU'. 'Sine', 'FReLU', 'Dynamic_ReLU_Type_A', 'Dynamic_ReLU_Type_B'
# arg['type_of_network'] == 'RCAN', stands for type of network, e.g. 'RCAN', 'gradient_map_dual_domain', 'k_space_dual_domain', 'wavelets_transform_dual_domain'
# arg['gradient_operator'] = ['sobel'] # stand for which gradient operator we want use for calculating gradient map, e.g. 'sobel', 'canny'
# arg['optimizer'] = ['Adam'] # stand for which optimizer we want use for training, e.g. 'Adam', 'SGD_with_momentum', 'look_ahead'
# arg['learning_rate_decay_method'] = ['cosine_learning_rate_decay'] # stand for which learning rate decay method we want use for training, e.g. 'cosine_learning_rate_decay', 'multi_step_learning_rate', 'step_learning_rate', 'cosine_learning_rate_warm_restarts'


"""""""""""""""""""""""""""""""""""""""""""""
1.1 MRI HR and LR Data pair preprocessing training part
"""""""""""""""""""""""""""""""""""""""""""""
# =============================================================================
# h5py.version
# =============================================================================

"The folder where to load the training LR, HR data pair"
folder_log_path = 'D:/Tech_Resource/Paper_Resource/MRI SR以及相关论文/our_project_code/data/sample_downsize_training_data_20200728'
file_names = os.listdir(folder_log_path)

num_low_resolution_mat_file = 0
num_high_resolution_groundtruth_mat_file = 0

for idx_file in file_names:
    print(idx_file)
    if 'LR_training_4' in os.path.join(folder_log_path, idx_file):
        print('One more low resolution image set exist')
        num_low_resolution_mat_file = num_low_resolution_mat_file + 1
        print(os.path.join(folder_log_path, idx_file))
        file_data_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
        data_low_resolution = file_data_low_resolution['LR'][:] #----- numpy array
        print(np.shape(data_low_resolution))
        print(data_low_resolution.dtype)
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
    elif 'HRGT_training_4' in os.path.join(folder_log_path, idx_file):
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


num_training_samples = math.floor(torch_data_low_resolution_sequence.size(0))
#print(num_training_samples)
#torch_data_low_resolution_training_sequence = torch_data_low_resolution_sequence[0: num_training_samples, :, :, :]
#torch_data_high_resolution_groundtruth_training_sequence = torch_data_high_resolution_groundtruth_sequence[0: num_training_samples, :, :, :]
#print(np.shape(torch_data_low_resolution_training_sequence))
#print(np.shape(torch_data_high_resolution_groundtruth_training_sequence))


#torch_data_low_resolution_test_sequence = torch_data_low_resolution_sequence[num_training_samples: -1, :, :, :]     
#torch_data_high_resolution_groundtruth_test_sequence = torch_data_high_resolution_groundtruth_sequence[num_training_samples: -1, :, :, :]
#print(np.shape(torch_data_low_resolution_test_sequence))
#print(np.shape(torch_data_high_resolution_groundtruth_test_sequence))


print('All mat files have been concatenated into one tensor for each type, data is ready to be loaded!')

"""""""""""""""""""""""""""""""""""""""""""""
1.2 Load MRI HR and LR Data pair training part
"""""""""""""""""""""""""""""""""""""""""""""
torch_data_low_resolution_training_sequence = torch_data_low_resolution_sequence.float()
torch_data_high_resolution_groundtruth_training_sequence = torch_data_high_resolution_groundtruth_sequence.float()

# tc.multiprocessing.freeze_support()

trainset = tc.utils.data.TensorDataset(torch_data_low_resolution_training_sequence, torch_data_high_resolution_groundtruth_training_sequence)

trainloader = tc.utils.data.DataLoader(
                    trainset, 
                    batch_size = batch_size,
                    shuffle = True, 
                    num_workers = 0)



#testset = tc.utils.data.TensorDataset(torch_data_low_resolution_test_sequence, torch_data_high_resolution_groundtruth_test_sequence)

#testloader = tc.utils.data.DataLoader(
#                    testset, 
#                    batch_size = batch_size,
#                    shuffle = True, 
#                    num_workers = 0)


"""""""""""""""""""""""""""""""""""""""""""""
2.1 MRI HR and LR Validation Data pair preprocessing validation part
"""""""""""""""""""""""""""""""""""""""""""""
num_low_resolution_mat_file = 0
num_high_resolution_groundtruth_mat_file = 0

for idx_file in file_names:
    print(idx_file)
    if 'LR_validation_4' in os.path.join(folder_log_path, idx_file):
        print('One more low resolution image set exist')
        num_low_resolution_mat_file = num_low_resolution_mat_file + 1
        print(os.path.join(folder_log_path, idx_file))
        file_data_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
        data_low_resolution = file_data_low_resolution['LR'][:] #----- numpy array
        print(np.shape(data_low_resolution))
        print(data_low_resolution.dtype)
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
    elif 'HRGT_validation_4' in os.path.join(folder_log_path, idx_file):
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


num_validation_samples = math.floor(torch_data_low_resolution_sequence.size(0))
#print(num_training_samples)
#torch_data_low_resolution_training_sequence = torch_data_low_resolution_sequence[0: num_training_samples, :, :, :]
#torch_data_high_resolution_groundtruth_training_sequence = torch_data_high_resolution_groundtruth_sequence[0: num_training_samples, :, :, :]
#print(np.shape(torch_data_low_resolution_training_sequence))
#print(np.shape(torch_data_high_resolution_groundtruth_training_sequence))


#torch_data_low_resolution_test_sequence = torch_data_low_resolution_sequence[num_training_samples: -1, :, :, :]     
#torch_data_high_resolution_groundtruth_test_sequence = torch_data_high_resolution_groundtruth_sequence[num_training_samples: -1, :, :, :]
#print(np.shape(torch_data_low_resolution_test_sequence))
#print(np.shape(torch_data_high_resolution_groundtruth_test_sequence))


print('All mat files have been concatenated into one tensor for each type, data is ready to be loaded!')

"""""""""""""""""""""""""""""""""""""""""""""
2.2 Load MRI HR and LR Validation Data pair validation part
"""""""""""""""""""""""""""""""""""""""""""""
torch_data_low_resolution_validation_sequence = torch_data_low_resolution_sequence.float()
torch_data_high_resolution_groundtruth_validation_sequence = torch_data_high_resolution_groundtruth_sequence.float()

# tc.multiprocessing.freeze_support()

validationset = tc.utils.data.TensorDataset(torch_data_low_resolution_validation_sequence, torch_data_high_resolution_groundtruth_validation_sequence)

validationloader = tc.utils.data.DataLoader(
                    validationset, 
                    batch_size = batch_size,
                    shuffle = False, 
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

"calculate gram matrix for MRI image"
def calculate_gram_matrix(input_image):
    b, c, h, w = input_image.size()
    F = input_image.view(b, c, h*w)
    G = tc.bmm(F, F.transpose(1, 2)) 
    G.div_(h*w)
    return G


"calculate 2D Gaussian weight for each 'element wise difference' in k space loss"
def gaussian(window_size, sigma):
    gauss = tc.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_2d_Gaussian_weights(window_size, num_of_samples, channel):
    '''
    Create a grid of weights which follow 2D Gaussian distribution(the weights at center area of grid are higher and weights at rest area of grid
    are lower). This function generates the weights which could emphasize the high frequency components(e.g. edge in the image) in the k space loss
    cause high frequency compoenents in k space are centrolized in the center area of k space data. 
    '''
    weights_in_1D_window = gaussian(window_size = window_size, sigma = 32).unsqueeze(1) # window_size is "how many weights we expect to generate over a Gaussian pdf
    weights_in_2D_window = weights_in_1D_window.mm(weights_in_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    weights_in_2D_window_pytorch = Variable(weights_in_2D_window.expand(num_of_samples, channel, window_size, window_size).contiguous())
    weights_in_2D_window_pytorch = weights_in_2D_window_pytorch/tc.max(weights_in_2D_window_pytorch)
    return weights_in_2D_window_pytorch


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


"calculate gradient map for any input MRI image"
def calculate_gradient_map(img):
    # sobel operator
    vertical_edge_mask = tc.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_edge_mask = tc.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    vertical_edge_mask = vertical_edge_mask.float().unsqueeze(0).unsqueeze(0).to(device)
    horizontal_edge_mask = horizontal_edge_mask.float().unsqueeze(0).unsqueeze(0).to(device)

    gradient_vertical_map = F.conv2d(img, vertical_edge_mask, padding = 1, stride = 1, groups = 1)
    gradient_horizontal_map = F.conv2d(img, horizontal_edge_mask, padding = 1, stride = 1, groups = 1)

    gradient_map = abs(gradient_vertical_map) + abs(gradient_horizontal_map)

    if Amplify_Small_Value_In_Gradient_Map == True:
        gradient_map = 1 - tc.exp(-2.5 * gradient_map) # 1 - exp(-ax), a = 2.5

    return gradient_map


"default conv layer"
def default_conv(in_channels, out_channels, kernel_size, bias = True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


"""(Not used yet in this code)Calculate PSNR for MRI image in shape (N, C, H, W)"""
def calc_psnr_for_mri_image(img1, img2):
    ### args:
        # img1: pytorch tensor, shape is [N, C, H, W]
        # img2: pytorch tensor, shape is [N, C, H, W]

    diff = (img1 - img2)
    mse = tc.pow(diff, 2).mean(2).mean(2)
    return -10 * tc.log10(mse).mean(0).mean(0)


"""
(Not used yet in this code)
L1 Charbonnier Loss. See more information regarding L1 Charboniier Loss from paper: 2018.Fast and Accurate Image Super-Resolution with 
Deep Laplacian Pyramid Networks. L1 Charboniier Loss in theory can be used to replace the (smooth) L1 loss, to provide reconstructed 
image with less over-smoothing issues and problem.
"""
class L1_Charbonnier_Loss(tc.nn.Module):
    def __init__(self):
        super(L1_Charbonnier_Loss,self).__init__()
        self.eps = 1e-4

    def forward(self, X, Y):
        diff = tc.add(X, -Y)
        error = tc.sqrt(diff * diff + self.eps)
        loss = tc.mean(error)
        print(loss.size())
        return loss


"Pyramidal Convolution(Py_Conv) Layer"
"""
2020.Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition. https://arxiv.org/abs/2006.11538
"""
class PyConv4(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0], bias=False)
        self.conv2_2 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1], bias=False)
        self.conv2_3 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2], bias=False)
        self.conv2_4 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3], bias=False)
    def forward(self, x):
        return tc.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)

class PyConv3(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0], bias=False)
        self.conv2_2 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1], bias=False)
        self.conv2_3 = nn.Conv2d(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2], bias=False)
    def forward(self, x):
        return tc.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)

class PyConv2(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = nn.Conv2d(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0], bias=False)
        self.conv2_2 = nn.Conv2d(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1], bias=False)
    def forward(self, x):
        return tc.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)

def py_conv(in_channels, out_channels, kernel_size, bias = False):
    # Some default settings for py_conv
    num_of_kernels = 3  # Note: this could be changed
    stride = 1
    if in_channels == 1 or in_channels == 2 or out_channels == 1 or out_channels == 2:
        # in_channels == 1 or out_channels == 1 means it is the 2D MRI image(in downsampling or upsmapling), py_conv only apply 
        # for feature map at image branch rather than on the 2D MRI image directly;
        # in_channels == 2 or out_channels == 2 means it is either the k space branch complex value data or it is the "modules_end_stage_fusion_of_outcome",
        # as explained above py_conv only apply for feature map at image branch
        return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias) # just return default conv for 2D MRI image
    else:
        if num_of_kernels == 1:
            raise ValueError(("Only 1 kernel size can NOT form py_conv")) 
        elif num_of_kernels == 2:
            pyconv_kernels = [kernel_size, kernel_size + 2]
            return PyConv2(in_channels, out_channels, pyconv_kernels=pyconv_kernels, stride=stride)
        elif num_of_kernels == 3:
            pyconv_kernels = [kernel_size, kernel_size + 2, kernel_size + 4]
            return PyConv3(in_channels, out_channels, pyconv_kernels=pyconv_kernels, stride=stride)
        elif num_of_kernels == 4:
            pyconv_kernels = [kernel_size, kernel_size + 2, kernel_size + 4, kernel_size + 6]
            return PyConv4(in_channels, out_channels, pyconv_kernels=pyconv_kernels, stride=stride)


"Coordinate Conv Layer"
"""
2018.An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution. https://arxiv.org/abs/1807.03247
"""
class AddCoords(nn.Module):
    def __init__(self, with_r = False):
        super().__init__()
        self.with_r = with_r    # If with_r = Ture: cancatenate another channel using "distance" between two index u, v

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = tc.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = tc.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = tc.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = tc.sqrt(tc.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + tc.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = tc.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r = False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r = with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
#        print("implement coord_conv layer in network")
        ret = self.conv(ret)
        return ret

def coord_conv(in_channels, out_channels, kernel_size, bias = True):
    return CoordConv(
        in_channels, out_channels, with_r = False, kernel_size = kernel_size,
        padding = (kernel_size//2), bias=bias)


"Deformable Conv Layer"
''' One approach to implement deformable conv layer '''
class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, use Modulated Defomable Convolution(Deformable ConvNets v2, 
            see 2018. Deformable ConvNets v2: More Deformable, Better Results. https://arxiv.org/abs/1811.11168).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x) # there are 2*kernel_size*kernel_size offset for each conv filter kernel(x, y axis offset for each element in the conv filter kernel)
        if self.modulation:
            m = tc.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = tc.cat([tc.clamp(q_lt[..., :N], 0, x.size(2)-1), tc.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = tc.cat([tc.clamp(q_rb[..., :N], 0, x.size(2)-1), tc.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = tc.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = tc.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = tc.cat([tc.clamp(p[..., :N], 0, x.size(2)-1), tc.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = tc.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
#        print("implement deformable_conv layer in network")
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = tc.meshgrid(
            tc.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            tc.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = tc.cat([tc.flatten(p_n_x), tc.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = tc.meshgrid(
            tc.arange(1, h*self.stride+1, self.stride),
            tc.arange(1, w*self.stride+1, self.stride))
        p_0_x = tc.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = tc.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = tc.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = tc.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        return x_offset


''' Another approach to implement deformable conv layer '''
class ConvOffset2D(nn.Conv2d):
    """
    ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation.
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1, bias = False, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = in_channels
        self.kernel_size = kernel_size
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, self.kernel_size, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))
        self.out_channel_maker = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        # Make the x_offset to have out_channels channels.
        x_offset = self.out_channel_maker(x_offset)

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return tc.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x

def deformable_conv(in_channels, out_channels, kernel_size, bias = True):
    # Aprroach one for deformable conv layer(running out of memory)
    """ return DeformConv2d(
        in_channels, out_channels, kernel_size = kernel_size,
        padding = (kernel_size//2), bias=bias) """
    # Aprroach two for deformable conv layer
    return ConvOffset2D(
        in_channels, out_channels, kernel_size = kernel_size,
        padding = (kernel_size//2), bias=bias)


"Sine Activation Function"
"""
2020.Implicit Neural Representations with Periodic Activation Functions. https://arxiv.org/abs/2006.09661
"""
class Sine(nn.Module):
    def __init__(self, w0 = 1.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        print("Sine activation Function is implement")
        return tc.sin(self.w0 * x)


"FReLU Activation Function"
"""
FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
2020.Funnel Activation for Visual Recognition. https://arxiv.org/abs/2007.11824
"""
class FReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x = tc.max(x, x1)
        print("FReLU is implement")
        return x


"Dynamic ReLU Activation Function"
"""
2020.Dynamic ReLU. https://arxiv.org/abs/2003.10027
"""
class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('lambdas', tc.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', tc.Tensor([1.] + [0.]*(2*k - 1)).float())
    def get_relu_coefs(self, x):
        theta = tc.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = tc.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta
    def forward(self, x):
        raise NotImplementedError

class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)
        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = tc.max(output, dim=-1)[0].transpose(0, -1)
        return result

class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)
    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)
        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v
        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = tc.max(output, dim=-1)[0].permute(1, 2, 0)
        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = tc.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result


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


"Calculate the fft to fetch k space result and ifft to go back to image domain"
class FFT_K_SPACE(nn.Module):
    def __init__(self):
        super(FFT_K_SPACE, self).__init__()

    def forward(self, x):
        # Take in image x at time domain and fetch the k space data at frequency domain. 
        # See https://pytorch.org/docs/stable/generated/torch.rfft.html#torch.rfft
        # Beware the shape of input for irfft in our case should be (N, C, H, W)
        k_space_result = tc.rfft(x, signal_ndim = 2, onesided = False)
        return k_space_result
""" class FFT_K_SPACE(nn.Module):
    def __init__(self):
        super(FFT_K_SPACE, self).__init__()
    def forward(self, x):
        x = tc.unsqueeze(x, -1) #----- create the additional last dimension for input matrix with (N, C, H, W)
        x_complex = tc.cat((x, tc.zeros_like(x)), -1)
        k_space_result = tc.fft(x_complex, 2)
        # print(k_space_result.size())
#        out = tc.sqrt(tc.mul(k_space_result[:, :, :, :, 0], k_space_result[:, :, :, :, 0]) + tc.mul(k_space_result[:, :, :, :, 1], k_space_result[:, :, :, :, 1]))
#        return out
        return k_space_result """

class IFFT_TIME_DOMAIN(nn.Module):
    def __init__(self):
        super(IFFT_TIME_DOMAIN, self).__init__()

    def forward(self, x):
        # Take in k space data x at frequency domain and fetch the image data at time domain. 
        # See https://pytorch.org/docs/stable/generated/torch.irfft.html
        # Beware the shape of input for irfft in our case should be (N, C, H, W, 2), the last 2 stands for real and image part of complex number
        time_domain_result = tc.irfft(x, signal_ndim = 2, onesided = False)
        return time_domain_result


"Warm Up Learning Rate"
class LearningRateWarmUP(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        cur_iteration += 1
        if cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_scheduler.step(cur_iteration-self.warmup_iteration)


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
    See figure 3 of original RCAN paper.
    Beware the CA Layer used in RCAN is actually same as the channel attention mechanism propsed in SENet(Squeeze-and-Excitation Networks).
    """
    def __init__(self, channel, reduction=16):
        """
        reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        """
        super(CALayer, self).__init__()
        # global average pooling(GAP): feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # global average pooling(GAP), output size is 1 for each channel
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


"Channel Attention Module(CAM), is another approach to calculate channel attention. See more in CBAM paper: 2018.CBAM: Convolutional Block Attention Module"
class ChannelAttention(nn.Module):
    """
    CAM is similar to CALayer block in RCAN, but also use max pooling rather than only average pooling for H x W field. 
    """
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


"Spatial Attention Module(SAM). See more in CBAM paper: 2018.CBAM: Convolutional Block Attention Module"
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = tc.mean(x, dim=1, keepdim=True)   # average along all channels
        max_out, _ = tc.max(x, dim=1, keepdim=True) # max along all channels
        x = tc.cat([avg_out, max_out], dim=1)       
        x = self.conv1(x)       # use conv with kernel size = 7 to "look around the pixels near every pixel"
        return self.sigmoid(x)


"End to End Channel and Spatial Attention Block, Either Sequential or Parallel for Channel and Spatial Attention"
class ChannelAndSpatialAttention(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7, channel_and_spatial_attention_mode = 'sequential_mode'):
        super(ChannelAndSpatialAttention, self).__init__()
        self.channel_attention_weight_generator = ChannelAttention(in_channel, reduction)
        self.spatial_attention_weight_generator = SpatialAttention(kernel_size)
        self.channel_and_spatial_attention_mode = channel_and_spatial_attention_mode
    
    def forward(self, x):
        if self.channel_and_spatial_attention_mode == 'sequential_mode':
            channel_attention_weight = self.channel_attention_weight_generator(x)
            x = channel_attention_weight*x
            spatial_attention_weight = self.spatial_attention_weight_generator(x)
            x = spatial_attention_weight*x
            return x
        elif self.channel_and_spatial_attention_mode == 'parallel_mode':
            channel_attention_weight = self.channel_attention_weight_generator(x)
            y = channel_attention_weight*x
            spatial_attention_weight = self.spatial_attention_weight_generator(x)
            z = spatial_attention_weight*x
            return y + z
        else:
            raise ValueError("Not supported channel and spatial attention mode yet")



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
            if i == 0: 
                if act == 'FReLU':
                    modules_body.append(FReLU(n_feat))
                elif act == 'Dynamic_ReLU_Type_A':
                    modules_body.append(DyReLUA(n_feat, conv_type='2d'))
                elif act == 'Dynamic_ReLU_Type_B':
                    modules_body.append(DyReLUB(n_feat, conv_type='2d'))
                else:
                    modules_body.append(act)
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
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=1) \
            for _ in range(n_rcablocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size)) # last conv after several RCABs as show in figure 2
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) # RCAB_1 --> RCAB_2 --> ... --> RCAB_(n_rcablocks) --> conv
        res += x # short skip connection in RG
        return res


"Dual Domain Gradient Map Residual Group"
class GradientMapDualResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_rcablocks):
        super(GradientMapDualResidualGroup, self).__init__()
        modules_body = [ResidualGroup(conv, n_feat, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks)]
        """ modules_body.append(conv(n_feat, n_feat, kernel_size)) """
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            modules_channel_attention_for_cross_branch_fusion = [CALayer(n_feat*2, reduction)]
        modules_connection = [(conv(2*n_feat, n_feat, kernel_size = 1, bias=True))]

        self.body = nn.Sequential(*modules_body)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            self.channel_attention_for_cross_branch_fusion = nn.Sequential(*modules_channel_attention_for_cross_branch_fusion)
        self.connection = nn.Sequential(*modules_connection)

    def forward(self, x):
        x1 = x[0,:,:,:,:] # Extract image branch data
        x2 = x[1,:,:,:,:] # Extract gradientmap branch data
        x1 = x1.squeeze(0)
        x2 = x2.squeeze(0)

        res1 = self.body(x1) # go through several RCABs in image branch
        res2 = self.body(x2) # go through several RCABs in gradientmap branch

        # fuse intermedian results in both branches into image branch
        res_image_branch = tc.cat((res1, res2), 1)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            res_image_branch = self.channel_attention_for_cross_branch_fusion(res_image_branch) # set up channel attention for both image and gradeint branch when theu fuse into one feature map
            print("channel attention for fusion:res_image_branch")
        res_image_branch = self.connection(res_image_branch)

        # fuse intermedian results in both branches into gradientmap branch
        res_gradientmap_branch = tc.cat((res1, res2), 1)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            res_gradientmap_branch = self.channel_attention_for_cross_branch_fusion(res_gradientmap_branch) # set up channel attention for both image and gradeint branch when theu fuse into one feature map
            print("channel attention for fusion:res_gradientmap_branch")
        res_gradientmap_branch = self.connection(res_gradientmap_branch)

        res_image_branch = res_image_branch.unsqueeze(0)
        res_gradientmap_branch = res_gradientmap_branch.unsqueeze(0)
        res = tc.cat((res_image_branch, res_gradientmap_branch), 0)
        return res


"Dual Domain K Space Residual Group"
class KSpaceDualResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_rcablocks):
        super(KSpaceDualResidualGroup, self).__init__()
        modules_body_time_domain = [ResidualGroup(conv, n_feat, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks)]
        modules_body_frequency_domain = [ResidualGroup(conv, 2*n_feat, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks)]
        """ modules_body.append(conv(n_feat, n_feat, kernel_size)) """
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            modules_channel_attention_for_cross_branch_fusion_in_image_branch = [CALayer(n_feat*2, reduction)]
            modules_channel_attention_for_cross_branch_fusion_in_k_space_branch = [CALayer(n_feat*4, reduction)]
        modules_connection_in_image_branch = [(conv(2*n_feat, n_feat, kernel_size = 1, bias=True))]
        modules_connection_in_k_space_branch = [(conv(4*n_feat, 2*n_feat, kernel_size = 1, bias=True))]

        ifft_operation = IFFT_TIME_DOMAIN()
        fft_operation = FFT_K_SPACE()

        self.modules_body_time_domain = nn.Sequential(*modules_body_time_domain)
        self.modules_body_frequency_domain = nn.Sequential(*modules_body_frequency_domain)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            self.channel_attention_for_cross_branch_fusion_in_image_branch = nn.Sequential(*modules_channel_attention_for_cross_branch_fusion_in_image_branch)
            self.channel_attention_for_cross_branch_fusion_in_k_space_branch = nn.Sequential(*modules_channel_attention_for_cross_branch_fusion_in_k_space_branch)
        self.connection_in_image_branch = nn.Sequential(*modules_connection_in_image_branch)
        self.connection_in_k_space_branch = nn.Sequential(*modules_connection_in_k_space_branch)
        self.ifft_operation = ifft_operation
        self.fft_operation = fft_operation

    def forward(self, x):
        # shape of x is (2, N, 2*C, H, W)
        length_of_x1 = x.shape[2]//2
        x1 = x[0,:,:,:,:] # Extract image branch data
        x2 = x[1,:,:,:,:] # Extract k space branch data
        x1 = x1.squeeze(0) # Now shape of x1 is (N, 2*C, H, W)
        x2 = x2.squeeze(0) # Now shape of x2 is (N, 2*C, H, W)

        res1 = self.modules_body_time_domain(x1[:, 0:length_of_x1, :, :]) # go through several RCABs in image branch, only half of channels have data. shape of res1 is (N, C, H, W)
        res2 = self.modules_body_frequency_domain(x2) # go through several RCABs in k space branch. shape of res2 is (N, 2*C, H, W)

        # fuse intermedian results in both branches into image branch
        num_of_channels_needed_k_space_branch = res2.shape[1]//2
        res2_in_format_fits_irfft = res2 # copy res2 to res2_in_format_fits_irfft, to save res2.
        res2_in_format_fits_irfft = res2_in_format_fits_irfft.unsqueeze(4) # shape of res2_in_format_fits_irfft is (N, 2*C, H, W, 1)
        # shape of res2_in_format_fits_irfft is (N, C, H, W, 2)
        res2_in_format_fits_irfft = tc.cat((res2_in_format_fits_irfft[:, 0:num_of_channels_needed_k_space_branch, :, :, :], res2_in_format_fits_irfft[:, num_of_channels_needed_k_space_branch:, :, :, :]), 4)
        res_image_branch = tc.cat((res1, self.ifft_operation(res2_in_format_fits_irfft)), 1) # shape of res_image_branch is (N, 2*C, H, W)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            res_image_branch = self.channel_attention_for_cross_branch_fusion_in_image_branch(res_image_branch) # set up channel attention for both image and gradeint branch when theu fuse into one feature map
            print("channel attention for fusion:res_image_branch")
        res_image_branch = self.connection_in_image_branch(res_image_branch) # shape of res_image_branch is (N, C, H, W)

        # fuse intermedian results in both branches into k space branch
        res1 = self.fft_operation(res1) # Now shape of res1 is (N, C, H, W, 2)
        res1 = tc.cat((res1[:, :, :, :, 0], res1[:, :, :, :, 1]), 1) # Now shape of res1 is (N, 2*C, H, W)
        res_k_space_branch = tc.cat((res1, res2), 1) # shape of res_k_space_branch is (N, 4*C, H, W)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            res_k_space_branch = self.channel_attention_for_cross_branch_fusion_in_k_space_branch(res_k_space_branch) # set up channel attention for both image and gradeint branch when theu fuse into one feature map
            print("channel attention for fusion:res_k_space_branch")
        res_k_space_branch = self.connection_in_k_space_branch(res_k_space_branch) # shape of res_k_space_branch is (N, 2*C, H, W)

        res_image_branch = res_image_branch.unsqueeze(0) # shape of res_image_branch is (1, N, C, H, W)
        res_image_branch = tc.cat((res_image_branch, res_image_branch), 2) # append data into shape (1, N, 2*C, H, W)
        res_k_space_branch = res_k_space_branch.unsqueeze(0) # shape of res_k_space_branch is (1, N, 2*C, H, W)
        res = tc.cat((res_image_branch, res_k_space_branch), 0) # shape of res is (2, N, 2*C, H, W)
        return res


"Dual Domain Wavelets Transform Residual Group"
class WaveletsTransformDualResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_rcablocks):
        super(WaveletsTransformDualResidualGroup, self).__init__()
        modules_body_time_domain = [ResidualGroup(conv, n_feat, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks)]
        modules_body_wavelets_high_frequency_components_domain = [ResidualGroup(conv, 3*n_feat, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks)]
        """ modules_body.append(conv(n_feat, n_feat, kernel_size)) """
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            modules_channel_attention_for_cross_branch_fusion_in_wavelets_high_frequency_components_branch = [CALayer(n_feat*6, reduction)]
        modules_connection_in_wavelets_high_frequency_components_branch = [(conv(6*n_feat, 3*n_feat, kernel_size = 1, bias=True))]

        self.modules_body_time_domain = nn.Sequential(*modules_body_time_domain)
        self.modules_body_wavelets_high_frequency_components_domain = nn.Sequential(*modules_body_wavelets_high_frequency_components_domain)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            self.channel_attention_for_cross_branch_fusion_in_wavelets_high_frequency_components_branch = nn.Sequential(*modules_channel_attention_for_cross_branch_fusion_in_wavelets_high_frequency_components_branch)
        self.connection_in_wavelets_high_frequency_components_branch = nn.Sequential(*modules_connection_in_wavelets_high_frequency_components_branch)

    def forward(self, x):
        # shape of x is (N, C, 4, H', W'), x contains one low frequency information(approximation coefficients) component whose shape
        # is (N, C, 1, H', W') and three high frequency information(horizontal detail coefficients, vertical detail coefficients, 
        # diagonal detail coefficients) components whose shape is (N, C, 3, H', W').
        x_low_freq = x[:,:,0,:,:] # low frequency information component. shape is (N, C, 1, H', W')
        x_high_freq = x[:,:,1:,:,:] # high frequency information components. shape is (N, C, 3, H', W')

        x_low_freq = x_low_freq.squeeze(2) # Now shape of x_low_freq is (N, C, H', W')
        x_img = calculate_inverse_wavelet_transform(x_low_freq, x_high_freq) # Extract the image branch data. Now shape of x_img is (N, C, H, W)
        x_high_freq = tc.cat((x_high_freq[:,:,0,:,:], x_high_freq[:,:,1,:,:], x_high_freq[:,:,2,:,:]), 1) # Extract the wavelets high frequency components branch data. Now shape of x_high_freq is (N, 3*C, H', W')

        x_img = self.modules_body_time_domain(x_img) # go through several RCABs in image branch. Now shape of x_img is (N, C, H, W)
        x_high_freq = self.modules_body_wavelets_high_frequency_components_domain(x_high_freq) # go through several RCABs in wavelets high frequency components branch. Now shape of x_high_freq is (N, 3*C, H', W')

        # fuse intermedian results in both branches
        y_low_frequency, y_high_frequency = calculate_wavelet_transform(x_img) # shape of y_low_frequency is (N, C, H', W'), shape of y_high_frequency (N, C, 3, H', W')
        y_high_frequency = tc.cat((y_high_frequency[:,:,0,:,:], y_high_frequency[:,:,1,:,:], y_high_frequency[:,:,2,:,:]), 1) # Now shape of y_high_frequency (N, 3*C, H', W')
        x_high_freq = tc.cat((x_high_freq, y_high_frequency), 1) # fuse x_high_freq from wavelets high frequency components branch with y_high_frequency from image branch. Now shape of x_high_freq is (N, 6*C, H', W')
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            x_high_freq = self.channel_attention_for_cross_branch_fusion_in_wavelets_high_frequency_components_branch(x_high_freq) # set up channel attention for both wavelets high frequency components branch and image branch when they fuse into one feature map
            print("channel attention for fusion:wavelets_high_frequency_components_branch")
        x_high_freq = self.connection_in_wavelets_high_frequency_components_branch(x_high_freq) # Now shape of x_high_freq is (N, 3*C, H', W')

        x_high_freq = x_high_freq.unsqueeze(2) # Now shape of x_high_freq is (N, 3*C, 1, H', W')
        num_of_features = x_high_freq.shape[1]//3
        x_high_freq = tc.cat((x_high_freq[:, 0:num_of_features, :, :, :], x_high_freq[:, num_of_features:2*num_of_features, :, :, :], 
            x_high_freq[:, 2*num_of_features:3*num_of_features, :, :, :]), 2) # Now shape of x_high_freq is (N, C, 3, H', W')
        y_low_frequency = y_low_frequency.unsqueeze(2) # Now shape of y_low_frequency is (N, C, 1, H', W')
        fused_data = tc.cat((y_low_frequency, x_high_freq), 2) # Now shape of fused_data is (N, C, 4, H', W')
        return fused_data


# TODO: Even consider using upsmaler module which is commonly used for high resolution task together with channel attenion to assign different 
# weights for different channel for upscaling. We could see similar idea in paper: 2020.Detecting Small Objects Using a Channel-Aware Deconvolutional Network.
"Upsampler Module, implemented by employeed of sub-pixel conv"
class Upsampler(nn.Sequential):
    """
    Upsampling/Upscale module, used as last part of "SR reconstruction network model" if the network model employ the "post-upsampling mode".
    Beware the actual upsampling approach is "sub-pixel conv" (which is nn.PixelShuffle() in Pytorch) which was proposed in
    paper: "2016. Real-Time single image and video super-resolution using an efficient sub-pixel convolutional neural network".
    Such sub-pixel conv actually constructs F ∗ S^2 feature maps of dimensions H ×W are reshaped into F feature maps of dimensions H ∗ S × W ∗ S, 
    where S is the upsampling factor.
    """
    def __init__(self, conv, scale, n_feats, use_channel_and_spatial_attention_inside_upsampler = False, channel_and_spatial_attention_mode = 'sequential_mode'):
        super(Upsampler, self).__init__()
        if scale == 2:
            if use_channel_and_spatial_attention_inside_upsampler == False:
                self.upsampler = nn.Sequential(*[
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                    nn.PixelShuffle(scale)
                ])
            else: # use_channel_and_spatial_attention_inside_upsampler == True
                self.upsampler = nn.Sequential(*[
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                    ChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                    nn.PixelShuffle(scale)
                ])
        elif scale == 4:
            if use_channel_and_spatial_attention_inside_upsampler == False:
                self.upsampler = nn.Sequential(*[
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                    nn.PixelShuffle(2),
                ])
            else: # use_channel_and_spatial_attention_inside_upsampler == True
                self.upsampler = nn.Sequential(*[
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                    ChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                    nn.PixelShuffle(2),
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                    ChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                    nn.PixelShuffle(2),
                ])
        else:
            raise ValueError("scale must be 2 or 4.")

    def forward(self, x):
        upsampled_x = self.upsampler(x)
        return upsampled_x


"Residual Channel Attention Network (RCAN) based Dual Domain Network for Super Resolution MRI"
class RCAN_Based_MRI_SR_Dual_Domain_2D(nn.Module):
    """
    RCAN(Deep Residual Channel Attention Network) = RIR(Residual in Residual module) + Upsampler Module.
    See bottom figure in figure 2 of original RCAN paper
    """
    def __init__(self, args):
        super(RCAN_Based_MRI_SR_Dual_Domain_2D, self).__init__()
        if args['type_of_network'] == 'RCAN':
            self.type_of_network = 'RCAN'
        elif args['type_of_network'] == 'gradient_map_dual_domain':
            self.type_of_network = 'gradient_map_dual_domain'
        elif args['type_of_network'] == 'k_space_dual_domain':
            self.type_of_network = 'k_space_dual_domain'
        elif args['type_of_network'] == 'wavelets_transform_dual_domain':
            self.type_of_network = 'wavelets_transform_dual_domain'

        if args['conv_layer_type'] == 'default_conv':
            conv = default_conv
        elif args['conv_layer_type'] == 'coord_conv':
            conv = coord_conv
        elif args['conv_layer_type'] == 'deformable_conv':
            conv = deformable_conv
        elif args['conv_layer_type'] == 'py_conv':
            conv = py_conv

        if args['activation_function_type'] == 'ReLU':
            act = nn.ReLU(True)
        elif args['activation_function_type'] == 'Sine':
            act = Sine(w0 = 1.0)
        elif args['activation_function_type'] == 'FReLU':
            act = 'FReLU'
        elif args['activation_function_type'] == 'Dynamic_ReLU_Type_A':
            act = 'Dynamic_ReLU_Type_A'
        elif args['activation_function_type'] == 'Dynamic_ReLU_Type_B':
            act = 'Dynamic_ReLU_Type_B'

        n_resgroups = args['n_resgroups'] # number of RGs in RIR/RCAN
        n_rcablocks = args['n_rcablocks'] # number of RCABs in one RG
        n_colors = 1 # number of channels going of input of entire model
        n_feats = args['n_feats'] # number of feature maps/channels going through the entire model
        kernel_size = 3 # conv filter size used for all conv in RCAN
        reduction = args['reduction'] # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        scale = args['scale'] # resize factor, e.g. 2, 4
        use_channel_and_spatial_attention_inside_upsampler = args['use_channel_and_spatial_attention_inside_upsampler'] # whether we use channel and spatial attention block inside upsampler, e.g. True, False
        channel_and_spatial_attention_mode = args['channel_and_spatial_attention_mode'] # which end to end channel and spatial block to use, e.g. 'sequential_mode', 'parallel_mode'

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        # define commonly use head module
        modules_head = [conv(n_colors, n_feats, kernel_size)] # the first conv layer in RCAN, show in figure 2 of RCAN paper

        # define commonly use pretail module
        modules_pretail = [conv(n_feats, n_feats, kernel_size)]

        # define commonly use(for dual domain network) end_stage_fusion_of_outcome module
        modules_end_stage_fusion_of_outcome = [conv(2*n_colors, n_colors, kernel_size)]

        # define body module for different type of network. 
        if self.type_of_network == 'RCAN':
            # The RIR(Residual in Residual) which consists of n_resgroups RGs, show in figure 2 of RCAN paper
            modules_body = [
                ResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks) \
                for _ in range(n_resgroups)]
        elif self.type_of_network == 'gradient_map_dual_domain':
            modules_body = [
                GradientMapDualResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks) \
                for _ in range(n_resgroups)]
        elif self.type_of_network == 'k_space_dual_domain':
            modules_body = [
                KSpaceDualResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks) \
                for _ in range(n_resgroups)]
            modules_head_for_k_space_branch = [conv(2*n_colors, 2*n_feats, kernel_size)]
            modules_pretail_for_k_space_branch = [conv(2*n_feats, 2*n_feats, kernel_size)]
            modules_tail_for_k_space_branch = [
                Upsampler(conv, scale, 2*n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                            channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                conv(2*n_feats, 2*n_colors, kernel_size)]
        elif self.type_of_network == 'wavelets_transform_dual_domain':
            modules_body = [
                WaveletsTransformDualResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks) \
                for _ in range(n_resgroups)]



        # define tail module. The last stage is upsampling module and one more conv layer, show in figure 2 of RCAN paper
        modules_tail = [
            Upsampler(conv, scale, n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                            channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
            conv(n_feats, n_colors, kernel_size)]

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
        self.pretail = nn.Sequential(*modules_pretail)
        self.tail = nn.Sequential(*modules_tail)
        if self.type_of_network == 'k_space_dual_domain':
            self.head_for_k_space_branch = nn.Sequential(*modules_head_for_k_space_branch)
            self.pretail_for_k_space_branch = nn.Sequential(*modules_pretail_for_k_space_branch)
            self.tail_for_k_space_branch = nn.Sequential(*modules_tail_for_k_space_branch)
            self.modules_end_stage_fusion_of_outcome = nn.Sequential(*modules_end_stage_fusion_of_outcome)
        if self.type_of_network == 'gradient_map_dual_domain':
            self.modules_end_stage_fusion_of_outcome = nn.Sequential(*modules_end_stage_fusion_of_outcome)

    def forward(self, x):
        # do NOT understand why need this, may NOT be useful for us 
        """ x = self.sub_mean(x) """
        if self.type_of_network == 'RCAN':
            x = self.head(x) # input image goes through first conv layer
            res = self.body(x) # data goes through sveral ResidualGroups
            res = self.pretail(res)
            res += x # long skip connection of RIR
            if (Maintain_Same_Size == True):
                res = self.down_size_converter(res) # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output
            x = self.tail(res) # data goes through upsampling module and one more conv layer
            # do NOT understand why need this, may NOT be useful for us
            """ x = self.add_mean(x) """
            return x, None, 'Single Branch Network'

        elif self.type_of_network == 'gradient_map_dual_domain':
            if plot_the_gradient_map_of_input_image == True:
                plt.subplot(1, 2, 1)
                plt.imshow(x.cpu()[0, 0, :, :], cmap='gray')
                plt.title("LR_image")
                plt.subplot(1, 2, 2)
                plt.title("gradient_map_of_LR_image")
                plt.imshow(calculate_gradient_map(x).cpu()[0, 0, :, :], cmap='gray')
                plt.suptitle("The example pair of LR and gradient map of LR image")
                plt.subplots_adjust()
                plt.show()
            x1 = self.head(x) # 1st image branch
            x2 = self.head(calculate_gradient_map(x)) # 2nd gradient branch
            x = tc.cat((x1.unsqueeze(0), x2.unsqueeze(0)), 0)
            res = self.body(x) # data goes through sveral DualResidualGroups

            image_branch_res = res[0,:,:,:,:] # Fetch image branch 
            image_branch_res = image_branch_res.squeeze(0)
            image_branch_res = self.pretail(image_branch_res)
            image_branch_res += x1 # long skip connection at image branch

            gradientmap_branch_res = res[1,:,:,:,:] # Fetch gradientmap branch 
            gradientmap_branch_res = gradientmap_branch_res.squeeze(0)
            gradientmap_branch_res = self.pretail(gradientmap_branch_res)
            gradientmap_branch_res += x2 # long skip connection at gradientmap branch
            if (Maintain_Same_Size == True):
                # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output
                image_branch_res = self.down_size_converter(image_branch_res)
                gradientmap_branch_res = self.down_size_converter(gradientmap_branch_res)
            image_branch_y = self.tail(image_branch_res) # data goes through upsampling module and one more conv layer
            gradientmap_branch_y = self.tail(gradientmap_branch_res) # data goes through upsampling module and one more conv layer
            # do NOT understand why need this, may NOT be useful for us
            """ y = self.add_mean(y) """

            image_branch_fused_y = tc.cat((image_branch_y, gradientmap_branch_y), 1)
            image_branch_fused_y = self.modules_end_stage_fusion_of_outcome(image_branch_fused_y)

            # (N, 1, H, W), (N, 1, H, W)
            return image_branch_fused_y, gradientmap_branch_y, 'Secondary branch is gradient map branch'

        elif self.type_of_network == 'k_space_dual_domain':
            if plot_the_k_space_data_of_input_image == True:
                plt.subplot(1, 3, 1)
                plt.imshow(x.cpu()[0, 0, :, :], cmap='gray') # vmin/vmax stand for windowing size
                plt.title("LR_image")
                plt.subplot(1, 3, 2)
                plt.title("real_part_k_space_data_of_LR_image")
                plt.imshow(tc.rfft(x, signal_ndim = 2, onesided = False).cpu()[0, 0, :, :, 0], cmap='gray', vmin=0, vmax=32) # vmin/vmax stand for windowing size
                plt.subplot(1, 3, 3)
                plt.title("image_part_k_space_data_of_LR_image")
                plt.imshow(tc.rfft(x, signal_ndim = 2, onesided = False).cpu()[0, 0, :, :, 1], cmap='gray', vmin=0, vmax=32) # vmin/vmax stand for windowing size
                plt.suptitle("The example pair of LR and k_space of LR image")
                plt.subplots_adjust()
                plt.show()
            x1 = self.head(x) # 1st image branch. shape from (N, 1, H, W) --> (N, C, H, W)

            x2 = tc.rfft(x, signal_ndim = 2, onesided = False) # 2nd k space branch, shape is (N, 1, H, W, 2)
            x2 = tc.cat((x2[:, :, :, :, 0], x2[:, :, :, :, 1]), 1) # change shape of x2 as (N, 2, H, W)
            x2 = self.head_for_k_space_branch(x2) # 2nd k space branch. shape from (N, 2, H, W) --> (N, 2*C, H, W)

            x1_new = x1
            x2_new = x2
            x1_new = x1_new.unsqueeze(0) # make shape of x1 is (1, N, C, H, W)
            x1_new = tc.cat((x1_new, x1_new), 2) # append data into shape (1, N, 2*C, H, W)
            x2_new = x2_new.unsqueeze(0) # make shape of x2 is (1, N, 2*C, H, W)
            x = tc.cat((x1_new, x2_new), 0) # make input into body, shape of x is (2, N, 2*C, H, W)

            res = self.body(x) # data goes through sveral KSpaceDualResidualGroup. shape of res is (2, N, 2*C, H, W)

            num_of_channels_needed = res.shape[2]//2
            image_branch_res = res[0, :, 0:num_of_channels_needed, :, :] # Fetch image branch 
            image_branch_res = image_branch_res.squeeze(0) # shape of image_branch_res is (N, C, H, W)
            image_branch_res = self.pretail(image_branch_res) # shape of image_branch_res is (N, C, H, W)
            image_branch_res += x1 # long skip connection at image branch

            k_space_branch_res = res[1,:,:,:,:] # Fetch k space branch 
            k_space_branch_res = k_space_branch_res.squeeze(0) # shape of k_space_branch_res is (N, 2*C, H, W)
            k_space_branch_res = self.pretail_for_k_space_branch(k_space_branch_res) # shape of k_space_branch_res is (N, 2*C, H, W)
            k_space_branch_res += x2 # long skip connection at gradientmap branch
            if (Maintain_Same_Size == True):
                # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output
                image_branch_res = self.down_size_converter(image_branch_res)
                gradientmap_branch_res = self.down_size_converter(gradientmap_branch_res)
            image_branch_y = self.tail(image_branch_res) # data goes through upsampling module and one more conv layer. # shape of image_branch_y is (N, 1, H, W)
            k_space_branch_y = self.tail_for_k_space_branch(k_space_branch_res) # data goes through upsampling module and one more conv layer. # shape of k_space_branch_y is (N, 2, H, W)
            # do NOT understand why need this, may NOT be useful for us
            """ y = self.add_mean(y) """

            k_space_branch_y = k_space_branch_y.unsqueeze(4) # (N, 2, H, W, 1)
            k_space_branch_y = tc.cat((k_space_branch_y[:, 0, :, :, :], k_space_branch_y[:, 1, :, :, :]), 3) # (N, H, W, 2)
            k_space_branch_y = k_space_branch_y.unsqueeze(1) # (N, 1, H, W, 2)
            image_branch_fused_y = tc.cat((image_branch_y, tc.irfft(k_space_branch_y, signal_ndim = 2, onesided = False)), 1) # shape image_branch_fused_y is (N, 2, H, W)
            image_branch_fused_y = self.modules_end_stage_fusion_of_outcome(image_branch_fused_y) # shape image_branch_fused_y is (N, 1, H, W)

            # (N, 1, H, W), (N, 1, H, W, 2)
            return image_branch_fused_y, k_space_branch_y, 'Secondary branch is k space branch'

        elif self.type_of_network == 'wavelets_transform_dual_domain':
            # y_low_frequency is the low frequency component(approximation coeefficient)
            # y_high_frequency is the 1st level high frequency components(include 1st level horizontal detail coefficients, vertical detail coefficients, diagonal detail coefficients)
            y_low_frequency, y_high_frequency = calculate_wavelet_transform(x) # shape of y_low_frequency: (N, C, H′, W′) and shape of y_high_frequency: (N, C, 3, H′, W′)
            if plot_the_wavelets_transform_data_of_input_image == True:
                plt.subplot(2, 3, 1)
                plt.imshow(x.cpu()[0, 0, :, :], cmap='gray') # vmin/vmax stand for windowing size
                plt.title("LR_image")
                plt.subplot(2, 3, 2)
                plt.imshow(y_low_frequency[0, 0, :, :].cpu(), cmap='gray') # Low frequency information(Approximation coefficients) of LR
                plt.title("wavelet_Low_Frequent_Data_LR_image")
                plt.subplot(2, 3, 3)
                plt.imshow(y_high_frequency[0, 0, 0, :, :].cpu(), cmap='gray') # Horizontal detail coefficients of LR
                plt.title("wavelet_Horizontal_Detail_Data_LR_image")
                plt.subplot(2, 3, 4)
                plt.imshow(y_high_frequency[0, 0, 1, :, :].cpu(), cmap='gray') # Vertical detail coefficients of LR
                plt.title("wavelet_Vertical_Detail_Data_LR_image")
                plt.subplot(2, 3, 5)
                plt.imshow(y_high_frequency[0, 0, 2, :, :].cpu(), cmap='gray') # Diagonal detail coefficients of LR
                plt.title("wavelet_Diagonal_Detail_Data_LR_image")
                plt.suptitle("The The example pair of LR and Wavelet Transform Data LR")
                plt.subplots_adjust()
                plt.show()
            x1 = self.head(x) # 1st image branch. shape from (N, 1, H, W) --> (N, C, H, W)
            y_low_frequency, y_high_frequency = calculate_wavelet_transform(x1) # shape of y_low_frequency: (N, C, H′, W′) and shape of y_high_frequency: list(N, C, 3, H′, W′)
            y_low_frequency = y_low_frequency.unsqueeze(2)  # shape from (N, C, H, W) --> (N, C, 1, H, W)
            x2 = tc.cat((y_low_frequency, y_high_frequency), 2) # Now shape of x1 is (N, C, 4, H', W')

            res = self.body(x2) # data goes through sveral WaveletsTransformDualResidualGroup. shape of res is (N, C, 4, H', W')

            y_low_frequency_component = res[:, :, 0, :, :]
            y_high_frequency_component = res[:, :, 1:, :, :]
            data_almost_done = calculate_inverse_wavelet_transform(y_low_frequency_component, y_high_frequency_component) # shape of data_almost_done is (N, C, H, W)
            image_branch_fused_y = self.pretail(data_almost_done) # shape of data_almost_done is (N, C, H, W)
            image_branch_fused_y += x1 # long skip connection for fused data

            if (Maintain_Same_Size == True):
                # extra down_size_converter is needed to shrik size of image scale times if we expect same size as input LR for SR output
                image_branch_fused_y = self.down_size_converter(image_branch_fused_y)
            image_branch_fused_y = self.tail(image_branch_fused_y) # data goes through upsampling module and one more conv layer. # shape of image_branch_fused_y is (N, 1, H, W)
            # do NOT understand why need this, may NOT be useful for us
            """ y = self.add_mean(y) """
            wavelets_branch_fused_y_low_frequency, wavelets_branch_fused_y_high_frequency = calculate_wavelet_transform(image_branch_fused_y) # shape of wavelets_branch_fused_y_high_frequency is (N, 1, 3, H', W')
            """ wavelets_branch_fused_y_high_frequency = wavelets_branch_fused_y_high_frequency.permute(0, 1, 3, 4, 2) # shape from (N, 1, 3, H', W') --> (N, 1, H', W', 3) """
            # (N, 1, H, W), (N, 1, 3, H', W')
            return image_branch_fused_y, wavelets_branch_fused_y_high_frequency, 'Secondary branch is wavelets high frequency components branch'


"Wrapper for Progressive Learning Super Resolution MRI Reconstruction for multiple size, e.g. 2x, 4x, 8x, etc."
class Progressive_Learning_Wrapper_MRI_SR_Dual_Domain_2D(nn.Module):
    def __init__(self, args):
        super(Progressive_Learning_Wrapper_MRI_SR_Dual_Domain_2D, self).__init__()
        self.number_of_progressive_stage = args['number_of_progressive_stage']
        self.stage = RCAN_Based_MRI_SR_Dual_Domain_2D(args)
    
    def forward(self, x):
        if self.number_of_progressive_stage == 1:
            return self.stage(x)
        elif self.number_of_progressive_stage == 2:
            x, _, _ = self.stage(x)
            return self.stage(x)
        elif self.number_of_progressive_stage == 3:
            x, _, _ = self.stage(x)
            x, _, _ = self.stage(x)
            return self.stage(x)



device=tc.device("cuda" if use_cuda else "cpu")
our_rcan_mri_sr_2d = Progressive_Learning_Wrapper_MRI_SR_Dual_Domain_2D(args)

# Weight initialization using He initialization.
""" for m in our_rcan_mri_sr_2d.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in') """

if tc.cuda.device_count()>1:
    our_rcan_mri_sr_2d=nn.DataParallel(our_rcan_mri_sr_2d)
our_rcan_mri_sr_2d.to(device)

print('this is our RCAN_MRI_SR_2D_Dual_Domain: ', our_rcan_mri_sr_2d)

feature_extractor = FeatureExtractor().to(device)
print('this is our FeatureExtractor: ', feature_extractor)

fft_k_space = FFT_K_SPACE().to(device)
print('this is our FFT_K_SPACE: ', fft_k_space)


"""""""""""""""""""""""""""""""""""""""
4. Setup optimization algorithm part
"""""""""""""""""""""""""""""""""""""""
"set an optimizer"
if args['optimizer'] == 'look_ahead':
    base_opt = opt.Adam(our_rcan_mri_sr_2d.parameters(), lr=1e-3, betas=(0.9, 0.999)) #----- use Adam algorithm as based optimizer A
    optimizer = lookahead.Lookahead(base_opt, k=5, alpha=0.5) # Initialize Lookahead
elif args['optimizer'] == 'Adam':
    optimizer = opt.Adam(our_rcan_mri_sr_2d.parameters(), lr = 0.0001, eps = 1e-08, weight_decay = 1e-5)    #----- use Adam algorithm for all parameters of our_classifier
elif args['optimizer'] == 'SGD_with_momentum':
    optimizer = opt.SGD(our_rcan_mri_sr_2d.parameters(), lr = 0.0001, momentum=0.9, weight_decay = 1e-9)    #----- use SGD algorithm for all parameters of our_lenet, by learning rate 0.01 and Momentum is 0.9
"set scheduler"
if args['learning_rate_decay_method'] == 'multi_step_learning_rate':
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
elif args['learning_rate_decay_method'] == 'step_learning_rate':
    scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
elif args['learning_rate_decay_method'] == 'exponential_learning_rate':
    scheduler = opt.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
elif args['learning_rate_decay_method'] == 'cosine_learning_rate_decay':
    # See https://blog.zhujian.life/posts/6eb7f24f.html for more info 
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCH_NUM, eta_min = 1e-8, last_epoch = -1) # 该函数实现了一个周期的余弦退火，可用于平缓的下降学习率
elif args['learning_rate_decay_method'] == 'cosine_learning_rate_warm_restarts':
    scheduler = opt.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult = 2, eta_min = 1e-8, last_epoch = -1)

if args['use_learning_rate_warm_up'] == True:
    scheduler = LearningRateWarmUP(optimizer = optimizer,
                                   warmup_iteration = args['how_many_epoch_to_be_used_for_warm_up'],  # Warm up stops at which epoch
                                   target_lr = args['initial_learning_rate_after_warm_up'],
                                   after_scheduler = scheduler)


"set loss related item"
loss_function_MSE = nn.MSELoss().to(device)        #----- MSE loss

loss_function_L1 = nn.SmoothL1Loss().to(device)       #----- smooth L1 loss

# loss_function_CE = nn.CrossEntropyLoss().to(device)

SSIM_function = pytorch_ssim_l1.SSIM(luminance_weight = args_loss_weight['ssim_luminance_weight'], contrast_weight = args_loss_weight['ssim_contrast_weight'], structure_weight = args_loss_weight['ssim_structure_weight']).to(device)       #----- ssim calculation

SSIM_map = pytorch_ssim_map.SSIMMap(luminance_weight = args_loss_weight['ssim_luminance_weight'], contrast_weight = args_loss_weight['ssim_contrast_weight'], structure_weight = args_loss_weight['ssim_structure_weight']).to(device)       #----- ssim calculation

# =============================================================================
# print('The loss function is L1Loss')
# loss_function = nn.L1Loss(size_average = False).to(device) 
# =============================================================================
# =============================================================================
# print('The loss function is CrossEntropyLoss')
# loss_function = nn.CrossEntropyLoss().to(device)        #----- here use cross entropy loss
# =============================================================================


"""""""""""""""""""""""""""
5. Train the RCAN_Based_MRI_SR_Dual_Domain_2D part
"""""""""""""""""""""""""""
"Train the RCAN_Based_MRI_SR_Dual_Domain_2D"
tc.set_num_threads(10)  #----- Sets the number of OpenMP threads used for parallelizing CPU operations

best_ssim = 0.0 # Initialization of best_ssim = 0.0

for epoch in range(EPOCH_NUM):
    "Set training mode"
    our_rcan_mri_sr_2d.train()
# =============================================================================
#     print('This is the ', epoch, ' epoch')
# =============================================================================
    running_loss = 0.0

    batch_number_test = 0
    feature_map_loss_test = 0.0
    pixel_wise_loss_test = 0.0
    k_space_freq_loss_test = 0.0
    ssim_loss_test = 0.0
    gradient_img_loss_test = 0.0
    ssim_test = 0.0

    batch_number_training = 0
    feature_map_loss_training = 0.0
    pixel_wise_loss_training = 0.0
    k_space_freq_loss_training = 0.0
    ssim_loss_training = 0.0
    gradient_img_loss_training = 0.0
    loss_training = 0.0
    ssim_training = 0.0

    if Use_Gram_Matrix_L1_Loss == True:
        gram_similarity_between_img_loss_training = 0.0
        gram_similarity_between_img_loss_test = 0.0

    k_space_branch_k_space_loss_training = 0.0
    k_space_branch_k_space_loss_test = 0.0
    gradient_grad_loss_training = 0.0
    gradient_grad_loss_test = 0.0
    wavelets_high_frequency_components_branch_high_frequency_loss_training = 0.0
    wavelets_high_frequency_components_branch_high_frequency_loss_test = 0.0

    # If using warm up, call the scheduler of warm up now
    if args['use_learning_rate_warm_up'] == True:
        scheduler.step(epoch)
        learning_rate = optimizer.param_groups[0]['lr']
        print('learning rate for epoch %d is : %f' % (epoch, optimizer.param_groups[0]['lr']))

    for i, data in enumerate(trainloader, 0):
# =============================================================================
#         print('This is the ', i, ' batch for the ', epoch, ' epoch' )
# =============================================================================
        # We call tc.cuda.empty_cache() if we use deformable_conv, due that deformable_conv will use huge amount of memory so we need to
        # call tc.cuda.empty_cache() trying to empty the unused GPU cache(although it may be useless also and it is still out of GPU memory
        # when applying deformable_conv).

        """
        每一次调用loss.backward()函数之前都要用optimizer.zero_grad()将梯度清零。因为如果梯度不清零，pytorch中会将上次计算的梯度和本次计算
        的梯度累加。
        PyTorch这种自动累加之前计算的梯度和本次计算梯度的机制逻辑的好处是，当我们的硬件限制不能使用更大的bachsize时，使用多次计算较小的
        bachsize的梯度平均值来代替，更方便，坏处当然是正常计算时我们只需要本次计算的梯度于是每次都要清零梯度。
        """
        "clear all stored gradients if there exist"
        optimizer.zero_grad()
        # print('The optimizer has been cleared' )


        if args['conv_layer_type'] == 'deformable_conv':
            tc.cuda.empty_cache()

        "load input data"
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        # print('The data have been loaded' )

        "forward prop"
        # outputs = our_resnext(inputs).double() #-- numpy arrays are 64-bit floating point and will be converted to torch.DoubleTensor standardly. Now, if you use them with your model, you'll need to make sure that your model parameters are also Double
        img_outputs, secondary_branch_outputs, network_model_type = our_rcan_mri_sr_2d(inputs) #-- or using default float as type, however remember to cast the input from Double to Float            
        # print(outputs.size())
        # print('the forward pass has been went')

        SR_img_copies = tc.cat((img_outputs, img_outputs, img_outputs), 1)
        # print(SR_copies.size())
        SR_features = feature_extractor(SR_img_copies)
        # print(SR_features.size())
        # print(SR_features.dtype)

        HR_copies = tc.cat((labels, labels, labels), 1)
        # print(HR_copies.size())
        HR_features = feature_extractor(HR_copies)
        # print(HR_features.size())
        # print(HR_features.dtype)

        SR_freq = fft_k_space(img_outputs)

        HR_freq = fft_k_space(labels)

        "calculate the gradients for all Variables during back prop"
        "vgg loss + pixel MSE loss + fft frequency loss, and we use weight_decay in Adam so that is L2 regularization"
        feature_map_loss = args_loss_weight['feature_map_weight']*loss_function_MSE(SR_features, HR_features)
        feature_map_loss_training += feature_map_loss.item()    # Only save the value of feature_map_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage
        # feature_map_loss = 0.000000001*loss_function_CE(SR_features, HR_features)
#           print("feature_map_loss: ", feature_map_loss)

#       pixel_wise_loss = 10*loss_function_MSE(outputs, labels)
        pixel_wise_loss = args_loss_weight['pixel_wise_weight']*loss_function_L1(img_outputs, labels)
        pixel_wise_loss_training += pixel_wise_loss.item()    # Only save the value of pixel_wise_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage 
#            print("pixel_wise_loss: ", pixel_wise_loss)

        if Amplify_High_Frequency_Value_In_K_Space_Loss == True:
            k_space_freq_loss = args_loss_weight['k_space_weight']*(loss_function_MSE(
                create_2d_Gaussian_weights(window_size = SR_freq.shape[2], num_of_samples = SR_freq.shape[0], channel = SR_freq.shape[1]).to(device)*SR_freq[:,:,:,:,0], 
                create_2d_Gaussian_weights(window_size = HR_freq.shape[2], num_of_samples = HR_freq.shape[0], channel = HR_freq.shape[1]).to(device)*HR_freq[:,:,:,:,0]) + 
                loss_function_MSE(
                    create_2d_Gaussian_weights(window_size = SR_freq.shape[2], num_of_samples = SR_freq.shape[0], channel = SR_freq.shape[1]).to(device)*SR_freq[:,:,:,:,1], 
                    create_2d_Gaussian_weights(window_size = HR_freq.shape[2], num_of_samples = HR_freq.shape[0], channel = HR_freq.shape[1]).to(device)*HR_freq[:,:,:,:,1]))
        else:
            k_space_freq_loss = args_loss_weight['k_space_weight']*(loss_function_MSE(SR_freq[:,:,:,:,0], HR_freq[:,:,:,:,0]) + loss_function_MSE(SR_freq[:,:,:,:,1], HR_freq[:,:,:,:,1]))
        k_space_freq_loss_training += k_space_freq_loss.item()
#            print(loss_function_MSE(SR_freq[:,:,:,:,0], HR_freq[:,:,:,:,0]))
#            print(loss_function_MSE(SR_freq[:,:,:,:,1], HR_freq[:,:,:,:,1]))
        """ print("k_space_freq_loss: ", k_space_freq_loss) """

        HR_ssim_weighted, HR_ssim = SSIM_function(labels, labels)
        SR_ssim_weighted, SR_ssim = SSIM_function(img_outputs, labels)
        HR_ssim_map_weighted, HR_ssim_map = SSIM_map(labels, labels)
        SR_ssim_map_weighted, SR_ssim_map = SSIM_map(img_outputs, labels)

        ssim_loss = args_loss_weight['ssim_weight']*loss_function_L1(SR_ssim_map_weighted, HR_ssim_map_weighted)
        ssim_loss_training += ssim_loss.item()    # Only save the value of ssim_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage 
        ssim_training += SR_ssim.item()    # Accumulation of SR_ssim in training over all batches in one epoch, will be used to calculate the average value of SR SSIM for one epoch.
        """ print("ssim_loss: ", ssim_loss) """

        gradient_img_loss = args_loss_weight['gradient_img_weight']*loss_function_L1(calculate_gradient_map(img_outputs), calculate_gradient_map(labels))
        gradient_img_loss_training += gradient_img_loss.item()  # Only save the value of gradient_img_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage
        """ print("gradient_img_loss: ", gradient_img_loss) """

        "TODO: The coefficient of this loss function needs to be adjusted"
        if Use_Gram_Matrix_L1_Loss == True:
            gram_similarity_between_img_loss = args_loss_weight['gram_similarity_weight']*loss_function_L1(calculate_gram_matrix(img_outputs), calculate_gram_matrix(labels))
            gram_similarity_between_img_loss_training += gram_similarity_between_img_loss.item()    # Only save the value of gram_similarity_between_img_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage

        if network_model_type == 'Secondary branch is k space branch':
            if Amplify_High_Frequency_Value_In_K_Space_Loss == True:
                k_space_branch_k_space_loss = args_loss_weight['k_space_branch_weight']*(loss_function_MSE(
                    create_2d_Gaussian_weights(window_size = secondary_branch_outputs.shape[2], num_of_samples = secondary_branch_outputs.shape[0], channel = secondary_branch_outputs.shape[1]).to(device)*secondary_branch_outputs[:,:,:,:,0], 
                    create_2d_Gaussian_weights(window_size = HR_freq.shape[2], num_of_samples = HR_freq.shape[0], channel = HR_freq.shape[1]).to(device)*HR_freq[:,:,:,:,0]) + 
                    loss_function_MSE(
                        create_2d_Gaussian_weights(window_size = secondary_branch_outputs.shape[2], num_of_samples = secondary_branch_outputs.shape[0], channel = secondary_branch_outputs.shape[1]).to(device)*secondary_branch_outputs[:,:,:,:,1], 
                        create_2d_Gaussian_weights(window_size = HR_freq.shape[2], num_of_samples = HR_freq.shape[0], channel = HR_freq.shape[1]).to(device)*HR_freq[:,:,:,:,1]))
            else:
                k_space_branch_k_space_loss = args_loss_weight['k_space_branch_weight']*(loss_function_MSE(secondary_branch_outputs[:,:,:,:,0], HR_freq[:,:,:,:,0]) + loss_function_MSE(secondary_branch_outputs[:,:,:,:,1], HR_freq[:,:,:,:,1]))
            k_space_branch_k_space_loss_training += k_space_branch_k_space_loss.item()  # Only save the value of k_space_branch_k_space_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage
        elif network_model_type == 'Secondary branch is gradient map branch':
            gradient_grad_loss = args_loss_weight['gradient_grd_weight']*loss_function_L1(secondary_branch_outputs, calculate_gradient_map(labels))
            gradient_grad_loss_training += gradient_grad_loss.item()    # Only save the value of gradient_grad_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage
        elif network_model_type == 'Secondary branch is wavelets high frequency components branch':
            _, HR_high_frequency_components = calculate_wavelet_transform(labels)
            wavelets_high_frequency_components_branch_high_frequency_loss = args_loss_weight['wavelets_branch_weight']*loss_function_L1(secondary_branch_outputs[:, :, 0, :, :], HR_high_frequency_components[:, :, 0, :, :]) + \
                args_loss_weight['wavelets_branch_weight']*loss_function_L1(secondary_branch_outputs[:, :, 1, :, :], HR_high_frequency_components[:, :, 1, :, :]) + args_loss_weight['wavelets_branch_weight']*loss_function_L1(secondary_branch_outputs[:, :, 2, :, :], HR_high_frequency_components[:, :, 2, :, :])
            wavelets_high_frequency_components_branch_high_frequency_loss_training += wavelets_high_frequency_components_branch_high_frequency_loss.item()  # Only save the value of wavelets_high_frequency_components_branch_high_frequency_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage



#            print('gradient_loss: ', gradient_map_loss)

#            loss = pixel_wise_loss + ssim_loss
        loss = pixel_wise_loss + feature_map_loss

#            if ssim_loss < 0.5:
#                loss = ssim_loss + feature_map_loss + pixel_wise_loss
#                print('ssim_loss')
#            else:
#                loss = pixel_wise_loss + feature_map_loss

        if tc.isnan(k_space_freq_loss) != 1:
            loss = loss + k_space_freq_loss

        if tc.isnan(ssim_loss) != 1 and Use_SSIM_L1_Loss == True:
            loss = loss + ssim_loss

        if tc.isnan(gradient_img_loss) != 1 and Use_Gradient_Map_L1_Loss == True:
            loss = loss + gradient_img_loss

        if Use_Gram_Matrix_L1_Loss == True:
            loss = loss + gram_similarity_between_img_loss

        if network_model_type == 'Secondary branch is k space branch':
            loss = loss + k_space_branch_k_space_loss

        if network_model_type == 'Secondary branch is gradient map branch':
            loss = loss + gradient_grad_loss

        if network_model_type == 'Secondary branch is wavelets high frequency components branch':
            loss = loss + wavelets_high_frequency_components_branch_high_frequency_loss


#            print('loss: ', loss)
#            loss = feature_map_loss + pixel_wise_loss + k_space_freq_loss
            # print('the loss has been checked')


            "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"
#            if tc.isnan(loss) == 1: #- loss == 'NaN':
#                break
            "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"

        loss_training += loss.item()  # Only save the value of loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage    

        "back prop"
        loss.backward()
            # print('the backward pass has gone')

        "update all Variables by using newly fetched gradients"
        optimizer.step() 
        """print('learning rate: %f' % (optimizer.param_groups[0]['lr']))"""
        # print('the all Variables have been updated')

        "print log info"
        running_loss += loss.data
        if i % 50 == 0: #----- print log info every 1000 batch
            if i == 0:
                print('[%d, %5d] loss: %.3f' \
                      % (epoch, i, running_loss))
            else:
                print('[%d, %5d] loss: %.3f' \
                      % (epoch, i, running_loss / 50))
            running_loss = 0.0
        """
        training_loss_for_current_epoch = loss_training / 50
        feature_map_loss_for_current_epoch = feature_map_loss
        pixel_wise_loss_for_current_epoch = pixel_wise_loss
        ssim_loss_for_current_epoch = ssim_loss
        gradient_img_loss_for_current_epoch = gradient_img_loss
        k_space_freq_loss_for_current_epoch = k_space_freq_loss
        if tc.isnan(gram_similarity_between_img_loss) != 1 and Use_Gram_Matrix_L1_Loss == True:
            gram_similarity_between_img_loss_for_current_epoch = gram_similarity_between_img_loss
        if network_model_type == 'Secondary branch is gradient map branch' and tc.isnan(gradient_grad_loss) != 1 and Use_Gradient_Map_L1_Loss == True:
            gradient_grad_loss_for_current_epoch = gradient_grad_loss
        if network_model_type == 'Secondary branch is k space branch' and tc.isnan(k_space_branch_k_space_loss) != 1:
            k_space_branch_k_space_loss_for_current_epoch = k_space_branch_k_space_loss
        if network_model_type == 'Secondary branch is wavelets high frequency components branch' and tc.isnan(wavelets_high_frequency_components_branch_high_frequency_loss) != 1:
            wavelets_high_frequency_components_branch_high_frequency_loss_for_current_epoch = wavelets_high_frequency_components_branch_high_frequency_loss
        """

    print('learning rate for epoch %d is : %f' % (epoch, optimizer.param_groups[0]['lr']))
    learning_rate = optimizer.param_groups[0]['lr']
    
    # If NOT using warm up, call the normal scheduler now
    if args['use_learning_rate_warm_up'] == False:
        scheduler.step()
    
    print('learning rate for next epoch is : %f' % (optimizer.param_groups[0]['lr']))

    batch_number_training = i+1 # Calculate for the current epoch, how many batches are used.

    with tc.no_grad():
        "Set evaluation Mode"    
        our_rcan_mri_sr_2d.eval()
        for i, data in enumerate(validationloader, 0):

            "load input data"
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            SR_img_test, SR_secondary_branch_outputs_test, network_model_type_test = our_rcan_mri_sr_2d(inputs)

            SR_test_copies = tc.cat((SR_img_test, SR_img_test, SR_img_test), 1)
            # print(SR_copies.size())
            SR_test_features = feature_extractor(SR_test_copies)
            # print(SR_features.size())

            HR_test_copies = tc.cat((labels, labels, labels), 1)
            # print(HR_copies.size())
            HR_test_features = feature_extractor(HR_test_copies)
            # print(HR_features.size())

            SR_test_freq = fft_k_space(SR_img_test)

            HR_test_freq = fft_k_space(labels)


            "calculate the gradients for all Variables during back prop"
            "vgg loss + pixel MSE loss + fft frequency loss, and we use weight_decay in Adam so that is L2 regularization"
            feature_map_loss_test += args_loss_weight['feature_map_weight']*loss_function_MSE(SR_test_features, HR_test_features)
#                print("feature_map_loss_test: ", feature_map_loss_test)

            pixel_wise_loss_test += args_loss_weight['pixel_wise_weight']*loss_function_L1(SR_img_test, labels)
#                print("pixel_wise_loss_test: ", pixel_wise_loss_test)

            if Amplify_High_Frequency_Value_In_K_Space_Loss == True:
                k_space_freq_loss_test += args_loss_weight['k_space_weight']*(loss_function_MSE(
                    create_2d_Gaussian_weights(window_size = SR_test_freq.shape[2], num_of_samples = SR_test_freq.shape[0], channel = SR_test_freq.shape[1]).to(device)*SR_test_freq[:,:,:,:,0], 
                    create_2d_Gaussian_weights(window_size = HR_test_freq.shape[2], num_of_samples = HR_test_freq.shape[0], channel = HR_test_freq.shape[1]).to(device)*HR_test_freq[:,:,:,:,0]) + 
                    loss_function_MSE(
                        create_2d_Gaussian_weights(window_size = SR_test_freq.shape[2], num_of_samples = SR_test_freq.shape[0], channel = SR_test_freq.shape[1]).to(device)*SR_test_freq[:,:,:,:,1], 
                        create_2d_Gaussian_weights(window_size = HR_test_freq.shape[2], num_of_samples = HR_test_freq.shape[0], channel = HR_test_freq.shape[1]).to(device)*HR_test_freq[:,:,:,:,1]))
            else:
                k_space_freq_loss_test += args_loss_weight['k_space_weight']*(loss_function_MSE(SR_test_freq[:,:,:,:,0], HR_test_freq[:,:,:,:,0])+loss_function_MSE(SR_test_freq[:,:,:,:,1], HR_test_freq[:,:,:,:,1]))
#                print("k_space_freq_loss_test: ", k_space_freq_loss_test)

            HR_ssim_test_weighted, HR_ssim_test = SSIM_function(labels,labels)
            SR_ssim_test_weighted, SR_ssim_test = SSIM_function(SR_img_test, labels)
            HR_ssim_map_test_weighted, HR_ssim_map_test = SSIM_map(labels, labels)
            SR_ssim_map_test_weighted, SR_ssim_map_test = SSIM_map(SR_img_test, labels)
            ssim_loss_test += args_loss_weight['ssim_weight']*loss_function_L1(SR_ssim_map_test_weighted, HR_ssim_map_test_weighted)
            ssim_test += SR_ssim_test   # Accumulation of SR_ssim in testing over all batches in one epoch, will be used to calculate the average value of SR SSIM for one epoch.
#                print("ssim_loss_test: ", ssim_loss_test)

            gradient_img_loss_test += args_loss_weight['gradient_img_weight']*loss_function_L1(calculate_gradient_map(SR_img_test), calculate_gradient_map(labels))
#                print('gradient_loss_test: ', gradient_map_loss_test)

            if Use_Gram_Matrix_L1_Loss == True:
                gram_similarity_between_img_loss_test = args_loss_weight['gram_similarity_weight']*loss_function_L1(calculate_gram_matrix(SR_img_test), calculate_gram_matrix(labels))

            if network_model_type_test == 'Secondary branch is k space branch':
                if Amplify_High_Frequency_Value_In_K_Space_Loss == True:
                    k_space_branch_k_space_loss_test += args_loss_weight['k_space_branch_weight']*(loss_function_MSE(
                        create_2d_Gaussian_weights(window_size = SR_secondary_branch_outputs_test.shape[2], num_of_samples = SR_secondary_branch_outputs_test.shape[0], channel = SR_secondary_branch_outputs_test.shape[1]).to(device)*SR_secondary_branch_outputs_test[:,:,:,:,0], 
                        create_2d_Gaussian_weights(window_size = HR_test_freq.shape[2], num_of_samples = HR_test_freq.shape[0], channel = HR_test_freq.shape[1]).to(device)*HR_test_freq[:,:,:,:,0]) + 
                        loss_function_MSE(
                            create_2d_Gaussian_weights(window_size = SR_secondary_branch_outputs_test.shape[2], num_of_samples = SR_secondary_branch_outputs_test.shape[0], channel = SR_secondary_branch_outputs_test.shape[1]).to(device)*SR_secondary_branch_outputs_test[:,:,:,:,1], 
                            create_2d_Gaussian_weights(window_size = HR_test_freq.shape[2], num_of_samples = HR_test_freq.shape[0], channel = HR_test_freq.shape[1]).to(device)*HR_test_freq[:,:,:,:,1]))
                else:
                    k_space_branch_k_space_loss_test += args_loss_weight['k_space_branch_weight']*(loss_function_MSE(SR_secondary_branch_outputs_test[:,:,:,:,0], HR_test_freq[:,:,:,:,0]) + loss_function_MSE(SR_secondary_branch_outputs_test[:,:,:,:,1], HR_test_freq[:,:,:,:,1]))
            elif network_model_type_test == 'Secondary branch is gradient map branch':
                gradient_grad_loss_test += args_loss_weight['gradient_grd_weight']*loss_function_L1(SR_secondary_branch_outputs_test, calculate_gradient_map(labels))
            elif network_model_type_test == 'Secondary branch is wavelets high frequency components branch':
                _, HR_high_frequency_components_test = calculate_wavelet_transform(labels)
                wavelets_high_frequency_components_branch_high_frequency_loss_test += args_loss_weight['wavelets_branch_weight']*loss_function_L1(SR_secondary_branch_outputs_test[:, :, 0, :, :], HR_high_frequency_components_test[:, :, 0, :, :]) + \
                    args_loss_weight['wavelets_branch_weight']*loss_function_L1(SR_secondary_branch_outputs_test[:, :, 1, :, :], HR_high_frequency_components_test[:, :, 1, :, :]) + args_loss_weight['wavelets_branch_weight']*loss_function_L1(SR_secondary_branch_outputs_test[:, :, 2, :, :], HR_high_frequency_components_test[:, :, 2, :, :])

            loss_test = pixel_wise_loss_test + feature_map_loss_test
#                print('loss_test: ', loss_test)

            if tc.isnan(k_space_freq_loss_test) != 1:
                loss_test = loss_test + k_space_freq_loss_test

            if tc.isnan(ssim_loss_test) != 1 and Use_SSIM_L1_Loss == True:
                loss_test = loss_test + ssim_loss_test

            if tc.isnan(gradient_img_loss_test) != 1 and Use_Gradient_Map_L1_Loss == True:
                loss_test = loss_test + gradient_img_loss_test

            if Use_Gram_Matrix_L1_Loss == True:
                loss_test = loss_test + gram_similarity_between_img_loss_test

            if network_model_type_test == 'Secondary branch is k space branch':
                loss_test = loss_test + k_space_branch_k_space_loss_test

            if network_model_type_test == 'Secondary branch is gradient map branch':
                loss_test = loss_test + gradient_grad_loss_test

            if network_model_type_test == 'Secondary branch is wavelets high frequency components branch':
                loss_test = loss_test + wavelets_high_frequency_components_branch_high_frequency_loss_test

    batch_number_test = i+1
    ssim_test = ssim_test/batch_number_test # Calculate avergae SR SSIM over all validation data samples in one epoch.
    print("ssim_test: ", ssim_test)
    feature_map_loss_test = feature_map_loss_test/batch_number_test
    print("feature_map_loss_test: ", feature_map_loss_test)
    pixel_wise_loss_test = pixel_wise_loss_test/batch_number_test
    print("pixel_wise_loss_test: ", pixel_wise_loss_test)
    k_space_freq_loss_test = k_space_freq_loss_test/batch_number_test
    print("k_space_freq_loss_test: ", k_space_freq_loss_test)
    ssim_loss_test = ssim_loss_test/batch_number_test
    print("ssim_loss_test: ", ssim_loss_test)
    gradient_img_loss_test = gradient_img_loss_test/batch_number_test
    print('gradient_img_loss_test: ', gradient_img_loss_test)
    if Use_Gram_Matrix_L1_Loss == True:
        gram_similarity_between_img_loss_test = gram_similarity_between_img_loss_test/batch_number_test
        print('gram_similarity_between_img_loss_test: ', gram_similarity_between_img_loss_test)
    if network_model_type_test == 'Secondary branch is k space branch':
        k_space_branch_k_space_loss_test = k_space_branch_k_space_loss_test/batch_number_test
        print('k_space_branch_k_space_loss_test: ', k_space_branch_k_space_loss_test)
    if network_model_type_test == 'Secondary branch is gradient map branch':
        gradient_grad_loss_test = gradient_grad_loss_test/batch_number_test
        print('gradient_grad_loss_test: ', gradient_grad_loss_test)
    if network_model_type_test == 'Secondary branch is wavelets high frequency components branch':
        wavelets_high_frequency_components_branch_high_frequency_loss_test = wavelets_high_frequency_components_branch_high_frequency_loss_test/batch_number_test
        print('wavelets_high_frequency_components_branch_high_frequency_loss_test: ', wavelets_high_frequency_components_branch_high_frequency_loss_test)
    loss_test = loss_test/batch_number_test
    print('loss_test: ', loss_test)
    
    ssim_training = ssim_training/batch_number_training # Calculate avergae SR SSIM over all training data samples in one epoch.
    training_loss_for_current_epoch = loss_training / batch_number_training
    feature_map_loss_for_current_epoch = feature_map_loss_training/ batch_number_training
    pixel_wise_loss_for_current_epoch = pixel_wise_loss_training/ batch_number_training
    ssim_loss_for_current_epoch = ssim_loss_training/ batch_number_training
    gradient_img_loss_for_current_epoch = gradient_img_loss_training/ batch_number_training
    k_space_freq_loss_for_current_epoch = k_space_freq_loss_training/ batch_number_training
    if Use_Gram_Matrix_L1_Loss == True:
        gram_similarity_between_img_loss_for_current_epoch = gram_similarity_between_img_loss_training/ batch_number_training
    if network_model_type == 'Secondary branch is gradient map branch':
        gradient_grad_loss_for_current_epoch = gradient_grad_loss_training/ batch_number_training
    if network_model_type == 'Secondary branch is k space branch':
        k_space_branch_k_space_loss_for_current_epoch = k_space_branch_k_space_loss_training/ batch_number_training
    if network_model_type == 'Secondary branch is wavelets high frequency components branch':
        wavelets_high_frequency_components_branch_high_frequency_loss_for_current_epoch = wavelets_high_frequency_components_branch_high_frequency_loss_training/ batch_number_training


    "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"    
#    if tc.isnan(loss) == 1: #- loss == 'NaN':
#        break
    "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"


    "Save the weights of network model when it achieves best average SR SSIM over all validation data samples in one epoch"
    print("best_ssim:", best_ssim)
    print("validation_ssim:", ssim_test)
    if ssim_test > best_ssim:
        best_ssim = ssim_test
        best_model_wts = copy.deepcopy(our_rcan_mri_sr_2d.state_dict())
        parameter_file = open('D:/Tech_Resource/Paper_Resource/MRI SR以及相关论文/our_project_code/code/baseline+our_model/output_file/network_weight_parameter.pkl', 'wb')
        pickle.dump(best_model_wts, parameter_file)
        parameter_file.close()
        best_epoch = epoch


    "Save the training loss for each epoch"
    if (epoch == 0):
        f = open('D:/Tech_Resource/Paper_Resource/MRI SR以及相关论文/our_project_code/code/baseline+our_model/output_file/result_RCAN_l1_gradssim_laf_100_32_2folds_2d_downsize_20200816.txt', 'w')
        f.write('The configuration of parameters:\n')
        f.write('batch_size is: %d\n' % batch_size)
        f.write('EPOCH_NUM is: %d\n' % EPOCH_NUM)
        f.write('Maintain_Same_Size is: %s\n' % Maintain_Same_Size)
        f.write('Use_SSIM_L1_Loss is: %s\n' % Use_SSIM_L1_Loss)
        f.write('Use_Gradient_Map_L1_Loss is: %s\n' % Use_Gradient_Map_L1_Loss)
        f.write('Use_Gram_Matrix_L1_Loss is: %s\n' % Use_Gram_Matrix_L1_Loss)
        f.write('Use_Channel_Attention_For_Cross_Branch_Fusion is: %s\n' % Use_Channel_Attention_For_Cross_Branch_Fusion)
        f.write('Amplify_Small_Value_In_Gradient_Map: %s\n' % Amplify_Small_Value_In_Gradient_Map)
        f.write('Amplify_High_Frequency_Value_In_K_Space_Loss: %s\n' % Amplify_High_Frequency_Value_In_K_Space_Loss)
        f.write('args is: %s\n' % args)
        f.write('args of loss weight is: %s\n' % args_loss_weight)
        f.write('------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        f.write(' \n')
#    f.write('Training Loss:')
#    f.write('\n')
    f.write('Best SR SSIM for validation data has been achieved at epoch : %d' % (best_epoch))
    f.write('\n')    
    f.write('Learning rate for current epoch is : %f' % (learning_rate))
    f.write('\n')
    f.write('Training SR SSIM for epoch %d is : %f' % (epoch, ssim_training))
    f.write('\n')
    f.write('The feature_map_loss for epoch %d is : %f' % (epoch, feature_map_loss_for_current_epoch))
    f.write('\n')
    f.write('The pixel_wise_loss for epoch %d is : %f' % (epoch, pixel_wise_loss_for_current_epoch))
    f.write('\n')
    f.write('The k_space_freq_loss for epoch %d is : %f' % (epoch, k_space_freq_loss_for_current_epoch))
    f.write('\n')
    f.write('The ssim_loss for epoch %d is : %f' % (epoch, ssim_loss_for_current_epoch))
    f.write('\n')
    f.write('The gradient_img_loss for epoch %d is : %f' % (epoch, gradient_img_loss_for_current_epoch))
    f.write('\n')
    if Use_Gram_Matrix_L1_Loss == True:
        f.write('The gram_similarity_between_img_loss for epoch %d is : %f' % (epoch, gram_similarity_between_img_loss_for_current_epoch))
        f.write('\n')
    if network_model_type == 'Secondary branch is gradient map branch':
        f.write('The gradient_grad_loss for epoch %d is : %f' % (epoch, gradient_grad_loss_for_current_epoch))
        f.write('\n')
    if network_model_type == 'Secondary branch is k space branch':
        f.write('The k_space_branch_k_space_loss for epoch %d is : %f' % (epoch, k_space_branch_k_space_loss_for_current_epoch))
        f.write('\n')
    if network_model_type == 'Secondary branch is wavelets high frequency components branch':
        f.write('The wavelets_high_frequency_components_branch_high_frequency_loss for epoch %d is : %f' % (epoch, wavelets_high_frequency_components_branch_high_frequency_loss_for_current_epoch))
        f.write('\n')
    f.write('The training_loss for epoch %d is : %f' % (epoch, training_loss_for_current_epoch))
    f.write('\n')
    f.write(' \n')


#    f.write('Validation Loss:')
#    f.write('\n')
    f.write('Validation SR SSIM for epoch %d is : %f' % (epoch, ssim_test))
    f.write('\n')    
    f.write('The feature_map_loss_validation for epoch %d is : %f' % (epoch, feature_map_loss_test))
    f.write('\n')
    f.write('The pixel_wise_loss_validation for epoch %d is : %f' % (epoch, pixel_wise_loss_test))
    f.write('\n')
    f.write('The k_space_freq_loss_validation for epoch %d is : %f' % (epoch, k_space_freq_loss_test))
    f.write('\n')
    f.write('The ssim_loss_validation for epoch %d is : %f' % (epoch, ssim_loss_test))
    f.write('\n')
    f.write('The gradient_img_loss_validation for epoch %d is : %f' % (epoch, gradient_img_loss_test))
    f.write('\n')
    if Use_Gram_Matrix_L1_Loss == True:
        f.write('The gram_similarity_between_img_loss_validation for epoch %d is : %f' % (epoch, gram_similarity_between_img_loss_test))
        f.write('\n')
    if network_model_type_test == 'Secondary branch is gradient map branch':
        f.write('The gradient_grad_loss_validation for epoch %d is : %f' % (epoch, gradient_grad_loss_test))
        f.write('\n')
    if network_model_type_test == 'Secondary branch is k space branch':
        f.write('The k_space_branch_k_space_loss_validation for epoch %d is : %f' % (epoch, k_space_branch_k_space_loss_test))
        f.write('\n')
    if network_model_type_test == 'Secondary branch is wavelets high frequency components branch':
        f.write('The wavelets_high_frequency_components_branch_high_frequency_loss_validation for epoch %d is : %f' % (epoch, wavelets_high_frequency_components_branch_high_frequency_loss_test))
        f.write('\n')
    f.write('The validation_loss for epoch %d is : %f' % (epoch, loss_test))
    f.write('\n')
    f.write('----------------------------------------------------------------------------------')
    f.write(' \n')
    f.write(' \n')
    if (epoch == EPOCH_NUM - 1):
        f.close()

"Reload best weight parameters into the network model, which will be used for evaludation in the following part"
our_rcan_mri_sr_2d.load_state_dict(best_model_wts)

print("training complete")





"Evaluation"
# Please beware the data should be Evaluation data rather than training data
folder_log_path = '/srv/DATA/RAID/HaoLi/Data/'
file_names = os.listdir(folder_log_path)

num_low_resolution_mat_file = 0
num_high_resolution_groundtruth_mat_file = 0

for idx_file in file_names:
    print(idx_file)
    if 'LR_eval_4' in os.path.join(folder_log_path, idx_file):
        print('One more low resolution image set exist')
        num_low_resolution_mat_file = num_low_resolution_mat_file + 1
        print(os.path.join(folder_log_path, idx_file))
        file_data_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
        data_low_resolution = file_data_low_resolution['LR'][:] #----- numpy array
        print(np.shape(data_low_resolution))
        torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_low_resolution = torch_data_low_resolution.permute(0, 2, 1)
        print(np.shape(torch_data_low_resolution))
        if num_low_resolution_mat_file == 1:
            torch_data_low_resolution_sequence = torch_data_low_resolution
        elif num_low_resolution_mat_file > 1:
            print(num_low_resolution_mat_file)
            torch_data_low_resolution_sequence = tc.cat((torch_data_low_resolution_sequence, torch_data_low_resolution), 0)
        print(np.shape(torch_data_low_resolution_sequence))
    elif 'HRGT_eval_4' in os.path.join(folder_log_path, idx_file):
        print('One more high resolution groundtruth image set exist')
        num_high_resolution_groundtruth_mat_file = num_high_resolution_groundtruth_mat_file + 1
        print(os.path.join(folder_log_path, idx_file))
        file_data_high_resolution_groundtruth = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
        data_high_resolution_groundtruth = file_data_high_resolution_groundtruth['HRGT'][:] #----- numpy array
        print(np.shape(data_high_resolution_groundtruth))
        torch_data_high_resolution_groundtruth = tc.from_numpy(data_high_resolution_groundtruth) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.permute(0, 2, 1)
        print(np.shape(torch_data_high_resolution_groundtruth))
        if num_high_resolution_groundtruth_mat_file == 1:
            torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth
        if num_high_resolution_groundtruth_mat_file > 1:
            print(num_high_resolution_groundtruth_mat_file)
            torch_data_high_resolution_groundtruth_sequence = tc.cat((torch_data_high_resolution_groundtruth_sequence, torch_data_high_resolution_groundtruth), 0)
        print(np.shape(torch_data_high_resolution_groundtruth_sequence))
    else:
        print('other type NOT support for now')

torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.unsqueeze(1)    
print(np.shape(torch_data_low_resolution_sequence))
torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.unsqueeze(1)  
print(np.shape(torch_data_high_resolution_groundtruth_sequence))

torch_data_low_resolution_eval_sequence = torch_data_low_resolution_sequence.float()
torch_data_high_resolution_groundtruth_eval_sequence = torch_data_high_resolution_groundtruth_sequence.float()

testset = tc.utils.data.TensorDataset(torch_data_low_resolution_eval_sequence, torch_data_high_resolution_groundtruth_eval_sequence)

new_batch_size_for_checking = 32

testloader = tc.utils.data.DataLoader(
                    testset, 
                    batch_size = new_batch_size_for_checking,
                    shuffle = False, 
                    num_workers = 0)        


"Resetup the batch size for train and test data, to avoid the errorCUDA out of memory"
#trainloader = tc.utils.data.DataLoader(
#                    trainset, 
#                    batch_size = new_batch_size_for_checking,
#                    shuffle = True, 
#                    num_workers = 0)

with tc.no_grad():
    our_rcan_mri_sr_2d.eval()
#    "exam the generated SR MRI image by using training LR image data and save them"
    # =============================================================================
    # for data in testloader:
    # =============================================================================
#    for i, training_data_2 in enumerate(trainloader, 0):

        #    LR_images_training, HR_images_training = training_data_2

        # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
        #    outputs = our_rcan_mri_sr_2d(Variable(LR_images_training).type(tc.FloatTensor).to(device))
        # =============================================================================
        #     print(outputs.data.size())
        # =============================================================================

#        if (i == math.floor((torch_data_low_resolution_training_sequence.size(0)/new_batch_size_for_checking)/2)): 

#            LR_images_training, HR_images_training = training_data_2
            # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
#            img_outputs, grad_outputs = our_rcan_mri_sr_2d(Variable(LR_images_training).type(tc.FloatTensor).to(device))

            #----- skip display "the last batch for one epoch test data" and skip "all the batches expect the batch in the middle"
#            SR_images_tensor_training = img_outputs.data.cpu().squeeze(1)
#            SR_grad_tensor_training = grad_outputs.data.cpu().squeeze(1)
            # HR_images_tensor = HR_images_temp.cpu().squeeze(1)
            # =============================================================================
            #         print(SR_images_tensor.size())
            # =============================================================================
            # print(HR_images_tensor.size())    
#            SR_images_exam_train = SR_images_tensor_training.numpy()
            # HR_images_test = HR_images_tensor.numpy()

#            "save the .mat files for SR LR, HR training images"
#            scipy.io.savemat('D:/HaoLi/SR/results/200730_RCAN_l1_ssim_grad_laf_100_32_2folds_2d_downsize_Dual/HR_training_image.mat', mdict = {'HR_training_image' : HR_images_training.numpy()})
#            scipy.io.savemat('D:/HaoLi/SR/results/200730_RCAN_l1_ssim_grad_laf_100_32_2folds_2d_downsize_Dual/LR_training_image.mat', mdict = {'LR_training_image' : LR_images_training.numpy()})
#            scipy.io.savemat('D:/HaoLi/SR/results/200730_RCAN_l1_ssim_grad_laf_100_32_2folds_2d_downsize_Dual/SR_training_image.mat', mdict = {'SR_training_image' : SR_images_exam_train})



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

#    print("examination of generated SR image by using training samples complete")




    "predict the SR MRI image by using testing LR image data and save them"
    # =============================================================================
    # for data in testloader:
    # =============================================================================
    for i, testing_data_2 in enumerate(testloader, 0):

        #    LR_images_test, HR_images_test = testing_data_2
        # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
        #    outputs = our_rcan_mri_sr_2d(Variable(LR_images_test).type(tc.FloatTensor).to(device))
        # =============================================================================
        #     print(outputs.data.size())
        # =============================================================================

#       if (i == math.floor((torch_data_low_resolution_test_sequence.size(0)/new_batch_size_for_checking)/2)): 

        LR_images_test, HR_images_test = testing_data_2
        HR_images_test = HR_images_test.type(tc.FloatTensor)
        img_outputs, secondary_branch_outputs, network_model_type = our_rcan_mri_sr_2d(Variable(LR_images_test).type(tc.FloatTensor).to(device))

        #----- skip display "the last batch for one epoch test data" and skip "all the batches expect the batch in the middle"
        SR_images_tensor_test = img_outputs.data.cpu().squeeze(1)
        if network_model_type != 'Single Branch Network':
            SR_secondary_branch_tensor_test = secondary_branch_outputs.data.cpu().squeeze(1)
        LR_images_tensor_test = LR_images_test.cpu().squeeze(1)
        HR_images_tensor_test = HR_images_test.cpu().squeeze(1)
        # HR_images_tensor = HR_images_temp.cpu().squeeze(1)
        # =============================================================================
        #         print(SR_images_tensor.size())
        # =============================================================================
        # print(HR_images_tensor.size())    

        # HR_images_test = HR_images_tensor.numpy()
        if i==0:
            SR_img_eval_tensor = SR_images_tensor_test
            if network_model_type != 'Single Branch Network':
                SR_secondary_branch_eval_tensor = SR_secondary_branch_tensor_test
            LR_eval_tensor = LR_images_tensor_test
            HR_eval_tensor = HR_images_tensor_test
        else:
            SR_img_eval_tensor = tc.cat((SR_img_eval_tensor, SR_images_tensor_test), 0)
            if network_model_type != 'Single Branch Network':
                SR_secondary_branch_eval_tensor = tc.cat((SR_secondary_branch_eval_tensor, SR_secondary_branch_tensor_test), 0)
            LR_eval_tensor = tc.cat((LR_eval_tensor, LR_images_tensor_test), 0)
            HR_eval_tensor = tc.cat((HR_eval_tensor, HR_images_tensor_test), 0)

    SR_images_test = SR_img_eval_tensor.numpy()
    if network_model_type != 'Single Branch Network':
        SR_secondary_branch_test = SR_secondary_branch_eval_tensor.numpy()
    LR_images_test = LR_eval_tensor.numpy()
    HR_images_test = HR_eval_tensor.numpy()
    "save the .mat files for SR, HR and LR training images"
    scipy.io.savemat('/srv/DATA/RAID/HaoLi/Results/20201027_RCAN_l1_ssim_grad_cosine_warm_restarts_1000_32_2folds_2d_downsize_baseline/SR_test_image.mat', mdict = {'SR_test_image' : SR_images_test})
    if network_model_type != 'Single Branch Network':
        scipy.io.savemat('/srv/DATA/RAID/HaoLi/Results/20201027_RCAN_l1_ssim_grad_cosine_warm_restarts_1000_32_2folds_2d_downsize_baseline/SR_secondary_branch_test.mat', mdict = {'SR_secondary_branch_test' : SR_secondary_branch_test})
    scipy.io.savemat('/srv/DATA/RAID/HaoLi/Results/20201027_RCAN_l1_ssim_grad_cosine_warm_restarts_1000_32_2folds_2d_downsize_baseline/HR_test_image.mat', mdict = {'HR_test_image' : HR_images_test})
    scipy.io.savemat('/srv/DATA/RAID/HaoLi/Results/20201027_RCAN_l1_ssim_grad_cosine_warm_restarts_1000_32_2folds_2d_downsize_baseline/LR_test_image.mat', mdict = {'LR_test_image' : LR_images_test})


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

now = time.perf_counter()

running_time = now - since
print('The spent time in minute is: ', running_time/60)
