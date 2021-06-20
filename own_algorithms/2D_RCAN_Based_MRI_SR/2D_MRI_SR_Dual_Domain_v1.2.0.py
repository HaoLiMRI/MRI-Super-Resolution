# -*- coding: utf-8 -*-
"-------------------------------------------------------------------------------------------------"
"""
2D_MRI_SR_Dual_Domain Reconstruct 
(2D Dual Domain Fusion Network for MRI Super-Resolution Image Reconstruction and MRI Motion Artifact Reduction)
This is the code for Dual Domain Fusion Network for MRI SR Reconstruction.
"""
"""
Author: chisyliu@hotmail.com *
        hao.li@med.uni-heidelberg.de *
        
        * Both authors contribute equally
Version: 1.4.0(Stable Version, even deformable conv works at least for RCAN network)
"""
"-------------------------------------------------------------------------------------------------"
"""
This is the current version we are working on, in 20210619
This is a demo code of 2D_MRI_SR_Dual_Domain. in this version we have already support following items:
    0)  Dual Domain Fusion Network Achitecture, where we already support:
        a) use RCAN or U-Net as main framework, for image single branch network.
        b) gradient map branch as secondary branch, together with main(image) branch, in the framework of RCAN
        b) k space branch as secondary branch, together with main(image) branch, in the framework of RCAN
        c) high frequency component extracted using wavelet transformation and wavelet branch as 
            secondary branch, together with main(image) branch, in the framework of RCAN
        d) interleaving fusion between image branch and  as secondary branch at intermedian level
        e) late stage fusion of outcome from image branch and outcome from secondary branchbranch
    Besides, we already support other features:
    1)  option to add a VGG feature extractor in front of the main "CNN based Reconstruct network"(so the input to "CNN based Reconstruct network" is feature map of LR image)
    2)  either Pixel-Wise MSE loss or Pixel-Wise smooth L1 loss (L1_Charbonnier_Loss is also availbe). 
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
    24) option to use "normal end to end channel and spatial attention block"(CAM and SAM are implemented for channel and spatial attention, see paper:2018.CBAM: Convolutional Block Attention Module
        for more detail information.) for upsampler (inside upsampler, after first conv and before pixel shuffle).
    25) option to use "self-attention based end to end channel and spatial attention block"(The non-local self-attention based channel attention and non-local self-attention are implemented according 
        to paper: 2018.Dual Attention Network for Scene Segmentation) for upsampler. The non-local self-attention spatial and channel attention mechanism employed in DANet is, the Self-Attention(Also 
        called as Non-Local Attention, see paper: 2018.Non-local Neural Networks and paper: 2019.Self-Attention Generative Adversarial Networks for more details about Self-Attention) which borrows from 
        the self-attention mechanism in the classical paper in NLP which proposes the Transformer technology: 2017.Attention is All You Needed.
    26) option to use "normal end to end channel and spatial attention block"(CAM and SAM in CBAM framework) and "self-attention based end to end channel and spatial attention block"(self_attention framework)
        as basic block in every RCAB(for both main branch and second branch if dual branch network is turned), to replace the CALayer.
    27) For all the "normal end to end channel and spatial attention block" and "self-attention based end to end channel and spatial attention block", either "sequential_mode" or "parallel_mode" could be selected.
    28) option to use "negative total variation loss(tv_loss)". minimize总变差（TV）loss促进了生成的图像中的空间平滑性,于是minimize negative tv loss防止过度平滑
    29) option to use "negative trace loss". 用minimize 1/trace(SR*HR)做为negative trace loss。trace(SR*HR)表示SR和HR的相似程度。两个向量内积是把一个向量投影到另一个上的长度，这个值可以用于描述两个向量的相似性。两个矩阵A、B的相似性
        可以用A、B两个矩阵的内积表征，被定义为Trace(AB)。见paper: 2015.LRTV: MR Image Super-Resolution With Low-Rank and Total Variation Regularizations
    30) option to use HR reference with self-attention in the end. 使用MRI HR reference的MRI SR网络并联两个现有的branch，一个branch用于LR的放大，另一个用于给HR reference的feature extraction（去掉upsampler），
        最后用一个self-attention的upsampler来把俩者fuse到一起生成MRI SR。这个方案的思路是用CNN去抓取LR图像和HR图像的局部特征的feature，然后用self-attention方案去找到这些局部feature在整个图上（全局上，更大的范围）的关系。
        模型的结构可以参考Paper: 2020.Attention-based Image Upsampling. https://arxiv.org/abs/2012.09904
    31) option to add long skip connection outside the entire network model to only reconstruct the residual part of HR MRI image.从而让网络从用LR生成SR变为用LR恢复SR和LR+bicubic padding相差的部分。
    32) option to use Gradient map guided pixel-wise loss. SR和HR分别求gradient map，再相减得到一个gradient map差的矩阵，再把这个gradient map差的矩阵从(H * W)变为(1 * HW)，然后再过一个softmax，再变回H * W，然后把得到的矩
        阵当做pixel-wise L1 loss的weight来元素乘在L1 loss的pixel上。
    33) option to use SSIM map guided pixel-wise loss. SR和HR求SSIM map，再用1减这个SSIM map得到一个矩阵当做pixel-wise L1 loss的weight来元素乘在L1 loss的pixel上。
    34) option to use external-attention. Which is a pure MLP based 'self-attention'. See paper: '2021.Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks' for more detail.
    35) option to use involution conv(but still has some bugs when using involution conv).
    36) option to use gMLP or aMLP, which is another "pure MLP" or "pure MLP with tiny attention" module. See paper: "2021.Pay Attention to MLPs" for more info.
    37) option to use uncertainty loss to estimate the uncertainty map(pixel-wise variance of estimated MRI SR output) of MRI SR image.
        注意：uncertainty KL loss相关代码已经完成，但是由于需要使用的network model在最后输出MRI SR结果时多输出一个channel的数据作为variance(或者log of variance)，
        同时我们决定在这个版本中不修改任何network model的最后输出，所以现在的代码无法运行uncertainty KL loss。如果需要运行uncertainty KL loss，则需要更新相应的
        network model，最后输出MRI SR结果时多输出一个channel的数据作为variance(或者log of variance)。


Some feature or bug fixing which have already been planed/started but still not finished yet:
    1) U-Net is already used as framework for image single branch. But not support for any of dual domain branch yet! Thus the class "U_Net_Based_MRI_SR_Dual_Domain_2D" still needs to be
        changed to support U-Net as framework for 3 types of dual domain branch.
    2) k space, wavelet secondary branch多个分量间分开，各走一个branch来实现。
    3) option to use HR reference with self-attention in the end这个方案已经代码已经完成。但现在只支持RCAN的image_single_domain或者gradient_map_dual_domain在没有long_skip_connection_to_reconstruct_residual_part_only时的
        HR reference based network。不支持其他配置时的HR reference based network。注意：现在gradient_map_dual_domain时用HR reference based network还有问题，会out of memory。
    4) 另外，现在option to use HR reference with self-attention in the end这个方案如果在网络用self-attention，则会out of memory。
    5) option to add long skip connection outside the entire network model to only reconstruct the residual part of HR MRI image这个选项现阶段仅支持非HR Reference based的网络结构。
    6) option to use involution conv, still has bugs which makes the loss as nan. 


We will plan to support other features:
    1) multi-kernel size deformable conv in different paths and fuse together, see [26] for similar idea
    2) kernel size wise attention[25] for multi-kernel size deformable conv
    3) multi-kernel size dilated conv in different paths and fuse together[26]
    4) kernel size wise attention[25] for multi-kernel size dilated conv
    5) feature scale wise attention for py_conv
    6) set threshold_low and threshold_high for "contrast between each pixel and all the pixels around it", if contrast is
        lower than threshold_high, we have to limit the contrast to let it should be larger than threshold_low. The actual
        threshold_low for contrast between each pixel and all the pixels around it may follow Gaussian distribution(The 
        closer the pixels are the larger threshold_low should be, doing like this lead the contrast between two pixels which
        are close to each other large enough, so they will NOT be samiliar and the super resolution result will NOT be too
        smooth in texture wise.)
    7) 完成基于He Kaiming的paper: 2019.Panoptic Feature Pyramid Networks内figure 3提出的为semantic segmentation任务提出的Panoptic FPN方案来增强U-Net framework对于多尺度信息的提取恢复。
        现阶段已经用"Channel and Spatial Attention Block"模块替换掉了CA Lyaer从而组成新的RCSAB，然后多个RCSAB构成新的RG，每个RG作为U-Net framework中的encoder的每一层。基于这种U-Net 
        framework我们可以在decoder的所有层连上这种Panoptic FPN结构从而构成Panoptic U-Net framework.
    8) non-local edge attention. 受paper: 2019.Local Relation Networks for Image Recognition中figure2的启发，那个图它引入了一个什么geometry prior，然后说要对each spatial position来做self-attention。
        我们可以像它这样，但不对每一个spatial position来做，而是做self-attention的部分引入一个LR的图的Gradient map, 然后将这个gradient map reshape成为一个1 x NW的向量，再和自己的转置相乘得到一个
        pixel-wise的互相关矩阵，再通过一个softmax或sigmoid（和self-attention的通过Pixel-wise互相关矩阵求取每个pixel和其他所有pixel之间的相关性再通过softmax的操作逻辑一样）变成一个归一化了的权重矩阵。
        再将这个基于Gradient map求出来的权重矩阵乘以self-attention模块中的Value矩阵，然后再和“key和query求Pixel-wise的互相关矩阵过softmax之后得到的权重矩阵”再相乘，从而输出一个同时被non-local 
        self-attention强调和gradient map edge强调过的feature map。那这个non-local edge attention模块同时放在整个网络的最前面（做第一个block）给LR输入加一个edge attention的guidence，和最后面（做最
        后一个block）给SR输出加一个edge attention的guidence。
        **Beware: 这个idea应该对MRI segmentation一样靠谱！！！！！
    9) 那么对于8)中提出的基于HR reference的MRI SR网络，其实也可以对输入的HR reference MRI数据在网络的一开始做non-local edge attention，以及fft求k space去掉低频仅保留高频再ifft之后做non-local edge attention
        类似的操作求出一个high frequency self-attention，这样得到一个被gradient map edge强调和的high frequency self-attention强调的feature map来和LR生成SR的branch进行fuse。
    10) consider adding dual regression loss(See paper: 2020.Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution).
    11) Pair-wise and Patch-wise Attention. See paper: 2019.Exploring self-attention for image recognition
    12) Criss-cross Attention. Criss-cross Attention could reduce the computational burden which introduces from non-local self-attention block(has a high complexity of O(N2), where N denotes 
        the number of input feature maps). The criss-cross attention module that for each pixel position generates a sparse attention map only on the criss-cross path. Further, by applying 
        criss-cross attention recurrently, each pixel position can capture context from all other pixels. Compared to non-local self-attention block, the criss-cross uses 11× lesser GPU memory, 
        and has a complexity of O(2√N).
        See paper: 2019.CCNet: Criss-cross attention for semantic segmentation
    13) 看懂PC（Phase Congruency）怎么算，把这个指标做loss项。 参考论文：2011.FSIM: A Feature Similarity Index for Image Quality Assessment 参考代码：https://github.com/sunxirui310/FSIM-FSIMc-matlab/blob/master/FSIM.m
    14) 受paper: 2019.Local Relation Networks for Image Recognition中figure2的启发，那个图它引入了一个什么geometry prior，然后说要对each spatial position来做self-attention。我们可以像它这样，但不对每一个spatial position来做，
        而是在option to use HR reference with self-attention in the end方案实现using HR reference with self-attention in the end那样最后做self-attention的部分引入一个比如LR的图的Gradient map像它这个geometry prior一样加到self-attention里面。
    15) 还是我们自己的HR reference网络，也应该让HR reference经过小波变换，然后只保留高频部分进入网络帮助LR做SR。方案一：对于我们自己的HR reference网络可以考虑对HR reference数据做DWT（小波变换），只保留3个高频分量。
        然后对LR数据像“Inverse-wavelet-only RCAN”这个idea一样处理，直接用网络试图从LR数据生成HR图像的小波分量，然后把用网络对LR数据生成的3个高频小波分量和FR reference做DWT得到的3个HR reference的高频分量进行fusion。
        最后再把LR数据用网络试图直接生成的一个低频分量和fuse后的3个高频分量一起做IDWT（小波反变换）恢复2x SR图像。如果是4x则用progressive构架重复2次2x放大的该方案。
    16) 还是我们自己的HR reference网络，也应该让HR reference经过小波变换，然后只保留高频部分进入网络帮助LR做SR。方案二：对于我们自己的HR reference网络可以考虑把LR复制3份，分别于HR reference的小波变换的3个高频分量
        各自过self-attention一起组成multi-head self-attention。




We also fixed bugs from previous versions, typical ones like:
    1) After PyTorch version 1.1, call scheduler.step() will overwrite the learning rate used in optimizer to be same as scheduler sets up immediately.
    2) Also beware the scheduler.step() should be called every epoch rather than every batch. It means scheduler.step() should
        only be after and outside of "for loop of batch".
    3) Previsouly the neural network has been initialized automatically by PyTorch framework, although such initialization has not been explicitly shown. 
        So we add xavier and Kaiming initialization explicitly in this version of code.(but might not be really used since it is NOT common to use weight 
        initialization for super resolution task).
    4) Add the code to always select the weights of network which provides the best value in average SSIM over all batches for validation
        in one epoch, and save the selected weights of network and corresponding LR and SR data.
    5) When accumulate the loss in the training step, only add the value of loss into by using loss_bullet.item(), e.g. ssim_loss_training += ssim_loss.item(), rather than adding the entire
        computational graph into(e.g. ssim_loss_training += ssim_loss). Thus avoid using too much GPU memory which is not necessary.
    6) Replace the mean SSIM (a single value) by using SSIM map (a matrix) in the ssim loss.
    7) Fix "wrongly reuse the same conv for different branch" bugs in GradientMapDualResidualGroup, RCAN_Based_MRI_SR_Dual_Domain_2D, Progressive_Learning_Wrapper_MRI_SR_Dual_Domain_2D.
    
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
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from random import randrange
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
batch_size = 1
EPOCH_NUM = 1
SELECTED_BATCH_FOR_PLOT_AND_SAVE_MAT_FILE = 10
Feature_Extractor_in_Front_of_Network = False # stand for whether we use feature extractor in front of network
Maintain_Same_Size = False # stand for whether we want the output image has same size or NOT(e.g. larger size) as input image, e.g. set as Ture when apply for MRI motion artifact reduction, or applying through-plane downsampling MRI SR reconstruction.
Use_SSIM_L1_Loss = False # stand for whether we want use SSIM L1 loss in the total loss function
Use_Gradient_Map_L1_Loss = True # stand for whether we want use gradient map L1 loss in the total loss function
Use_Gram_Matrix_L1_Loss = False # stand for whether we want use gram matrix L1 loss(between SR and HR, for increasing texture similarity between SR and HR) in the total loss function
Use_Negative_TV_Loss = True # stand for whether we want to use "1/(total variation + 1.000e-10) loss"(on SR , for providing over smoothing)
Use_Negative_Trace_Loss = True # stand for whether we want to use "1/(trace(sr*hr) + 1.000e-10) loss"(on HR and SR, for increasing similarity between SR and HR)
Use_Gradient_Map_Guided_Pixel_Wise_Loss = False # stand for whether we want to use gradient map guided "attention weights" to multiply with pixel-wise loss. Can NOT be True if Use_SSIM_Map_Guided_Pixel_Wise_Loss is True
Use_SSIM_Map_Guided_Pixel_Wise_Loss = False # stand for whether we want to use SSIM map guided "attention weights" to multiply with pixel-wise loss. Can NOT be True if Use_Gradient_Map_Guided_Pixel_Wise_Loss is True
Predict_Variance_Of_Pixel_For_MRI_SR_And_Use_Uncertainty_KL_Loss = True # stand for whether we let the network model to predict the variance for each pixel and use uncertainty KL loss to minimize the variance for each pixel as well.
Use_Channel_Attention_For_Cross_Branch_Fusion = True # stand for whether we give weight for every channel of feature maps(from both image and secondary branch) before they fuse together
Amplify_Small_Value_In_Gradient_Map = False # stand for whether we want to amplify small values in gradient map to emphasize the information from gradient values which stand for texture
Amplify_High_Frequency_Value_In_K_Space_Loss = False # stand for whether we want to amplify high frequence loss values in k space loss

# If we want to plot some information
plot_the_gradient_map_of_input_image = False
plot_the_k_space_data_of_input_image = False
plot_the_wavelets_transform_data_of_input_image = False

# --------------------------- configuration of parameters for 2D_MRI_SR_Dual_Domain Reconstruct --------------------------- #
args = {'use_HR_reference' : True, 
        'use_channel_and_spatial_attention_inside_RCAB_for_HR_reference_fuser': True,
        'channel_and_spatial_attention_framework_for_HR_reference_fuser': 'external_attention',
        'channel_and_spatial_attention_mode_for_HR_reference_fuser': 'parallel_mode',

        'main_network_framework': 'RCAN', 'type_of_network': 'image_single_domain', 'long_skip_connection_to_reconstruct_residual_part_only': False,
        
        'use_channel_and_spatial_attention_inside_upsampler': False, 'use_channel_and_spatial_attention_inside_RCAB': False,
        'channel_and_spatial_attention_framework': 'external_attention', 'channel_and_spatial_attention_mode': 'sequential_mode',

        'n_colors': 1, 'n_resgroups': 5, 'n_rcablocks': 5, 'n_feats': 64, 'reduction': 16, 
        
        'scale': 2, 'number_of_progressive_stage': 1,

        'conv_layer_type': 'default_conv', 'activation_function_type': 'ReLU', 'gradient_operator': 'sobel', 
        
        'optimizer': 'Adam', 'learning_rate_decay_method': 'cosine_learning_rate_warm_restarts',
        'use_learning_rate_warm_up': False, 'how_many_epoch_to_be_used_for_warm_up': 10, 'initial_learning_rate_after_warm_up': 0.0001}

args_loss_weight = {'feature_map_weight': 20, 'pixel_wise_weight': 20000, 'k_space_weight': 2, 'ssim_weight': 100, \
                    'gradient_img_weight': 1000, 'gradient_grd_weight': 10, 'k_space_branch_weight': 0.02, \
                    'wavelets_branch_weight': 5, 'gram_similarity_weight': 5, 'negative_total_variation_weight': 3, 'negative_trace_weight': 3,\
                    'uncertainty_kl_loss_weight': 1, 'use_ssim_guided_uncertainty_kl_loss' : True,
                    'ssim_luminance_weight': 2, 'ssim_contrast_weight': 2, 'ssim_structure_weight': 4}

# args['use_HR_reference'] = True, stands for whether we select to use HR reference for MRI SR, e.g. True, False
# args['use_channel_and_spatial_attention_inside_RCAB_for_HR_reference_fuser'] = True, stands for whether we select to use attention when fusing the feature maps from HR reference and LR MRI image in the last stage, e.g. True, False
# args['channel_and_spatial_attention_framework_for_HR_reference_fuser'] = 'self_attention', stands for which channel and spatial framework is used when fusing the feature maps from HR reference and LR MRI image in the last stage, e.g. 'CBAM', 'self_attention', 'external_attention', 'gMLP'
# args['channel_and_spatial_attention_mode_for_HR_reference_fuser'] = 'parallel_mode', stands for which end to end channel and spatial block to use when fusing the feature maps from HR reference and LR MRI image in the last stage, e.g. 'sequential_mode', 'parallel_mode'

# args['main_network_framework'] = 'RCAN', stands for which main network framework to use, e.g. 'U_Net', 'RCAN'
# args['type_of_network'] == 'image_single_domain', stands for type of network, e.g. 'image_single_domain', 'gradient_map_dual_domain', 'k_space_dual_domain', 'wavelets_transform_dual_domain'
# args['long_skip_connection_to_reconstruct_residual_part_only'] == False, stands for whether we add long skip connection outside the entire network model to only reconstruct the residual part of HR MRI image, e.g. True, False

# args['use_channel_and_spatial_attention_inside_upsampler'] = True, stands for whether we use channel and spatial attention block inside upsampler, e.g. True, False
# args['use_channel_and_spatial_attention_inside_RCAB'] = True, stands for whether we use channel and spatial attention block inside RCAB to replace CALayer, e.g. True, False
# args['channel_and_spatial_attention_framework'] = 'self_attention', stands for which channel and spatial framework is used in the code, e.g. 'CBAM', 'self_attention', 'external_attention', 'gMLP'
# args['channel_and_spatial_attention_mode'] = 'sequential_mode', stands for which end to end channel and spatial block to use, e.g. 'sequential_mode', 'parallel_mode'

# args['n_colors'] = 1, stands for number of channels of input image, e.g. 1 for MRI image, 3 for RGB image.
# args['n_resgroups'] = 20, stands for number of RGs in RIR/RCAN (per stage)
# args['n_rcablocks'] = 10, stands for number of RCABs in one RG (per stage)
# args['n_feats'] = 128, stands for how many "number of channels" for feature map going through model
# args['reduction'] = 16, stands for reduction is the r mentioned in 3.3 Channel Attention in RCAN paper

# args['scale'] = 2, stands for scale factor used in one upsampler, e.g. 2, 4
# args['number_of_progressive_stage'] = 2, stands for number of stages(number of "MRI_SR_Dual_Domain_2D network"), e.g. 1, 2, 3, to ultilize progressive upsampling

# args['conv_layer_type'] = 'default_conv', stands for type of conv layer, e.g. 'default_conv', 'coord_conv', 'deformable_conv', 'py_conv', 'involution'
# args['activation_function_type'] = 'ReLU', stands for type of activation function, e.g. 'ReLU'. 'Sine', 'FReLU', 'Dynamic_ReLU_Type_A', 'Dynamic_ReLU_Type_B'
# arg['gradient_operator'] = ['sobel'] # stand for which gradient operator we want use for calculating gradient map, e.g. 'sobel', 'canny'

# arg['optimizer'] = ['Adam'] # stand for which optimizer we want use for training, e.g. 'Adam', 'SGD_with_momentum', 'look_ahead'
# arg['learning_rate_decay_method'] = ['cosine_learning_rate_decay'] # stand for which learning rate decay method we want use for training, e.g. 'cosine_learning_rate_decay', 'multi_step_learning_rate', 'step_learning_rate', 'cosine_learning_rate_warm_restarts'


if args['use_HR_reference'] == False and args['n_colors'] == 1:
    """
    The data loading pipeline for ordinary SISR MRI SR:
    """
    """""""""""""""""""""""""""""""""""""""""""""
    1.1.a. MRI HR and LR Data pair preprocessing training part
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
        if 'LR_training_2' in os.path.join(folder_log_path, idx_file):
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
        elif 'HRGT_training_2' in os.path.join(folder_log_path, idx_file):
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

    """Note for 2D matrix data with dimension H x W, everytime before loading into trainset and testset, we have to adapt the dimension into format N x C X H x W. Cause all
    the dimension of input/output are using N x C x H x W. N denotes number of data, C denotes number of channels, H means height, W stays width"""        
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
    1.2.a. Load MRI HR and LR Data pair training part
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
    2.1.a. MRI HR and LR Validation Data pair preprocessing validation part
    """""""""""""""""""""""""""""""""""""""""""""
    num_low_resolution_mat_file = 0
    num_high_resolution_groundtruth_mat_file = 0

    for idx_file in file_names:
        print(idx_file)
        if 'LR_validation_2' in os.path.join(folder_log_path, idx_file):
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
        elif 'HRGT_validation_2' in os.path.join(folder_log_path, idx_file):
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

    """Note for 2D matrix data with dimension H x W, everytime before loading into trainset and testset, we have to adapt the dimension into format N x C X H x W. Cause all
    the dimension of input/output are using N x C x H x W. N denotes number of data, C denotes number of channels, H means height, W stays width"""        
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
    2.2.a. Load MRI HR and LR Validation Data pair validation part
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

    """""""""""""""""""""""""""""""""""""""""""""
    3.1.a. MRI HR and LR Evaluation(Test) Data pair preprocessing evaluation(test) part
    """""""""""""""""""""""""""""""""""""""""""""
    # Please beware the data should be Evaluation data rather than training data
    """ folder_log_path = '/srv/DATA/RAID/HaoLi/Data/'
    file_names = os.listdir(folder_log_path) """

    num_low_resolution_mat_file = 0
    num_high_resolution_groundtruth_mat_file = 0

    for idx_file in file_names:
        print(idx_file)
        if 'LR_eval_2' in os.path.join(folder_log_path, idx_file):
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
        elif 'HRGT_eval_2' in os.path.join(folder_log_path, idx_file):
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

    """""""""""""""""""""""""""""""""""""""""""""
    3.2.a. Load MRI HR and LR Evaluation(Test) Data pair evaluation(test) part
    """""""""""""""""""""""""""""""""""""""""""""
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

elif args['use_HR_reference'] == True and args['n_colors'] == 1:
    """
    The data loading pipeline for MRI SR with HR reference:
    """
    """""""""""""""""""""""""""""""""""""""""""""
    1.1.b. MRI HR reference, HR and LR Data pair preprocessing training part
    """""""""""""""""""""""""""""""""""""""""""""
    # =============================================================================
    # h5py.version
    # =============================================================================
    "The folder where to load the training LR, HR and HR reference data pair"
    folder_log_path = 'D:/Tech_Resource/Paper_Resource/MRI SR以及相关论文/our_project_code/data/sample_downsize_training_data_HR_reference_20210111'
    file_names = os.listdir(folder_log_path)

    num_low_resolution_mat_file = 0
    num_high_resolution_groundtruth_mat_file = 0
    num_reference_mat_file = 0

    for idx_file in file_names:
        print(idx_file)
        if 'LR_training_2' in os.path.join(folder_log_path, idx_file):
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
            print(np.shape(torch_data_low_resolution))
            if num_low_resolution_mat_file == 1:
                torch_data_low_resolution_sequence = torch_data_low_resolution
            elif num_low_resolution_mat_file > 1:
                print(num_low_resolution_mat_file)
                torch_data_low_resolution_sequence = tc.cat((torch_data_low_resolution_sequence, torch_data_low_resolution), 0)
            print(np.shape(torch_data_low_resolution_sequence))
        elif 'HRGT_training_2' in os.path.join(folder_log_path, idx_file):
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
        elif 'REF_training_2' in os.path.join(folder_log_path, idx_file):
            print('One more reference image set exist')
            num_reference_mat_file = num_reference_mat_file + 1
            print(os.path.join(folder_log_path, idx_file))
            file_data_reference = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
            data_reference = file_data_reference['REF'][:] #----- numpy array
            print(np.shape(data_reference))
            torch_data_reference = tc.from_numpy(data_reference) #----- torch type data could be read by tc.utils.data.TensorDataset
            "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
            torch_data_reference = torch_data_reference.permute(0, 2, 1)
            print(np.shape(torch_data_reference))
            if num_reference_mat_file == 1:
                torch_data_reference_sequence = torch_data_reference
            if num_reference_mat_file > 1:
                print(num_reference_mat_file)
                torch_data_reference_sequence = tc.cat((torch_data_reference_sequence, torch_data_reference), 0)
            print(np.shape(torch_data_reference_sequence))
        else:
            print('other type NOT support for now')

    """Note for 2D matrix data with dimension H x W, everytime before loading into trainset and testset, we have to adapt the dimension into format N x C X H x W. Cause all
    the dimension of input/output are using N x C x H x W. N denotes number of data, C denotes number of channels, H means height, W stays width"""        
    torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.unsqueeze(1)    
    print(np.shape(torch_data_low_resolution_sequence))
    torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.unsqueeze(1)  
    print(np.shape(torch_data_high_resolution_groundtruth_sequence))
    torch_data_reference_sequence = torch_data_reference_sequence.unsqueeze(1)  
    print(np.shape(torch_data_reference_sequence))

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
    1.2.b. Load MRI HR reference, HR and LR Data pair training part
    """""""""""""""""""""""""""""""""""""""""""""
    torch_data_low_resolution_training_sequence = torch_data_low_resolution_sequence.float()
    torch_data_high_resolution_groundtruth_training_sequence = torch_data_high_resolution_groundtruth_sequence.float()
    torch_data_reference_training_sequence = torch_data_reference_sequence.float()

    # tc.multiprocessing.freeze_support()

    trainset = tc.utils.data.TensorDataset(torch_data_low_resolution_training_sequence, torch_data_high_resolution_groundtruth_training_sequence,
                                            torch_data_reference_training_sequence)

    trainloader = tc.utils.data.DataLoader(
                        trainset, 
                        batch_size = batch_size,
                        shuffle = True, 
                        num_workers = 0)

    """""""""""""""""""""""""""""""""""""""""""""
    2.1.b. MRI HR reference, HR and LR Validation Data pair preprocessing validation part
    """""""""""""""""""""""""""""""""""""""""""""
    num_low_resolution_mat_file = 0
    num_high_resolution_groundtruth_mat_file = 0
    num_reference_mat_file = 0

    for idx_file in file_names:
        print(idx_file)
        if 'LR_validation_2' in os.path.join(folder_log_path, idx_file):
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
            print(np.shape(torch_data_low_resolution))
            if num_low_resolution_mat_file == 1:
                torch_data_low_resolution_sequence = torch_data_low_resolution
            elif num_low_resolution_mat_file > 1:
                print(num_low_resolution_mat_file)
                torch_data_low_resolution_sequence = tc.cat((torch_data_low_resolution_sequence, torch_data_low_resolution), 0)
            print(np.shape(torch_data_low_resolution_sequence))
        elif 'HRGT_validation_2' in os.path.join(folder_log_path, idx_file):
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
        elif 'REF_validation_2' in os.path.join(folder_log_path, idx_file):
            print('One more referece image set exist')
            num_reference_mat_file = num_reference_mat_file + 1
            print(os.path.join(folder_log_path, idx_file))
            file_data_reference = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
            data_reference = file_data_reference['REF'][:] #----- numpy array
            print(np.shape(data_reference))
            torch_data_reference = tc.from_numpy(data_reference) #----- torch type data could be read by tc.utils.data.TensorDataset
            "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
            torch_data_reference = torch_data_reference.permute(0, 2, 1)
            print(np.shape(torch_data_reference))
            if num_reference_mat_file == 1:
                torch_data_reference_sequence = torch_data_reference
            if num_reference_mat_file > 1:
                print(num_reference_mat_file)
                torch_data_reference_sequence = tc.cat((torch_data_reference_sequence, torch_data_reference), 0)
            print(np.shape(torch_data_reference_sequence))
        else:
            print('other type NOT support for now')

    """Note for 2D matrix data with dimension H x W, everytime before loading into trainset and testset, we have to adapt the dimension into format N x C X H x W. Cause all
    the dimension of input/output are using N x C x H x W. N denotes number of data, C denotes number of channels, H means height, W stays width"""        
    torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.unsqueeze(1)    
    print(np.shape(torch_data_low_resolution_sequence))
    torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.unsqueeze(1)  
    print(np.shape(torch_data_high_resolution_groundtruth_sequence))
    torch_data_reference_sequence = torch_data_reference_sequence.unsqueeze(1)  
    print(np.shape(torch_data_reference_sequence))

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
    2.2.b. Load MRI HR reference, HR and LR Validation Data pair validation part
    """""""""""""""""""""""""""""""""""""""""""""
    torch_data_low_resolution_validation_sequence = torch_data_low_resolution_sequence.float()
    torch_data_high_resolution_groundtruth_validation_sequence = torch_data_high_resolution_groundtruth_sequence.float()
    torch_data_reference_validation_sequence = torch_data_reference_sequence.float()

    # tc.multiprocessing.freeze_support()

    validationset = tc.utils.data.TensorDataset(torch_data_low_resolution_validation_sequence, torch_data_high_resolution_groundtruth_validation_sequence, 
                                                    torch_data_reference_validation_sequence)

    validationloader = tc.utils.data.DataLoader(
                        validationset, 
                        batch_size = batch_size,
                        shuffle = False, 
                        num_workers = 0)

    """""""""""""""""""""""""""""""""""""""""""""
    3.1.b. MRI HR reference, HR and LR Evaluation(Test) Data pair preprocessing evaluation(test) part
    """""""""""""""""""""""""""""""""""""""""""""
    # Please beware the data should be Evaluation data rather than training data
    """ folder_log_path = '/srv/DATA/RAID/HaoLi/Data/'
    file_names = os.listdir(folder_log_path) """

    num_low_resolution_mat_file = 0
    num_high_resolution_groundtruth_mat_file = 0
    num_reference_mat_file = 0

    for idx_file in file_names:
        print(idx_file)
        if 'LR_eval_2' in os.path.join(folder_log_path, idx_file):
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
        elif 'HRGT_eval_2' in os.path.join(folder_log_path, idx_file):
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
        elif 'REF_eval_2' in os.path.join(folder_log_path, idx_file):
            print('One more reference image set exist')
            num_reference_mat_file = num_reference_mat_file + 1
            print(os.path.join(folder_log_path, idx_file))
            file_data_reference = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
            data_reference = file_data_reference['REF'][:] #----- numpy array
            print(np.shape(data_reference))
            torch_data_reference = tc.from_numpy(data_reference) #----- torch type data could be read by tc.utils.data.TensorDataset
            "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
            torch_data_reference = torch_data_reference.permute(0, 2, 1)
            print(np.shape(torch_data_reference))
            if num_reference_mat_file == 1:
                torch_data_reference_sequence = torch_data_reference
            if num_reference_mat_file > 1:
                print(num_reference_mat_file)
                torch_data_reference_sequence = tc.cat((torch_data_reference_sequence, torch_data_reference), 0)
            print(np.shape(torch_data_reference_sequence))
        else:
            print('other type NOT support for now')

    torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.unsqueeze(1)    
    print(np.shape(torch_data_low_resolution_sequence))
    torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.unsqueeze(1)  
    print(np.shape(torch_data_high_resolution_groundtruth_sequence))
    torch_data_reference_sequence = torch_data_reference_sequence.unsqueeze(1)  
    print(np.shape(torch_data_reference_sequence))

    """""""""""""""""""""""""""""""""""""""""""""
    3.2.b. Load MRI HR reference, HR and LR Evaluation(Test) Data pair evaluation(test) part
    """""""""""""""""""""""""""""""""""""""""""""
    torch_data_low_resolution_eval_sequence = torch_data_low_resolution_sequence.float()
    torch_data_high_resolution_groundtruth_eval_sequence = torch_data_high_resolution_groundtruth_sequence.float()
    torch_data_reference_eval_sequence = torch_data_reference_sequence.float()

    testset = tc.utils.data.TensorDataset(torch_data_low_resolution_eval_sequence, torch_data_high_resolution_groundtruth_eval_sequence,
                                            torch_data_reference_eval_sequence)

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

elif args['use_HR_reference'] == False and args['n_colors'] == 3:
    """
    The data loading pipeline for ordinary SISR RGB Visual Image SR:
    """
    # @TODO: The corresponding data loading part to be added 
    pass

elif args['use_HR_reference'] == True and args['n_colors'] == 3:
    """
    The data loading pipeline for RGB Visual Image SR with HR reference:
    """
    # @TODO: The corresponding data loading part to be added 
    pass
else:
    raise ValueError("Not supported type of input image in loading input data!")



 

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


"""
Negative Total Variation Loss(TV loss). negative_tv_loss = 1 - TV.
Minimize总变差（TV）loss促进了生成的图像中的空间平滑性。于是minimize Negative Total Variation Loss将防止图像过分平滑。
See more information regarding TV Loss from paper: 2015.iSeeBetter: Spatio-temporal video super-resolution using recurrent generative back-projection networks
"""
class NegativeTVLoss(nn.Module):
    def __init__(self, negative_tv_loss_weight = 1, tv_loss_weight = 1):
        super(NegativeTVLoss, self).__init__()
        self.negative_tv_loss_weight = negative_tv_loss_weight
        self.tv_loss = TVLoss(tv_loss_weight)
    
    def forward(self, x):
        return self.negative_tv_loss_weight * (1 - self.tv_loss(x))

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = tc.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = tc.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]


"""
Negative Trace Loss. Negative_Trace_Loss = 1/(trace(SR*HR) + 1.000e-10).
trace(SR*HR)表示SR和HR的相似程度。两个向量内积是把一个向量投影到另一个上的长度，这个值可以用于描述两个向量的相似性。两个矩阵A、B的相似性
可以用A、B两个矩阵的内积表征，被定义为Trace(AB)。于是minimize Negative_Trace_Loss可以最大化相似两个矩阵。
见paper: 2015.LRTV: MR Image Super-Resolution With Low-Rank and Total Variation Regularizations
"""
class NegativeTraceLoss(nn.Module):
    def __init__(self, negative_trace_loss_weight = 1):
        super(NegativeTraceLoss, self).__init__()
        self.negative_trace_loss_weight = negative_trace_loss_weight
    
    def forward(self, SR, HR):
        total_loss = 0.00
        for i in range(SR.size(0)):
            total_loss = total_loss + 1 / (tc.trace(SR[i, 0, :, :]*HR[i, 0, :, :]) + 1.000e-10)
        return self.negative_trace_loss_weight * total_loss/SR.size(0)


"""
Gradient Map Guided Weight For Pixel Wise Loss.
SR和HR分别求gradient map，再相减得到一个gradient map差的矩阵，再把这个gradient map差的矩阵从(H * W)变为(1 * HW)，然后再过一个softmax，再变回H * W，
然后把得到的矩阵当做pixel-wise L1 loss的weight来元素乘在L1 loss的pixel上。
"""
class GradientMapGuidedWeightForPixelWiseLoss(nn.Module):
    def __init__(self):
        super(GradientMapGuidedWeightForPixelWiseLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 3)
    
    def forward(self, SR, HR):
        gradient_map_difference_matrix = calculate_gradient_map(HR) - calculate_gradient_map(SR)
        N, C, H, W = gradient_map_difference_matrix.size(0), gradient_map_difference_matrix.size(1), gradient_map_difference_matrix.size(2), gradient_map_difference_matrix.size(3)
        gradient_map_difference_matrix = gradient_map_difference_matrix.reshape(N, C, 1, H*W)
        gradient_map_difference_weight_matrix = self.softmax(gradient_map_difference_matrix)
        gradient_map_difference_weight_matrix = gradient_map_difference_weight_matrix.reshape(N, C, H, W)
        return gradient_map_difference_weight_matrix


"""
SSIM Map Guided Weight For Pixel Wise Loss.
SR和HR求SSIM map，再用1减这个SSIM map得到一个矩阵当做pixel-wise L1 loss的weight来元素乘在L1 loss的pixel上。
"""
class SSIMMapGuidedWeightForPixelWiseLoss(nn.Module):
    def __init__(self):
        super(SSIMMapGuidedWeightForPixelWiseLoss, self).__init__()
    
    def forward(self, SR, HR):
        ssim_map_weighted, ssim_map = pytorch_ssim_map.ssim_map(SR, HR, luminance_weight = 1, contrast_weight = 1, structure_weight = 1)
        one_minus_ssim_map_weight_matrix = 1 - ssim_map_weighted
        return one_minus_ssim_map_weight_matrix


"""
Uncertainty KL Loss. 对HR的每一个像素的ground truth值都当做dirac分布，然后把RCAN网络输出部分输出两个变量，一个是每个像素的均值，另一个是每个像素的方差。
然后通过minimize KL散度的方式得到一个MSE loss的变形，用这个loss来训练网络从而可以预测每个像素的方差。
见论文：2019.Bounding Box Regression with Uncertainty for Accurate Object Detection公式(9),(10). 
另外，对这个方案，我们可以考虑不对每一个SR image的pixel都求variance，而是只对当前SR image中那些SSIM Map中值小于一定threshold的pixel求variance。
这个threhold可以设为当前SSIM map中所有元素的均值减去一倍(68%置信区间)或者二倍(95%置信区间)的方差。
"""
class UncertaintyKlLoss(nn.Module):
    def __init__(self, uncertainty_kl_loss_weight = 1, use_ssim_guided_uncertainty_kl_loss = True):
        super(UncertaintyKlLoss, self).__init__()
        self.uncertainty_kl_loss_weight = uncertainty_kl_loss_weight
        self.use_ssim_guided_uncertainty_kl_loss = use_ssim_guided_uncertainty_kl_loss
    
    def forward(self, SR, log_of_variance_of_SR, HR, ssim_map):
        if self.use_ssim_guided_uncertainty_kl_loss == False:
            uncertainty_kl_loss = self.uncertainty_kl_loss_weight * tc.sum(tc.exp(-log_of_variance_of_SR) * tc.square(HR - SR) + 0.5 * log_of_variance_of_SR)
        else:   # self.use_ssim_guided_uncertainty_kl_loss == True:
            selection_matrix = tc.zeros_like(ssim_map)
            threshold = tc.mean(ssim_map, dim = (2, 3)) - tc.std(ssim_map, dim = (2, 3))
            selection_matrix[ssim_map < threshold] = 1
            uncertainty_kl_loss = self.uncertainty_kl_loss_weight * tc.sum( selection_matrix * (tc.exp(-log_of_variance_of_SR) * tc.square(HR - SR) + 0.5 * log_of_variance_of_SR) )
        return uncertainty_kl_loss


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


"Involution Conv Layer"
"""
2021.Involution: Inverting the Inherence of Convolution for Visual Recognition
"""
class Involution(nn.Module):
    """
    Implementation of `Involution: Inverting the Inherence of Convolution for Visual Recognition`.
    """
    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3, stride=1, reduction_ratio=2):
        super().__init__()
        channels_reduced = max(1, in_channels // reduction_ratio)
        padding = kernel_size // 2
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, channels_reduced, 1),
            nn.BatchNorm2d(channels_reduced),
            nn.ReLU(inplace=True))
        self.span = nn.Conv2d(channels_reduced, kernel_size * kernel_size * groups, 1)
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)
        self.resampling = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

    @classmethod
    def get_name(cls):
        """
        Return this layer name.
        Returns:
            str: layer name.
        """
        return 'Involution'

    def forward(self, input_tensor):
        """
        Calculate Involution.
        override function from PyTorch.
        """
        _, _, height, width = input_tensor.size()
        if self.stride > 1:
            out_size = lambda x: (x + 2 * self.padding - self.kernel_size) // self.stride + 1
            height, width = out_size(height), out_size(width)
        uf_x = rearrange(self.unfold(input_tensor), 'b (g d k j) (h w) -> b g d (k j) h w',
                         g=self.groups, k=self.kernel_size, j=self.kernel_size, h=height, w=width)
        if self.stride > 1:
            input_tensor = F.adaptive_avg_pool2d(input_tensor, (height, width))
        kernel = rearrange(self.span(self.reduce(input_tensor)), 'b (k j g) h w -> b g (k j) h w',
                           k=self.kernel_size, j=self.kernel_size)
        out = rearrange(tc.einsum('bgdxhw, bgxhw -> bgdhw', uf_x, kernel), 'b g d h w -> b (g d) h w')
        if self.resampling:
            out = self.resampling(out)
        return out.contiguous()

def involution(in_channels, out_channels, kernel_size, bias = True):
    return Involution(
        in_channels, out_channels, groups=1, kernel_size = kernel_size,
        stride=1, reduction_ratio=2)


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
        k_space_result = tc.fft.fftn(x, dim = (-3,-2,-1))
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
    def __init__(self, in_channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_channel, in_channel // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_channel // reduction, in_channel, 1, bias=False)
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

"End to End CAM Channel and SAM Spatial Attention Block, Either Sequential or Parallel for Channel and Spatial Attention"
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
            raise ValueError("Not supported channel and spatial attention mode yet! Please select 'sequential_mode' or 'parallel_mode'")



"Non-Local Self-Attention based Channel Attention, is another approach to calculate channel attention. See more in figure 3.B of DANet paper: 2018.Dual Attention Network for Scene Segmentation"
class SelfAttentionBasedChannelAttention(nn.Module):
    """
    Non-local self-attention based channel attention.
    """
    def __init__(self, in_channel):
        super(SelfAttentionBasedChannelAttention, self).__init__()
        self.conv_1x1_for_v = nn.Conv2d(in_channel, in_channel, kernel_size = 1, bias=False)
        self.conv_1x1_for_k = nn.Conv2d(in_channel, in_channel, kernel_size = 1, bias=False)
        self.conv_1x1_for_q = nn.Conv2d(in_channel, in_channel, kernel_size = 1, bias=False)
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, x):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        input_feature_map = x
        value = self.conv_1x1_for_v(x).reshape(N, C, H*W)   # Reshape input data from (N, C, H, W) to (N, C, (H*W))
        key = self.conv_1x1_for_k(x).reshape(N, C, H*W)  # Shape of key is (N, C, (H*W))
        query = self.conv_1x1_for_q(x).reshape(N, C, H*W).permute(0, 2, 1) # Transpose the data from (N, C, (H*W)) to (N, (H*W), C) for query
        attention_map = tc.matmul(key, query)   # Shape of attention_map is (N, C, C)
        attention_map = self.softmax(attention_map)     # Shape of attention_map is (N, C, C)
        attention_feature_map =  tc.matmul(attention_map, value)    # Shape of attention_feature_map is (N, C, (H*W))
        attention_feature_map = attention_feature_map.reshape(N, C, H, W)   # Shape of attention_feature_map is (N, C, H, W)
        return input_feature_map + attention_feature_map

"Non-Local Self-Attention based Spatial Attention, is another approach to calculate spatial attention. See more in figure 2 of paper: 2018.Non-local Neural Networks. or figure 3.A of DANet paper: 2018.Dual Attention Network for Scene Segmentation"
class SelfAttentionBasedSpatialAttention(nn.Module):
    """
    Non-local self-attention based spatial attention is originally from paper: 2018.Non-local Neural Networks which borrows from the self-attention mechanism in the classical paper in NLP 
    which proposes the Transformer technology: 2017.Attention is All You Needed.
    """
    def __init__(self, in_channel):
        super(SelfAttentionBasedSpatialAttention, self).__init__()
        self.conv_1x1_for_v = nn.Conv2d(in_channel, in_channel, kernel_size = 1, bias=False)
        self.conv_1x1_for_k = nn.Conv2d(in_channel, in_channel, kernel_size = 1, bias=False)
        self.conv_1x1_for_q = nn.Conv2d(in_channel, in_channel, kernel_size = 1, bias=False)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        input_feature_map = x
        value = self.conv_1x1_for_v(x).reshape(N, C, H*W)   # Reshape input data from (N, C, H, W) to (N, C, (H*W)) for value
        key = self.conv_1x1_for_k(x).reshape(N, C, H*W)  # Shape of key is (N, C, (H*W))
        query = self.conv_1x1_for_q(x).reshape(N, C, H*W).permute(0, 2, 1) # Transpose the data from (N, C, (H*W)) to (N, (H*W), C) for query
        attention_map = tc.matmul(query, key)   # Shape of attention_map is (N, (H*W), (H*W))
        attention_map = self.softmax(attention_map)     # Shape of attention_map is (N, (H*W), (H*W))
        attention_feature_map =  tc.matmul(value, attention_map)    # Shape of attention_feature_map is (N, C, (H*W))
        attention_feature_map = attention_feature_map.reshape(N, C, H, W)   # Shape of attention_feature_map is (N, C, H, W)
        return input_feature_map + attention_feature_map

"Self Attention based End to End Channel and Spatial Attention Block, Either Sequential or Parallel for Channel and Spatial Attention"
class SelfAttentionBasedChannelAndSpatialAttention(nn.Module):
    def __init__(self, in_channel, channel_and_spatial_attention_mode = 'sequential_mode'):
        super(SelfAttentionBasedChannelAndSpatialAttention, self).__init__()
        self.self_attention_channel_attention = SelfAttentionBasedChannelAttention(in_channel)
        self.self_attention_spatial_attention = SelfAttentionBasedSpatialAttention(in_channel)
        self.self_attention_based_channel_and_spatial_attention_mode = channel_and_spatial_attention_mode
    
    def forward(self, x):
        if self.self_attention_based_channel_and_spatial_attention_mode == 'sequential_mode':
            x = self.self_attention_channel_attention(x)
            x = self.self_attention_spatial_attention(x)
            return x
        elif self.self_attention_based_channel_and_spatial_attention_mode == 'parallel_mode':
            channel_attention_output = self.self_attention_channel_attention(x)
            spatial_attention_output = self.self_attention_spatial_attention(x)
            return channel_attention_output + spatial_attention_output
        else:
            raise ValueError("Not supported channel and spatial attention mode yet")



"External-Attention based Spatial Attention, is a pure MLP based 'self-attention'. See paper: '2021.Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks' for more detail."
class ExternalAttentionBasedSpatialAttention(nn.Module):
    """
    See paper: '2021.Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks' 
            and https://mp.weixin.qq.com/s/L75TsIkB0h_GxQ6fDiVKOQ
            for more detail.
    """
    def __init__(self, in_channel):
        super(ExternalAttentionBasedSpatialAttention, self).__init__()
        self.k = 32 
        self.first_conv = nn.Conv2d(in_channel, in_channel, 1)
        self.k_linear = nn.Conv1d(in_channel, self.k, 1, bias=False)    # MLP_1
        self.v_linear = nn.Conv1d(self.k, in_channel, 1, bias=False)    # MLP_2
        self.softmax = nn.Softmax(dim = -1)     # Normalization over space H*W for spatial-wise usage.

    def forward(self, x):
        idn = x[:]
        b, c, h, w = x.size()
        x = self.first_conv(x)
        x = x.view(b, c, -1) # b, c, n 
        attn = self.k_linear(x) # b, c, n
        attn = self.softmax(attn)
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-9)    # Normalization over channel C.
        x = self.v_linear(attn) # b, c, n 
        x = x.view(b, c, h, w)
        x = x + idn 
        return x

"External-Attention based Channel Attention, is an alternative external-attention but with softmax over channel C(NOT over space H*W). "
class ExternalAttentionBasedChannelAttention(nn.Module):
    """
    Almost same as External-Attention, but softmax over channel C and normalization over space H*W.
    """
    def __init__(self, in_channel):
        super(ExternalAttentionBasedChannelAttention, self).__init__()
        self.k = 32 
        self.first_conv = nn.Conv2d(in_channel, in_channel, 1)
        self.k_linear = nn.Conv1d(in_channel, self.k, 1, bias=False)    # MLP_1
        self.v_linear = nn.Conv1d(self.k, in_channel, 1, bias=False)    # MLP_2
        self.softmax = nn.Softmax(dim = -2)     # Softmax over space C for channel-wise usage.

    def forward(self, x):
        idn = x[:]
        b, c, h, w = x.size()
        x = self.first_conv(x)
        x = x.view(b, c, -1) # b, c, n 
        attn = self.k_linear(x) # b, c, n
        attn = self.softmax(attn)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)   # Normalization over space H*W.
        x = self.v_linear(attn) # b, c, n 
        x = x.view(b, c, h, w)
        x = x + idn 
        return x

"External-Attention based End to End Channel and Spatial Attention Block, Either Sequential or Parallel for Channel and Spatial Attention"
class ExternalAttentionBasedChannelAndSpatialAttention(nn.Module):
    def __init__(self, in_channel, channel_and_spatial_attention_mode = 'sequential_mode'):
        super(ExternalAttentionBasedChannelAndSpatialAttention, self).__init__()
        self.external_attention_channel_attention = ExternalAttentionBasedChannelAttention(in_channel)
        self.external_attention_spatial_attention = ExternalAttentionBasedSpatialAttention(in_channel)
        self.external_attention_based_channel_and_spatial_attention_mode = channel_and_spatial_attention_mode
    
    def forward(self, x):
        if self.external_attention_based_channel_and_spatial_attention_mode == 'sequential_mode':
            x = self.external_attention_channel_attention(x)
            x = self.external_attention_spatial_attention(x)
            return x
        elif self.external_attention_based_channel_and_spatial_attention_mode == 'parallel_mode':
            channel_attention_output = self.external_attention_channel_attention(x)
            spatial_attention_output = self.external_attention_spatial_attention(x)
            return channel_attention_output + spatial_attention_output
        else:
            raise ValueError("Not supported channel and spatial attention mode yet")



"gMLP or aMLP, which is another 'pure MLP' or 'pure MLP with tiny attention' module. See paper: '2021.Pay Attention to MLPs' for more info."
class gMLPVision(nn.Module):
    def __init__(
        self,
        *,
        image_size = 32,
        patch_size = 1,
        dim = 512,          # Input feature's patch embedding dimension. See figure 1 in paper: 2021.Pay Attention to MLPs.
        depth = 3,          # Number of gMLP layers, L. See figure 1 in paper: 2021.Pay Attention to MLPs.
        ff_mult = 4,
        channels = 3,
        attn_dim = None,    # For tiny attention using.
        prob_survival = 1.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'image size must be divisible by the patch size'
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        dim_ff = dim * ff_mult      # Hidden dimension?
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding: Generate N tokens, where N is the num_patches, equals to h*w/(patch_size**2). Each token is a vector with dimension (1, dim).
        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(channels * patch_size ** 2, dim)
        )   # Size of the output from this module is (batch_size, num_patches, dim).

        self.prob_survival = prob_survival

        self.layers = nn.ModuleList([ResidualForGmlp(nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_ff * 2),
            nn.GELU(),
            SpatialGatingUnit(dim_ff, num_patches, attn_dim),
            nn.Linear(dim_ff, dim)
        )) for i in range(depth)])
        
    def feature_mapping(self, x):
        x = x.view(
            x.size(0),
            int(self.image_size / self.patch_size),
            int(self.image_size / self.patch_size),
            self.dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        x = self.to_patch_embed(x)      # (B, C, H, W) --> (B, N, dim) where N equal to H*W/(patch_size**2)
        """ layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival) """
        layers = self.layers
        x = nn.Sequential(*layers)(x)   # (B, N, dim) --> (B, N, dim)
        x = self.feature_mapping(x)     # (B, N, dim) --> (B, dim, H/patch_size, W/patch_size)
        return x

# functions
def exists(val):
    return val is not None

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = tc.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# helper classes
class ResidualForGmlp(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class AttentionForGmlp(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = tc.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = tc.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -tc.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = tc.einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, attn_dim = None, causal = False):
        super().__init__()
        self.causal = causal

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)
        self.attn = AttentionForGmlp(dim * 2, dim, attn_dim, causal) if exists(attn_dim) else None
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x):
        device = x.device

        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            mask = tc.ones(weight.shape[:2], device = device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.)

        gate = F.conv1d(gate, weight, bias)

        if exists(self.attn):
            gate += self.attn(x)
        return gate * res



"Residual Channel Attention Block (RCAB)"
class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB): There are several RCABs belong to one RG(Residual Group).
    See figure 4 of original RCAN paper
    """
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, use_channel_and_spatial_attention_inside_RCAB = False, 
        channel_and_spatial_attention_framework = 'CBAM', channel_and_spatial_attention_mode = 'sequential_mode'):

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
        if use_channel_and_spatial_attention_inside_RCAB == False:
            modules_body.append(CALayer(n_feat, reduction)) # use CA Layer
        else: # use_channel_and_spatial_attention_inside_RCAB == True
            if channel_and_spatial_attention_framework == 'CBAM':
                modules_body.append(ChannelAndSpatialAttention(in_channel = n_feat, reduction = reduction, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode)) # use CBAM attention block
            elif channel_and_spatial_attention_framework == 'self_attention':
                modules_body.append(SelfAttentionBasedChannelAndSpatialAttention(in_channel = n_feat, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode)) # use self_attention based channel and spatial attention block
            elif channel_and_spatial_attention_framework == 'external_attention':
                modules_body.append(ExternalAttentionBasedChannelAndSpatialAttention(in_channel = n_feat, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode)) # use external_attention based channel and spatial attention block
            elif channel_and_spatial_attention_framework == 'gMLP':
                modules_body.append(gMLPVision(channels = n_feat, attn_dim = 2)) # use gMLP based channel and spatial attention block
            else:
                raise ValueError("Not supported channel and spatial attention framework! Please select either 'CBAM' or 'self_attention'")

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) # conv --> ReLU --> conv --> CA/channel and spatial attention block
        #res = self.body(x).mul(self.res_scale)
        res += x # local skip link of RCAB
        return res


"Residual Group (RG)"
class ResidualGroup(nn.Module):
    """
    RG(Residual Group): There are several RGs belong to one RIR(Residual in Residual module).
    See upper figure in figure 2 of original RCAN paper
    """
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_rcablocks, use_channel_and_spatial_attention_inside_RCAB = False, 
                        channel_and_spatial_attention_framework = 'CBAM', channel_and_spatial_attention_mode = 'sequential_mode'):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=1, 
                use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB,
                channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                channel_and_spatial_attention_mode = channel_and_spatial_attention_mode) \
            for _ in range(n_rcablocks)]

        modules_body.append(conv(n_feat, n_feat, kernel_size)) # last conv after several RCABs as show in figure 2
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) # RCAB_1 --> RCAB_2 --> ... --> RCAB_(n_rcablocks) --> conv
        res += x # short skip connection in RG
        return res


"Dual Domain Gradient Map Residual Group"
class GradientMapDualResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_rcablocks,
                        use_channel_and_spatial_attention_inside_RCAB = False, 
                        channel_and_spatial_attention_framework = 'CBAM', channel_and_spatial_attention_mode = 'sequential_mode'):
        super(GradientMapDualResidualGroup, self).__init__()
        modules_body = [ResidualGroup(conv, n_feat, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                        use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB, 
                        channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                        channel_and_spatial_attention_mode = channel_and_spatial_attention_mode
                        )]
        """ modules_body.append(conv(n_feat, n_feat, kernel_size)) """
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            modules_channel_attention_for_cross_branch_fusion = [CALayer(n_feat*2, reduction)]
        modules_connection = [(conv(2*n_feat, n_feat, kernel_size = 1, bias=True))]

        self.body_image_branch = nn.Sequential(*modules_body)
        self.body_gradientmap_branch = nn.Sequential(*modules_body)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            self.channel_attention_for_cross_branch_fusion_in_image_branch = nn.Sequential(*modules_channel_attention_for_cross_branch_fusion)
            self.channel_attention_for_cross_branch_fusion_in_gradientmap_branch = nn.Sequential(*modules_channel_attention_for_cross_branch_fusion)
        self.connection_in_image_branch = nn.Sequential(*modules_connection)
        self.connection_in_gradientmap_branch = nn.Sequential(*modules_connection)

    def forward(self, x):
        x1 = x[0,:,:,:,:] # Extract image branch data
        x2 = x[1,:,:,:,:] # Extract gradientmap branch data
        x1 = x1.squeeze(0)
        x2 = x2.squeeze(0)

        res1 = self.body_image_branch(x1) # go through several RCABs in image branch
        res2 = self.body_gradientmap_branch(x2) # go through several RCABs in gradientmap branch

        # fuse intermedian results in both branches into image branch
        res_image_branch = tc.cat((res1, res2), 1)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            res_image_branch = self.channel_attention_for_cross_branch_fusion_in_image_branch(res_image_branch) # set up channel attention for both image and gradeint branch when theu fuse into one feature map
            """ print("channel attention for fusion:res_image_branch") """
        res_image_branch = self.connection_in_image_branch(res_image_branch)

        # fuse intermedian results in both branches into gradientmap branch
        res_gradientmap_branch = tc.cat((res1, res2), 1)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            res_gradientmap_branch = self.channel_attention_for_cross_branch_fusion_in_gradientmap_branch(res_gradientmap_branch) # set up channel attention for both image and gradeint branch when theu fuse into one feature map
            """ print("channel attention for fusion:res_gradientmap_branch") """
        res_gradientmap_branch = self.connection_in_gradientmap_branch(res_gradientmap_branch)

        res_image_branch = res_image_branch.unsqueeze(0)
        res_gradientmap_branch = res_gradientmap_branch.unsqueeze(0)
        res = tc.cat((res_image_branch, res_gradientmap_branch), 0)
        return res


"Dual Domain K Space Residual Group"
class KSpaceDualResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_rcablocks,
                        use_channel_and_spatial_attention_inside_RCAB = False, 
                        channel_and_spatial_attention_framework = 'CBAM', channel_and_spatial_attention_mode = 'sequential_mode'):
        super(KSpaceDualResidualGroup, self).__init__()
        modules_body_time_domain = [ResidualGroup(conv, n_feat, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                                                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB,
                                                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework,
                                                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode)]
        modules_body_frequency_domain = [ResidualGroup(conv, 2*n_feat, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                                                        use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB,
                                                        channel_and_spatial_attention_framework = channel_and_spatial_attention_framework,
                                                        channel_and_spatial_attention_mode = channel_and_spatial_attention_mode)]
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
            """ print("channel attention for fusion:res_image_branch") """
        res_image_branch = self.connection_in_image_branch(res_image_branch) # shape of res_image_branch is (N, C, H, W)

        # fuse intermedian results in both branches into k space branch
        res1 = self.fft_operation(res1) # Now shape of res1 is (N, C, H, W, 2)
        res1 = tc.cat((res1[:, :, :, :, 0], res1[:, :, :, :, 1]), 1) # Now shape of res1 is (N, 2*C, H, W)
        res_k_space_branch = tc.cat((res1, res2), 1) # shape of res_k_space_branch is (N, 4*C, H, W)
        if Use_Channel_Attention_For_Cross_Branch_Fusion == True:
            res_k_space_branch = self.channel_attention_for_cross_branch_fusion_in_k_space_branch(res_k_space_branch) # set up channel attention for both image and gradeint branch when theu fuse into one feature map
            """ print("channel attention for fusion:res_k_space_branch") """
        res_k_space_branch = self.connection_in_k_space_branch(res_k_space_branch) # shape of res_k_space_branch is (N, 2*C, H, W)

        res_image_branch = res_image_branch.unsqueeze(0) # shape of res_image_branch is (1, N, C, H, W)
        res_image_branch = tc.cat((res_image_branch, res_image_branch), 2) # append data into shape (1, N, 2*C, H, W)
        res_k_space_branch = res_k_space_branch.unsqueeze(0) # shape of res_k_space_branch is (1, N, 2*C, H, W)
        res = tc.cat((res_image_branch, res_k_space_branch), 0) # shape of res is (2, N, 2*C, H, W)
        return res


"Dual Domain Wavelets Transform Residual Group"
class WaveletsTransformDualResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_rcablocks, use_channel_and_spatial_attention_inside_RCAB = False, 
                        channel_and_spatial_attention_framework = 'CBAM', channel_and_spatial_attention_mode = 'sequential_mode'):
        super(WaveletsTransformDualResidualGroup, self).__init__()
        modules_body_time_domain = [ResidualGroup(conv, n_feat, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                                                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB,
                                                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework,
                                                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode)]
        modules_body_wavelets_high_frequency_components_domain = [ResidualGroup(conv, 3*n_feat, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                                                                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB,
                                                                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework,
                                                                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode)]
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


"Upsampler Module, implemented by employeed of sub-pixel conv"
class Upsampler(nn.Sequential):
    """
    Upsampling/Upscale module, used as last part of "SR reconstruction network model" if the network model employ the "post-upsampling mode".
    Beware the actual upsampling approach is "sub-pixel conv" (which is nn.PixelShuffle() in Pytorch) which was proposed in
    paper: "2016. Real-Time single image and video super-resolution using an efficient sub-pixel convolutional neural network".
    Such sub-pixel conv actually constructs F ∗ S^2 feature maps of dimensions H ×W are reshaped into F feature maps of dimensions H ∗ S × W ∗ S, 
    where S is the upsampling factor.
    """
    def __init__(self, conv, scale, n_feats, use_channel_and_spatial_attention_inside_upsampler = False, 
        channel_and_spatial_attention_framework = 'CBAM', channel_and_spatial_attention_mode = 'sequential_mode'):
        super(Upsampler, self).__init__()
        if scale == 2:
            if use_channel_and_spatial_attention_inside_upsampler == False:
                self.upsampler = nn.Sequential(*[
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                    nn.PixelShuffle(scale)
                ])
            else: # use_channel_and_spatial_attention_inside_upsampler == True
                if channel_and_spatial_attention_framework == 'CBAM':
                    self.upsampler = nn.Sequential(*[
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        ChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                        nn.PixelShuffle(scale)
                    ])
                elif channel_and_spatial_attention_framework == 'self_attention':
                    self.upsampler = nn.Sequential(*[
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        SelfAttentionBasedChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                        nn.PixelShuffle(scale)
                    ])
                elif channel_and_spatial_attention_framework == 'external_attention':
                    self.upsampler = nn.Sequential(*[
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        ExternalAttentionBasedChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                        nn.PixelShuffle(scale)
                    ])
                elif channel_and_spatial_attention_framework == 'gMLP':
                    self.upsampler = nn.Sequential(*[
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        gMLPVision(channels = n_feats * 4, attn_dim = 2),
                        nn.PixelShuffle(scale)
                    ])
                else:
                    raise ValueError("Not supported channel and spatial attention framework! Please select either 'CBAM' or 'self_attention'")
        elif scale == 4:
            if use_channel_and_spatial_attention_inside_upsampler == False:
                self.upsampler = nn.Sequential(*[
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                    nn.PixelShuffle(2),
                ])
            else: # use_channel_and_spatial_attention_inside_upsampler == True
                if channel_and_spatial_attention_framework == 'CBAM':
                    self.upsampler = nn.Sequential(*[
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        ChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                        nn.PixelShuffle(2),
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        ChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                        nn.PixelShuffle(2),
                    ])
                elif channel_and_spatial_attention_framework == 'self_attention':
                    self.upsampler = nn.Sequential(*[
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        SelfAttentionBasedChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                        nn.PixelShuffle(2),
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        SelfAttentionBasedChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                        nn.PixelShuffle(2),
                    ])
                elif channel_and_spatial_attention_framework == 'external_attention':
                    self.upsampler = nn.Sequential(*[
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        ExternalAttentionBasedChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                        nn.PixelShuffle(2),
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        ExternalAttentionBasedChannelAndSpatialAttention(in_channel = n_feats * 4, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                        nn.PixelShuffle(2),
                    ])
                elif channel_and_spatial_attention_framework == 'gMLP':
                    self.upsampler = nn.Sequential(*[
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        gMLPVision(channels = n_feats * 4, attn_dim = 2),
                        nn.PixelShuffle(2),
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                        gMLPVision(channels = n_feats * 4, attn_dim = 2),
                        nn.PixelShuffle(2),
                    ])
                else:
                    raise ValueError("Not supported channel and spatial attention framework! Please select either 'CBAM' or 'self_attention'")
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
    def __init__(self, args, not_use_last_conv_to_change_num_channels_to_n_colors = False):
        super(RCAN_Based_MRI_SR_Dual_Domain_2D, self).__init__()
        if args['type_of_network'] == 'image_single_domain':
            self.type_of_network = 'image_single_domain'
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
        elif args['conv_layer_type'] == 'involution':
            conv = involution

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

        self.not_apply_last_conv_to_change_num_of_channels_to_n_colors = args['use_HR_reference'] and not_use_last_conv_to_change_num_channels_to_n_colors
        n_resgroups = args['n_resgroups'] # number of RGs in RIR/RCAN
        n_rcablocks = args['n_rcablocks'] # number of RCABs in one RG
        n_colors = args['n_colors'] # number of channels going of input of entire model
        n_feats = args['n_feats'] # number of feature maps/channels going through the entire model
        kernel_size = 3 # conv filter size used for all conv in RCAN
        reduction = args['reduction'] # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        scale = args['scale'] # resize factor, e.g. 2, 4
        use_channel_and_spatial_attention_inside_upsampler = args['use_channel_and_spatial_attention_inside_upsampler'] # whether we use channel and spatial attention block inside upsampler, e.g. True, False
        use_channel_and_spatial_attention_inside_RCAB = args['use_channel_and_spatial_attention_inside_RCAB']   #  whether we use channel and spatial attention block inside RCAB to replace CALayer, e.g. True, False
        channel_and_spatial_attention_framework = args['channel_and_spatial_attention_framework']   # which channel and spatial framework is used in the code, e.g. 'CBAM', 'self_attention', 'external_attention', 'gMLP'
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
        if self.type_of_network == 'image_single_domain':
            # The RIR(Residual in Residual) which consists of n_resgroups RGs, show in figure 2 of RCAN paper
            modules_body = [
                ResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB, 
                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode) \
                for _ in range(n_resgroups)]
        elif self.type_of_network == 'gradient_map_dual_domain':
            modules_body = [
                GradientMapDualResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB, 
                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode) \
                for _ in range(n_resgroups)]
            modules_head_for_gradient_map_branch = [conv(n_colors, n_feats, kernel_size)]
            modules_pretail_for_gradient_map_branch = [conv(n_feats, n_feats, kernel_size)]
            if self.not_apply_last_conv_to_change_num_of_channels_to_n_colors == True:
                modules_tail_for_gradient_map_branch = [
                Upsampler(conv, scale, n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                                channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode)]
            else:
                modules_tail_for_gradient_map_branch = [
                Upsampler(conv, scale, n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                                channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                conv(n_feats, n_colors, kernel_size)]
        elif self.type_of_network == 'k_space_dual_domain':
            modules_body = [
                KSpaceDualResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB, 
                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode) \
                for _ in range(n_resgroups)]
            modules_head_for_k_space_branch = [conv(2*n_colors, 2*n_feats, kernel_size)]
            modules_pretail_for_k_space_branch = [conv(2*n_feats, 2*n_feats, kernel_size)]
            modules_tail_for_k_space_branch = [
            Upsampler(conv, scale, 2*n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                        channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
            conv(2*n_feats, 2*n_colors, kernel_size)]
        elif self.type_of_network == 'wavelets_transform_dual_domain':
            modules_body = [
                WaveletsTransformDualResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB, 
                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode) \
                for _ in range(n_resgroups)]

        # define tail module. The last stage is upsampling module and one more conv layer, show in figure 2 of RCAN paper
        if self.not_apply_last_conv_to_change_num_of_channels_to_n_colors == True:
            modules_tail = [
            Upsampler(conv, scale, n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                            channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode)]
        else:
            modules_tail = [
                Upsampler(conv, scale, n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                                channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                conv(n_feats, n_colors, kernel_size)]

        # Add a downsize converter by using conv layer.
        # The purpose of original RCAN designed in original RCAN paper is to upscale scale times of LR image, the output from RIR has same size as input LR image. However, here in our task
        # we actually want to maintain output(SR) and input(LR) same as, it means we need to downsize the image before leave it into upscale module. That is the reason we need extra conv layer to perform 
        # down size with scale time first before going through upscale with scale time
        if (scale == 2):
            # W2=(W1−F+2P)/S+1, H2=(H1−F+2P)/S+1.
            self.down_size_converter = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
            if self.type_of_network == 'gradient_map_dual_domain':
                self.down_size_converter_for_gradient_map_branch = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
            elif self.type_of_network == 'k_space_dual_domain':
                self.down_size_converter_for_k_space_branch = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
        elif (scale == 4):
            self.down_size_converter = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2),
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
            ])
            if self.type_of_network == 'gradient_map_dual_domain':
                self.down_size_converter_for_gradient_map_branch = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2),
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
                ])
            elif self.type_of_network == 'k_space_dual_domain':
                self.down_size_converter_for_k_space_branch = nn.Sequential(*[
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
            self.head_for_gradient_map_branch = nn.Sequential(*modules_head_for_gradient_map_branch)
            self.pretail_for_gradient_map_branch = nn.Sequential(*modules_pretail_for_gradient_map_branch)
            self.tail_for_gradient_map_branch = nn.Sequential(*modules_tail_for_gradient_map_branch)
            self.modules_end_stage_fusion_of_outcome = nn.Sequential(*modules_end_stage_fusion_of_outcome)

    def forward(self, x):
        # do NOT understand why need this, may NOT be useful for us 
        """ x = self.sub_mean(x) """
        if self.type_of_network == 'image_single_domain':
            x = self.head(x) # input image goes through first conv layer
            res = self.body(x) # data goes through sveral ResidualGroups
            res = self.pretail(res)
            res += x # long skip connection of RIR
            if (Maintain_Same_Size == True):
                res = self.down_size_converter(res) # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output
            x = self.tail(res) # data goes through upsampling module and one more conv layer
            # do NOT understand why need this, may NOT be useful for us
            """ x = self.add_mean(x) """
            return x, None, 'Single Branch RCAN framework Network'

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
            x2 = self.head_for_gradient_map_branch(calculate_gradient_map(x)) # 2nd gradient branch
            x = tc.cat((x1.unsqueeze(0), x2.unsqueeze(0)), 0)
            res = self.body(x) # data goes through sveral DualResidualGroups

            image_branch_res = res[0,:,:,:,:] # Fetch image branch 
            image_branch_res = image_branch_res.squeeze(0)
            image_branch_res = self.pretail(image_branch_res)
            image_branch_res += x1 # long skip connection at image branch

            gradientmap_branch_res = res[1,:,:,:,:] # Fetch gradientmap branch 
            gradientmap_branch_res = gradientmap_branch_res.squeeze(0)
            gradientmap_branch_res = self.pretail_for_gradient_map_branch(gradientmap_branch_res)
            gradientmap_branch_res += x2 # long skip connection at gradientmap branch
            if (Maintain_Same_Size == True):
                # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output
                image_branch_res = self.down_size_converter(image_branch_res)
                gradientmap_branch_res = self.down_size_converter_for_gradient_map_branch(gradientmap_branch_res)
            image_branch_y = self.tail(image_branch_res) # data goes through upsampling module and one more conv layer
            gradientmap_branch_y = self.tail_for_gradient_map_branch(gradientmap_branch_res) # data goes through upsampling module and one more conv layer
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
                k_space_branch_res = self.down_size_converter_for_k_space_branch(k_space_branch_res)
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





"downsmapling(downsize) conv layer(used in DownsamplingResBlock, which will be used in U-Net)"
def downsampling_conv(in_channels, out_channels, kernel_size = 3, stride = 2, bias = False):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = kernel_size//2, bias = bias)

"The Downsampling Res Block for U-Net"
class DownsamplingResBlock(nn.Module):
    def __init__(self, conv, act, in_channels, out_channels, scale, hidden_channels=None):
        super(DownsamplingResBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        self.block = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size = 1, stride = 1, padding=0, bias=False),   # """ conv(in_channels, hidden_channels, 1), """
                act,
                conv(hidden_channels, hidden_channels, 3),
                act,
                nn.Conv2d(hidden_channels, in_channels, kernel_size = 1, stride = 1, padding=0, bias=False),   # """ conv(hidden_channels, in_channels, 1), """
                act
            )

        if scale == 2:
            self.downsampling_module = nn.Sequential(
                                        downsampling_conv(in_channels, out_channels),
                                        act
            )
        elif scale == 4:
            self.downsampling_module = nn.Sequential(
                                        downsampling_conv(in_channels, in_channels, 3),
                                        act,
                                        downsampling_conv(in_channels, out_channels, 3),
                                        act
            )
        else:
            raise ValueError("scale must be 2 or 4.")

    def forward(self, x):
        y = x + self.block(x)
        return self.downsampling_module(y)

"Not finished yet!!!!! U_Net based Dual Domain Network for Super Resolution MRI"
class U_Net_Based_MRI_SR_Dual_Domain_2D(nn.Module):
    """
    U-Net framework in this code, consists of each RG(Residual Group explained in RCAN paper) as one layer of encoder in U-Net.
    """
    def __init__(self, args):
        super(U_Net_Based_MRI_SR_Dual_Domain_2D, self).__init__()
        if args['type_of_network'] == 'image_single_domain':
            self.type_of_network = 'image_single_domain'
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
        elif args['conv_layer_type'] == 'involution':
            conv = involution

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

        n_resgroups = args['n_resgroups'] # number of RGs in U-Net framework
        self.n_resgroups = n_resgroups # pass to self
        n_rcablocks = args['n_rcablocks'] # number of RCABs in one RG
        n_colors = args['n_colors'] # number of channels going of input of entire model
        n_feats = args['n_feats'] # number of feature maps/channels we expect to have after the encoder of U-Net framework(which contains n_resgroups residual groups)
        kernel_size = 3 # conv filter size used for all conv in RCAN
        reduction = 1 # reduction is hardcoded as 1 for U-Net. The r mentioned in 3.3 Channel Attention in RCAN paper
        scale = args['scale'] # resize factor, e.g. 2, 4, for every down/up sampling block
        use_channel_and_spatial_attention_inside_upsampler = args['use_channel_and_spatial_attention_inside_upsampler'] # whether we use channel and spatial attention block inside upsampler, e.g. True, False
        use_channel_and_spatial_attention_inside_RCAB = args['use_channel_and_spatial_attention_inside_RCAB']   #  whether we use channel and spatial attention block inside RCAB to replace CALayer, e.g. True, False
        channel_and_spatial_attention_framework = args['channel_and_spatial_attention_framework']   # which channel and spatial framework is used in the code, e.g. 'CBAM', 'self_attention'
        channel_and_spatial_attention_mode = args['channel_and_spatial_attention_mode'] # which end to end channel and spatial block to use, e.g. 'sequential_mode', 'parallel_mode'

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        # define commonly use head module
        modules_head = [conv(n_colors, n_feats//np.power(2, n_resgroups), kernel_size)] # the first conv layer in U-Net framework.

        # define commonly use(for dual domain network) end_stage_fusion_of_outcome module
        modules_end_stage_fusion_of_outcome = [conv(2*n_colors, n_colors, kernel_size)]

        # define encoder module for different type of network.
        modules_layer = []
        u_net_down_layer = []
        u_net_up_layer = []
        if self.type_of_network == 'image_single_domain':
            for i in range(n_resgroups):
                modules_layer.append( ResidualGroup(
                    conv, n_feats//np.power(2, n_resgroups - i), kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB, 
                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode) )

                u_net_down_layer.append( DownsamplingResBlock(conv = conv, act =act, in_channels = n_feats//np.power(2, n_resgroups - i), 
                    out_channels = n_feats//np.power(2, n_resgroups - i - 1), scale = scale) )
                
                if i == n_resgroups - 1:
                    u_net_up_layer.append( nn.Sequential(
                                                nn.Conv2d(n_feats, n_feats, kernel_size = 1),
                                                act,
                                                Upsampler(conv, scale, n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                                                            channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                                                            channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                                                nn.Conv2d(n_feats, n_feats//2, kernel_size = 1),
                                                ) )
                else:
                    u_net_up_layer.append( nn.Sequential(
                                                nn.Conv2d(n_feats//np.power(2, n_resgroups - i - 2), n_feats//np.power(2, n_resgroups - i - 1), kernel_size = 1),
                                                act,
                                                Upsampler(conv, scale, n_feats//np.power(2, n_resgroups - i - 1), use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                                                            channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                                                            channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                                                nn.Conv2d(n_feats//np.power(2, n_resgroups - i - 1), n_feats//np.power(2, n_resgroups - i), kernel_size = 1),
                                                ) )
            
            self.modules_layer = nn.ModuleList(modules_layer)
            self.u_net_down_layer = nn.ModuleList(u_net_down_layer)
            self.u_net_up_layer = nn.ModuleList(u_net_up_layer)
            self.last_upsampling_block = Upsampler(conv, scale, n_feats//np.power(2, n_resgroups), use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                                                            channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                                                            channel_and_spatial_attention_mode = channel_and_spatial_attention_mode)

        elif self.type_of_network == 'gradient_map_dual_domain':
            modules_encoder = [
                GradientMapDualResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB, 
                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode) \
                for _ in range(n_resgroups)]
            modules_head_for_gradient_map_branch = [conv(n_colors, n_feats, kernel_size)]
            modules_pretail_for_gradient_map_branch = [conv(n_feats, n_feats, kernel_size)]
            modules_tail_for_gradient_map_branch = [
            Upsampler(conv, scale, n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                            channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
            conv(n_feats, n_colors, kernel_size)]
        elif self.type_of_network == 'k_space_dual_domain':
            modules_encoder = [
                KSpaceDualResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB, 
                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode) \
                for _ in range(n_resgroups)]
            modules_head_for_k_space_branch = [conv(2*n_colors, 2*n_feats, kernel_size)]
            modules_pretail_for_k_space_branch = [conv(2*n_feats, 2*n_feats, kernel_size)]
            modules_tail_for_k_space_branch = [
                Upsampler(conv, scale, 2*n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                            channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
                conv(2*n_feats, 2*n_colors, kernel_size)]
        elif self.type_of_network == 'wavelets_transform_dual_domain':
            modules_encoder = [
                WaveletsTransformDualResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB, 
                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode) \
                for _ in range(n_resgroups)]

        # define tail module. The last stage is last stage upsampling module and one more conv layer to make channel number equals to n_colors in the output image.
        modules_tail = [
            Upsampler(conv, scale, 2*n_feats//np.power(2, n_resgroups), use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                            channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode),
            conv(2*n_feats//np.power(2, n_resgroups), n_colors, kernel_size)]

        # Add a downsize converter by using conv layer.
        # The purpose of original RCAN designed in original RCAN paper is to upscale scale times of LR image, the output from RIR has same size as input LR image. However, here in our task
        # we actually want to maintain output(SR) and input(LR) same as, it means we need to downsize the image before leave it into upscale module. That is the reason we need extra conv layer to perform 
        # down size with scale time first before going through upscale with scale time
        if (scale == 2):
            # W2=(W1−F+2P)/S+1, H2=(H1−F+2P)/S+1.
            self.down_size_converter = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
            if self.type_of_network == 'gradient_map_dual_domain':
                self.down_size_converter_for_gradient_map_branch = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
            elif self.type_of_network == 'k_space_dual_domain':
                self.down_size_converter_for_k_space_branch = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
        elif (scale == 4):
            self.down_size_converter = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2),
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
            ])
            if self.type_of_network == 'gradient_map_dual_domain':
                self.down_size_converter_for_gradient_map_branch = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2),
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
                ])
            elif self.type_of_network == 'k_space_dual_domain':
                self.down_size_converter_for_k_space_branch = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2),
                nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=2)
                ])
        else:
            raise ValueError("scale must be 2 or 4.")

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
        if self.type_of_network == 'k_space_dual_domain':
            self.head_for_k_space_branch = nn.Sequential(*modules_head_for_k_space_branch)
            self.pretail_for_k_space_branch = nn.Sequential(*modules_pretail_for_k_space_branch)
            self.tail_for_k_space_branch = nn.Sequential(*modules_tail_for_k_space_branch)
            self.modules_end_stage_fusion_of_outcome = nn.Sequential(*modules_end_stage_fusion_of_outcome)
        if self.type_of_network == 'gradient_map_dual_domain':
            self.head_for_gradient_map_branch = nn.Sequential(*modules_head_for_gradient_map_branch)
            self.pretail_for_gradient_map_branch = nn.Sequential(*modules_pretail_for_gradient_map_branch)
            self.tail_for_gradient_map_branch = nn.Sequential(*modules_tail_for_gradient_map_branch)
            self.modules_end_stage_fusion_of_outcome = nn.Sequential(*modules_end_stage_fusion_of_outcome)

    def forward(self, x):
        # U-Net framework.
        if self.type_of_network == 'image_single_domain':
            x = self.head(x) # input image goes through first conv layer

            #------------------------------- U-Net framework ---------------------------------#
            res = [None]*self.n_resgroups
            for i in range(self.n_resgroups):
                if i == 0:
                    x = self.modules_layer[i](x)
                    res[i] = self.u_net_down_layer[i](x)
                else:
                    res[i - 1] = self.modules_layer[i](res[i - 1])
                    res[i] = self.u_net_down_layer[i](res[i - 1])
            up_res = [None]*self.n_resgroups
            for j in range(self.n_resgroups - 1, -1, -1):
                if j == self.n_resgroups - 1:
                    up_res[j] = self.u_net_up_layer[j](res[j])
                else:
                    up_res[j] = self.u_net_up_layer[j]( tc.cat([res[j], up_res[j + 1]], dim = 1) )
            output_unet_framework = tc.cat([up_res[0], x], dim=1)
            #----------------------------- U-Net framework end -------------------------------#

            output_unet_framework = self.tail(output_unet_framework)    # one more upsampling block to enlarge the size of image and one more conv to make channel number equals to n_colors.

            return output_unet_framework, None, 'Single Branch U-Net framework Network'


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
            x2 = self.head_for_gradient_map_branch(calculate_gradient_map(x)) # 2nd gradient branch
            x = tc.cat((x1.unsqueeze(0), x2.unsqueeze(0)), 0)
            res = self.encoder(x) # data goes through sveral DualResidualGroups

            image_branch_res = res[0,:,:,:,:] # Fetch image branch 
            image_branch_res = image_branch_res.squeeze(0)
            image_branch_res = self.pretail(image_branch_res)
            image_branch_res += x1 # long skip connection at image branch

            gradientmap_branch_res = res[1,:,:,:,:] # Fetch gradientmap branch 
            gradientmap_branch_res = gradientmap_branch_res.squeeze(0)
            gradientmap_branch_res = self.pretail_for_gradient_map_branch(gradientmap_branch_res)
            gradientmap_branch_res += x2 # long skip connection at gradientmap branch
            if (Maintain_Same_Size == True):
                # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output
                image_branch_res = self.down_size_converter(image_branch_res)
                gradientmap_branch_res = self.down_size_converter_for_gradient_map_branch(gradientmap_branch_res)
            image_branch_y = self.tail(image_branch_res) # data goes through upsampling module and one more conv layer
            gradientmap_branch_y = self.tail_for_gradient_map_branch(gradientmap_branch_res) # data goes through upsampling module and one more conv layer
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
            x = tc.cat((x1_new, x2_new), 0) # make input into encoder, shape of x is (2, N, 2*C, H, W)

            res = self.encoder(x) # data goes through sveral KSpaceDualResidualGroup. shape of res is (2, N, 2*C, H, W)

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
                k_space_branch_res = self.down_size_converter_for_k_space_branch(k_space_branch_res)
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

            res = self.encoder(x2) # data goes through sveral WaveletsTransformDualResidualGroup. shape of res is (N, C, 4, H', W')

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
    def __init__(self, args, not_use_last_conv_to_change_num_channels_to_n_colors = False):
        super(Progressive_Learning_Wrapper_MRI_SR_Dual_Domain_2D, self).__init__()
        self.number_of_progressive_stage = args['number_of_progressive_stage']
        self.long_skip_connection_to_reconstruct_residual_part_only = args['long_skip_connection_to_reconstruct_residual_part_only']
        self.total_scale_factor = args['scale'] ** args['number_of_progressive_stage']
        if args['main_network_framework'] == 'RCAN':
            if self.number_of_progressive_stage == 1:
                self.stage_1 = RCAN_Based_MRI_SR_Dual_Domain_2D(args, not_use_last_conv_to_change_num_channels_to_n_colors = not_use_last_conv_to_change_num_channels_to_n_colors)
            elif self.number_of_progressive_stage == 2:
                self.stage_1 = RCAN_Based_MRI_SR_Dual_Domain_2D(args)
                self.stage_2 = RCAN_Based_MRI_SR_Dual_Domain_2D(args, not_use_last_conv_to_change_num_channels_to_n_colors = not_use_last_conv_to_change_num_channels_to_n_colors)
            elif self.number_of_progressive_stage == 3:
                self.stage_1 = RCAN_Based_MRI_SR_Dual_Domain_2D(args)
                self.stage_2 = RCAN_Based_MRI_SR_Dual_Domain_2D(args)
                self.stage_3 = RCAN_Based_MRI_SR_Dual_Domain_2D(args, not_use_last_conv_to_change_num_channels_to_n_colors = not_use_last_conv_to_change_num_channels_to_n_colors)
            else:
                raise ValueError("Not support more than 3 stage!")
        elif args['main_network_framework'] == 'U_Net':
            if self.number_of_progressive_stage == 1:
                self.stage_1 = U_Net_Based_MRI_SR_Dual_Domain_2D(args)
            elif self.number_of_progressive_stage == 2:
                self.stage_1 = U_Net_Based_MRI_SR_Dual_Domain_2D(args)
                self.stage_2 = U_Net_Based_MRI_SR_Dual_Domain_2D(args)
            elif self.number_of_progressive_stage == 3:
                self.stage_1 = U_Net_Based_MRI_SR_Dual_Domain_2D(args)
                self.stage_2 = U_Net_Based_MRI_SR_Dual_Domain_2D(args)
                self.stage_3 = U_Net_Based_MRI_SR_Dual_Domain_2D(args)
            else:
                raise ValueError("Not support more than 3 stage!")
    
    def forward(self, x):
        if self.number_of_progressive_stage == 1:
            img_outputs, secondary_branch_outputs, network_model_type = self.stage_1(x)
            if self.long_skip_connection_to_reconstruct_residual_part_only:
                x = F.interpolate(x, scale_factor=self.total_scale_factor, mode='bicubic')
                img_outputs = x + img_outputs
            return img_outputs, secondary_branch_outputs, network_model_type
        elif self.number_of_progressive_stage == 2:
            img_outputs, _, _ = self.stage_1(x)
            img_outputs, secondary_branch_outputs, network_model_type = self.stage_2(img_outputs)
            if self.long_skip_connection_to_reconstruct_residual_part_only:
                x = F.interpolate(x, scale_factor=self.total_scale_factor, mode='bicubic')
                img_outputs = x + img_outputs
            return img_outputs, secondary_branch_outputs, network_model_type
        elif self.number_of_progressive_stage == 3:
            img_outputs, _, _ = self.stage_1(x)
            img_outputs, _, _ = self.stage_2(img_outputs)
            img_outputs, secondary_branch_outputs, network_model_type = self.stage_3(img_outputs)
            if self.long_skip_connection_to_reconstruct_residual_part_only:
                x = F.interpolate(x, scale_factor=self.total_scale_factor, mode='bicubic')
                img_outputs = x + img_outputs
            return img_outputs, secondary_branch_outputs, network_model_type
        else:
            raise ValueError("Not support more than 3 stage!")





"HR Reference based MRI Reconstruction for multiple size, e.g. 2x, 4x, 8x, etc."
class HR_Reference_Based_MRI_SR_Dual_Domain_2D(nn.Module):
    def __init__(self, args):
        super(HR_Reference_Based_MRI_SR_Dual_Domain_2D, self).__init__()
        """
        For the HR reference based MRI reconstruction network. The network framework and network type of normal SR branch are based on the args setting 
        up, e.g. RCAN/U-Net as network framework, image single network/gradient map dual domain network/k-space dual domain network/wavelet dual domain 
        network as network type. However, HR reference branch is always a simple RCAN network without upsampler.
        """

        "----------------------------------- Define normal SR branch, make sure not_use_last_conv_to_change_num_channels_to_n_colors = True ---------------------------------"
        if args['main_network_framework'] == 'RCAN' and (args['type_of_network'] == 'image_single_domain' or args['type_of_network'] == 'gradient_map_dual_domain') and args['long_skip_connection_to_reconstruct_residual_part_only'] == False:
            self.SR_branch = Progressive_Learning_Wrapper_MRI_SR_Dual_Domain_2D(args, not_use_last_conv_to_change_num_channels_to_n_colors = True)
        else:
            raise ValueError("For now only support HR reference based network when using RCAN, image_single_domain or gradient_map_dual_domain network type, and no long_skip_connection! Not support HR reference based network by current configuration!")

        "----------------------------------- Define HR reference branch ---------------------------------"
        "Beware: If we do not want the parameter setting for HR reference branch is changable according to the args setting up, just hardcode the following parameters for HR reference branch."
        if args['conv_layer_type'] == 'default_conv':
            conv_for_HR_reference_branch = default_conv
        elif args['conv_layer_type'] == 'coord_conv':
            conv_for_HR_reference_branch = coord_conv
        elif args['conv_layer_type'] == 'deformable_conv':
            conv_for_HR_reference_branch = deformable_conv
        elif args['conv_layer_type'] == 'py_conv':
            conv_for_HR_reference_branch = py_conv
        elif args['conv_layer_type'] == 'involution':
            conv_for_HR_reference_branch = involution

        if args['activation_function_type'] == 'ReLU':
            act_for_HR_reference_branch = nn.ReLU(True)
        elif args['activation_function_type'] == 'Sine':
            act_for_HR_reference_branch = Sine(w0 = 1.0)
        elif args['activation_function_type'] == 'FReLU':
            act_for_HR_reference_branch = 'FReLU'
        elif args['activation_function_type'] == 'Dynamic_ReLU_Type_A':
            act_for_HR_reference_branch = 'Dynamic_ReLU_Type_A'
        elif args['activation_function_type'] == 'Dynamic_ReLU_Type_B':
            act_for_HR_reference_branch = 'Dynamic_ReLU_Type_B'
        
        n_resgroups_for_HR_reference_branch = args['n_resgroups'] # number of RGs in RIR/RCAN
        n_rcablocks_for_HR_reference_branch = args['n_rcablocks'] # number of RCABs in one RG
        n_colors = args['n_colors'] # number of channels going of input of entire model
        n_feats_for_HR_reference_branch = args['n_feats'] # number of feature maps/channels going through the entire model
        kernel_size_for_HR_reference_branch = 3 # conv filter size used for all conv in RCAN
        reduction_for_HR_reference_branch = args['reduction'] # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        use_channel_and_spatial_attention_inside_RCAB_for_HR_reference_branch = args['use_channel_and_spatial_attention_inside_RCAB']   #  whether we use channel and spatial attention block inside RCAB to replace CALayer, e.g. True, False
        channel_and_spatial_attention_framework_for_HR_reference_branch = args['channel_and_spatial_attention_framework']   # which channel and spatial framework is used in the code, e.g. 'CBAM', 'self_attention'
        channel_and_spatial_attention_mode_for_HR_reference_branch = args['channel_and_spatial_attention_mode'] # which end to end channel and spatial block to use, e.g. 'sequential_mode', 'parallel_mode'

        # define head module for HR reference branch
        HR_reference_branch_modules_head = [conv_for_HR_reference_branch(n_colors, n_feats_for_HR_reference_branch, kernel_size_for_HR_reference_branch)] # the first conv layer in RCAN, show in figure 2 of RCAN paper
        
        # define body module for HR reference branch.
        HR_reference_branch_modules_body = [
            ResidualGroup(
                    conv_for_HR_reference_branch, n_feats_for_HR_reference_branch, kernel_size_for_HR_reference_branch, reduction_for_HR_reference_branch, 
                    act=act_for_HR_reference_branch, res_scale=1, n_rcablocks=n_rcablocks_for_HR_reference_branch,
                    use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB_for_HR_reference_branch, 
                    channel_and_spatial_attention_framework = channel_and_spatial_attention_framework_for_HR_reference_branch, 
                    channel_and_spatial_attention_mode = channel_and_spatial_attention_mode_for_HR_reference_branch) \
            for _ in range(n_resgroups_for_HR_reference_branch)]
        self.HR_reference_branch_head = nn.Sequential(*HR_reference_branch_modules_head)
        self.HR_reference_branch_body = nn.Sequential(*HR_reference_branch_modules_body)

        "----------------------------------- Define the last stage fuser ---------------------------------"
        use_channel_and_spatial_attention_inside_RCAB_for_HR_reference_fuser = args['use_channel_and_spatial_attention_inside_RCAB_for_HR_reference_fuser']
        channel_and_spatial_attention_framework_for_HR_reference_fuser = args['channel_and_spatial_attention_framework_for_HR_reference_fuser']
        channel_and_spatial_attention_mode_for_HR_reference_fuser = args['channel_and_spatial_attention_mode_for_HR_reference_fuser']
        self.last_stage_fuser = nn.Sequential(nn.Conv2d(in_channels = args['n_feats'] + n_feats_for_HR_reference_branch, out_channels = args['n_feats'], kernel_size = 1),
                        nn.ReLU(True),
                        RCAB(conv = deformable_conv, n_feat = args['n_feats'], kernel_size = 3, reduction = 16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, 
                            use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB_for_HR_reference_fuser,
                            channel_and_spatial_attention_framework = channel_and_spatial_attention_framework_for_HR_reference_fuser, 
                            channel_and_spatial_attention_mode = channel_and_spatial_attention_mode_for_HR_reference_fuser),
                        nn.Conv2d(args['n_feats'], args['n_feats']//2, kernel_size = 1),
                        nn.ReLU(True),
                        nn.Conv2d(args['n_feats']//2, n_colors, kernel_size = 1)
                        )

    def forward(self, LR, HR_reference):
        # Feature extraction at SR branch.
        # The output x is the output from single branch if 'image_single_domain' type of network is selected, otherwise x is the fused output from 
        # both image main branch and secondary branch if 'gradient_map_dual_domain', 'k_space_dual_domain', or 'wavelets_transform_dual_domain' 
        # type of network is selected.
        x, output_from_secondary_branch_in_SR_branch, network_model_type_of_SR_branch = self.SR_branch(LR)
        
        # Feature extraction at HR reference branch
        y = self.HR_reference_branch_head(HR_reference)
        y = self.HR_reference_branch_body(y)

        # Fuse the output from SR branch and HR reference branch
        x = tc.cat((x, y), 1)   # Beware number of channels for x is only n_colors, but neumber of channels for y is n_feats, in the current implementation.
        final_HR_reference_based_result = self.last_stage_fuser(x)
        
        return final_HR_reference_based_result, output_from_secondary_branch_in_SR_branch, network_model_type_of_SR_branch





device=tc.device("cuda" if use_cuda else "cpu")
if args['use_HR_reference'] == True:
    our_rcan_mri_sr_2d = HR_Reference_Based_MRI_SR_Dual_Domain_2D(args)
else:    
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

negative_tv_loss = NegativeTVLoss(negative_tv_loss_weight = args_loss_weight['negative_total_variation_weight']).to(device)

negative_trace_loss = NegativeTraceLoss(negative_trace_loss_weight = args_loss_weight['negative_trace_weight']).to(device)

uncertainty_kl_loss = UncertaintyKlLoss(uncertainty_kl_loss_weight = args_loss_weight['uncertainty_kl_loss_weight'], use_ssim_guided_uncertainty_kl_loss = args_loss_weight['use_ssim_guided_uncertainty_kl_loss']).to(device)

# =============================================================================
# print('The loss function is L1Loss')
# loss_function = nn.L1Loss(size_average = False).to(device) 
# =============================================================================
# =============================================================================
# print('The loss function is CrossEntropyLoss')
# loss_function = nn.CrossEntropyLoss().to(device)        #----- here use cross entropy loss
# =============================================================================


"""""""""""""""""""""""""""
5. Train the MRI_SR_Dual_Domain_2D part
"""""""""""""""""""""""""""
"Train the MRI_SR_Dual_Domain_2D"
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
    
    if Use_Negative_TV_Loss == True:
        negative_total_variation_for_img_loss_training = 0.0
        negative_toal_variation_for_img_loss_test = 0.0
    
    if Use_Negative_Trace_Loss == True:
        negative_trace_for_img_loss_training = 0.0
        negative_trace_for_img_loss_test = 0.0

    if Predict_Variance_Of_Pixel_For_MRI_SR_And_Use_Uncertainty_KL_Loss == True:
        uncertainty_kl_loss_for_img_loss_training = 0.0
        uncertainty_kl_loss_for_img_loss_test = 0.0

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

        if args['use_HR_reference'] == False:
            "load input data"
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            # print('The data have been loaded' )

            "forward prop"
            # outputs = our_resnext(inputs).double() #-- numpy arrays are 64-bit floating point and will be converted to torch.DoubleTensor standardly. Now, if you use them with your model, you'll need to make sure that your model parameters are also Double
            img_outputs, secondary_branch_outputs, network_model_type = our_rcan_mri_sr_2d(inputs) #-- or using default float as type, however remember to cast the input from Double to Float            
            # print(outputs.size())
            # print('the forward pass has been went')
        elif args['use_HR_reference'] == True:
            "load input data"
            inputs, labels, references = data
            inputs, labels, references = Variable(inputs).to(device), Variable(labels).to(device), Variable(references).to(device)
            # print('The data have been loaded' )

            "forward prop"
            # outputs = our_resnext(inputs).double() #-- numpy arrays are 64-bit floating point and will be converted to torch.DoubleTensor standardly. Now, if you use them with your model, you'll need to make sure that your model parameters are also Double
            img_outputs, secondary_branch_outputs, network_model_type = our_rcan_mri_sr_2d(inputs, references) #-- or using default float as type, however remember to cast the input from Double to Float            
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
        if Use_Gradient_Map_Guided_Pixel_Wise_Loss == True:
            gradient_map_guided_weight_for_pixel_wise_loss = GradientMapGuidedWeightForPixelWiseLoss()
            gradient_map_difference_weight_matrix = gradient_map_guided_weight_for_pixel_wise_loss(img_outputs, labels)
            pixel_wise_loss = args_loss_weight['pixel_wise_weight']*loss_function_L1(gradient_map_difference_weight_matrix*img_outputs, gradient_map_difference_weight_matrix*labels)
        elif Use_SSIM_Map_Guided_Pixel_Wise_Loss == True:
            ssim_map_guided_weight_for_pixel_wise_loss = SSIMMapGuidedWeightForPixelWiseLoss()
            ssim_map_difference_weight_matrix = ssim_map_guided_weight_for_pixel_wise_loss(img_outputs, labels)
            pixel_wise_loss = args_loss_weight['pixel_wise_weight']*loss_function_L1(ssim_map_difference_weight_matrix*img_outputs, ssim_map_difference_weight_matrix*labels)
        else:
            pixel_wise_loss = args_loss_weight['pixel_wise_weight']*loss_function_L1(img_outputs, labels)
        pixel_wise_loss_training += pixel_wise_loss.item()    # Only save the value of pixel_wise_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage 
#            print("pixel_wise_loss: ", pixel_wise_loss)

        if Amplify_High_Frequency_Value_In_K_Space_Loss == True:
            k_space_freq_loss = args_loss_weight['k_space_weight']*(loss_function_MSE(
                create_2d_Gaussian_weights(window_size = SR_freq.shape[2], num_of_samples = SR_freq.shape[0], channel = SR_freq.shape[1]).to(device)*SR_freq.real, 
                create_2d_Gaussian_weights(window_size = HR_freq.shape[2], num_of_samples = HR_freq.shape[0], channel = HR_freq.shape[1]).to(device)*HR_freq.real) + 
                loss_function_MSE(
                    create_2d_Gaussian_weights(window_size = SR_freq.shape[2], num_of_samples = SR_freq.shape[0], channel = SR_freq.shape[1]).to(device)*SR_freq.imag, 
                    create_2d_Gaussian_weights(window_size = HR_freq.shape[2], num_of_samples = HR_freq.shape[0], channel = HR_freq.shape[1]).to(device)*HR_freq.imag))
        else:
            k_space_freq_loss = args_loss_weight['k_space_weight']*(loss_function_MSE(SR_freq.real, HR_freq.real) + loss_function_MSE(SR_freq.imag, HR_freq.imag))
        k_space_freq_loss_training += k_space_freq_loss.item()
#            print(loss_function_MSE(SR_freq.real, HR_freq.real))
#            print(loss_function_MSE(SR_freq.imag, HR_freq.imag))
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

        if Use_Negative_TV_Loss == True:
            negative_total_variation_for_img_loss = negative_tv_loss(img_outputs)
            negative_total_variation_for_img_loss_training += negative_total_variation_for_img_loss.item()    # Only save the value of gram_similarity_between_img_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage
        
        if Use_Negative_Trace_Loss == True:
            negative_trace_for_img_loss = negative_trace_loss(img_outputs, labels)
            negative_trace_for_img_loss_training += negative_trace_for_img_loss.item()

        if Predict_Variance_Of_Pixel_For_MRI_SR_And_Use_Uncertainty_KL_Loss == True:
            uncertainty_kl_loss_for_img_loss = uncertainty_kl_loss(img_outputs, log_of_variance_of_img_outputs, labels, SR_ssim_map_weighted)
            uncertainty_kl_loss_for_img_loss_training += uncertainty_kl_loss_for_img_loss.item()

        if network_model_type == 'Secondary branch is k space branch':
            if Amplify_High_Frequency_Value_In_K_Space_Loss == True:
                k_space_branch_k_space_loss = args_loss_weight['k_space_branch_weight']*(loss_function_MSE(
                    create_2d_Gaussian_weights(window_size = secondary_branch_outputs.shape[2], num_of_samples = secondary_branch_outputs.shape[0], channel = secondary_branch_outputs.shape[1]).to(device)*secondary_branch_outputs.real, 
                    create_2d_Gaussian_weights(window_size = HR_freq.shape[2], num_of_samples = HR_freq.shape[0], channel = HR_freq.shape[1]).to(device)*HR_freq.real) + 
                    loss_function_MSE(
                        create_2d_Gaussian_weights(window_size = secondary_branch_outputs.shape[2], num_of_samples = secondary_branch_outputs.shape[0], channel = secondary_branch_outputs.shape[1]).to(device)*secondary_branch_outputs.imag, 
                        create_2d_Gaussian_weights(window_size = HR_freq.shape[2], num_of_samples = HR_freq.shape[0], channel = HR_freq.shape[1]).to(device)*HR_freq.imag))
            else:
                k_space_branch_k_space_loss = args_loss_weight['k_space_branch_weight']*(loss_function_MSE(secondary_branch_outputs.real, HR_freq.real) + loss_function_MSE(secondary_branch_outputs.imag, HR_freq.imag))
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
        
        if Use_Negative_TV_Loss == True:
            loss = loss + negative_total_variation_for_img_loss
        
        if Use_Negative_Trace_Loss == True:
            loss = loss + negative_trace_for_img_loss

        if Predict_Variance_Of_Pixel_For_MRI_SR_And_Use_Uncertainty_KL_Loss == True:
            loss = loss + uncertainty_kl_loss_for_img_loss

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
            
            if args['use_HR_reference'] == False:
                "load input data"
                inputs, labels = data
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

                SR_img_test, SR_secondary_branch_outputs_test, network_model_type_test = our_rcan_mri_sr_2d(inputs)
            elif args['use_HR_reference'] == True:
                "load input data"
                inputs, labels, references = data
                inputs, labels, references = Variable(inputs).to(device), Variable(labels).to(device), Variable(references).to(device)

                SR_img_test, SR_secondary_branch_outputs_test, network_model_type_test = our_rcan_mri_sr_2d(inputs, references)

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

            if Use_Gradient_Map_Guided_Pixel_Wise_Loss == True:
                gradient_map_guided_weight_for_pixel_wise_loss = GradientMapGuidedWeightForPixelWiseLoss()
                gradient_map_difference_weight_matrix = gradient_map_guided_weight_for_pixel_wise_loss(SR_img_test, labels)
                pixel_wise_loss_test += args_loss_weight['pixel_wise_weight']*loss_function_L1(gradient_map_difference_weight_matrix*SR_img_test, gradient_map_difference_weight_matrix*labels)
            elif Use_SSIM_Map_Guided_Pixel_Wise_Loss == True:
                ssim_map_guided_weight_for_pixel_wise_loss = SSIMMapGuidedWeightForPixelWiseLoss()
                ssim_map_difference_weight_matrix = ssim_map_guided_weight_for_pixel_wise_loss(SR_img_test, labels)
                pixel_wise_loss_test = args_loss_weight['pixel_wise_weight']*loss_function_L1(ssim_map_difference_weight_matrix*SR_img_test, ssim_map_difference_weight_matrix*labels)
            else:
                pixel_wise_loss_test += args_loss_weight['pixel_wise_weight']*loss_function_L1(SR_img_test, labels)
#                print("pixel_wise_loss_test: ", pixel_wise_loss_test)

            if Amplify_High_Frequency_Value_In_K_Space_Loss == True:
                k_space_freq_loss_test += args_loss_weight['k_space_weight']*(loss_function_MSE(
                    create_2d_Gaussian_weights(window_size = SR_test_freq.shape[2], num_of_samples = SR_test_freq.shape[0], channel = SR_test_freq.shape[1]).to(device)*SR_test_freq.real, 
                    create_2d_Gaussian_weights(window_size = HR_test_freq.shape[2], num_of_samples = HR_test_freq.shape[0], channel = HR_test_freq.shape[1]).to(device)*HR_test_freq.real) + 
                    loss_function_MSE(
                        create_2d_Gaussian_weights(window_size = SR_test_freq.shape[2], num_of_samples = SR_test_freq.shape[0], channel = SR_test_freq.shape[1]).to(device)*SR_test_freq.imag, 
                        create_2d_Gaussian_weights(window_size = HR_test_freq.shape[2], num_of_samples = HR_test_freq.shape[0], channel = HR_test_freq.shape[1]).to(device)*HR_test_freq.imag))
            else:
                k_space_freq_loss_test += args_loss_weight['k_space_weight']*(loss_function_MSE(SR_test_freq.real, HR_test_freq.real)+loss_function_MSE(SR_test_freq.imag, HR_test_freq.imag))
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

            if Use_Negative_TV_Loss == True:
                negative_total_variation_for_img_loss_test = negative_tv_loss(SR_img_test)

            if Use_Negative_Trace_Loss == True:
                negative_trace_for_img_loss_test = negative_trace_loss(SR_img_test, labels)

            if Predict_Variance_Of_Pixel_For_MRI_SR_And_Use_Uncertainty_KL_Loss == True:
                uncertainty_kl_loss_for_img_loss_test = uncertainty_kl_loss(SR_img_test, log_of_variance_of_SR_img_test, labels, SR_ssim_map_test_weighted)

            if network_model_type_test == 'Secondary branch is k space branch':
                if Amplify_High_Frequency_Value_In_K_Space_Loss == True:
                    k_space_branch_k_space_loss_test += args_loss_weight['k_space_branch_weight']*(loss_function_MSE(
                        create_2d_Gaussian_weights(window_size = SR_secondary_branch_outputs_test.shape[2], num_of_samples = SR_secondary_branch_outputs_test.shape[0], channel = SR_secondary_branch_outputs_test.shape[1]).to(device)*SR_secondary_branch_outputs_test.real, 
                        create_2d_Gaussian_weights(window_size = HR_test_freq.shape[2], num_of_samples = HR_test_freq.shape[0], channel = HR_test_freq.shape[1]).to(device)*HR_test_freq.real) + 
                        loss_function_MSE(
                            create_2d_Gaussian_weights(window_size = SR_secondary_branch_outputs_test.shape[2], num_of_samples = SR_secondary_branch_outputs_test.shape[0], channel = SR_secondary_branch_outputs_test.shape[1]).to(device)*SR_secondary_branch_outputs_test.imag, 
                            create_2d_Gaussian_weights(window_size = HR_test_freq.shape[2], num_of_samples = HR_test_freq.shape[0], channel = HR_test_freq.shape[1]).to(device)*HR_test_freq.imag))
                else:
                    k_space_branch_k_space_loss_test += args_loss_weight['k_space_branch_weight']*(loss_function_MSE(SR_secondary_branch_outputs_test.real, HR_test_freq.real) + loss_function_MSE(SR_secondary_branch_outputs_test.imag, HR_test_freq.imag))
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
            
            if Use_Negative_TV_Loss == True:
                loss_test = loss_test + negative_total_variation_for_img_loss_test
            
            if Use_Negative_Trace_Loss == True:
                loss_test = loss_test + negative_trace_for_img_loss_test

            if Predict_Variance_Of_Pixel_For_MRI_SR_And_Use_Uncertainty_KL_Loss == True:
                loss_test = loss_test + uncertainty_kl_loss_for_img_loss_test

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
    if Use_Negative_TV_Loss == True:
        negative_total_variation_for_img_loss_test = negative_total_variation_for_img_loss_test/batch_number_test
        print('negative_total_variation_for_img_loss_test: ', negative_total_variation_for_img_loss_test)
    if Use_Negative_Trace_Loss == True:
        negative_trace_for_img_loss_test = negative_trace_for_img_loss_test/batch_number_test
        print('negative_trace_for_img_loss_test: ', negative_trace_for_img_loss_test)
    if Predict_Variance_Of_Pixel_For_MRI_SR_And_Use_Uncertainty_KL_Loss == True:
        uncertainty_kl_loss_for_img_loss_test = uncertainty_kl_loss_for_img_loss_test/batch_number_test
        print('uncertainty_kl_loss_for_img_loss_test: ', uncertainty_kl_loss_for_img_loss_test)
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
    if Use_Negative_TV_Loss == True:
        negative_total_variation_for_img_loss_for_current_epoch = negative_total_variation_for_img_loss_training/ batch_number_training
    if Use_Negative_Trace_Loss == True:
        negative_trace_for_img_loss_for_current_epoch = negative_trace_for_img_loss_training/ batch_number_training
    if Predict_Variance_Of_Pixel_For_MRI_SR_And_Use_Uncertainty_KL_Loss == True:
        uncertainty_kl_loss_for_img_loss_for_current_epoch = uncertainty_kl_loss_for_img_loss_training/ batch_number_training
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
        f.write('Use_Negative_TV_Loss is: %s\n' % Use_Negative_TV_Loss)
        f.write('Use_Negative_Trace_Loss is: %s\n' % Use_Negative_Trace_Loss)
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
    if Use_Negative_TV_Loss == True:
        f.write('The negative_total_variation_for_img_loss for epoch %d is : %f' % (epoch, negative_total_variation_for_img_loss_for_current_epoch))
        f.write('\n')
    if Use_Negative_Trace_Loss == True:
        f.write('The negative_trace_for_img_loss for epoch %d is : %f' % (epoch, negative_trace_for_img_loss_for_current_epoch))
        f.write('\n')
    if Predict_Variance_Of_Pixel_For_MRI_SR_And_Use_Uncertainty_KL_Loss == True:
        f.write('The uncertainty_kl_loss_for_img_loss for epoch %d is : %f' % (epoch, uncertainty_kl_loss_for_img_loss_for_current_epoch))
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
    if Use_Negative_TV_Loss == True:
        f.write('The negative_total_variation_for_img_loss_validation for epoch %d is : %f' % (epoch, negative_total_variation_for_img_loss_test))
        f.write('\n')
    if Use_Negative_Trace_Loss == True:
        f.write('The negative_trace_for_img_loss_validation for epoch %d is : %f' % (epoch, negative_trace_for_img_loss_test))
        f.write('\n')
    if Predict_Variance_Of_Pixel_For_MRI_SR_And_Use_Uncertainty_KL_Loss == True:
        f.write('The uncertainty_kl_loss_for_img_loss_validation for epoch %d is : %f' % (epoch, uncertainty_kl_loss_for_img_loss_test))
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





"Evaluation(Test)"
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
        if args['use_HR_reference'] == False:
            LR_images_test, HR_images_test = testing_data_2
            HR_images_test = HR_images_test.type(tc.FloatTensor)
            img_outputs, secondary_branch_outputs, network_model_type = our_rcan_mri_sr_2d(Variable(LR_images_test).type(tc.FloatTensor).to(device))
        elif args['use_HR_reference'] == True:
            LR_images_test, HR_images_test, references_test = testing_data_2
            HR_images_test = HR_images_test.type(tc.FloatTensor)
            img_outputs, secondary_branch_outputs, network_model_type = our_rcan_mri_sr_2d(Variable(LR_images_test).type(tc.FloatTensor).to(device),
                                                                                            Variable(references_test).type(tc.FloatTensor).to(device))

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
