# -*- coding: utf-8 -*-
"-------------------------------------------------------------------------------------------------"
"""
2D_MRI_SR_Mainly_Transformer_MLP Reconstruct 
(2D Network Mainly based on Transformer and/or MLP for MRI Super-Resolution Image Reconstruction and MRI Motion Artifact Reduction).
"""
"""
Author: chisyliu@hotmail.com *
        hao.li@med.uni-heidelberg.de *
        
        * Both authors contribute equally
Version: 1.0.0
"""
"-------------------------------------------------------------------------------------------------"
"""
This is the current version we are working on, in 20210527
This is a demo code of U_Net_Based_MRI_SR_Transformer_MLP_2D. in this version we have already support following items:
    1)  use gMLP or aMLP based downsampler and pixel shuffle based upsampler in U-Net for MRI SR. gMLP which is another "pure MLP" or "pure MLP with tiny attention" module. See paper: "2021.Pay Attention to MLPs" for more info.
    2)  use gMLP based upsampler to replace pixel shuffle based upsampler. 




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
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import matplotlib.pyplot as plt
from math import exp
import numpy as np
import h5py
import math
import os
import time
import scipy.io
import copy
import pickle

import pytorch_ssim_l1_org
import pytorch_ssim_map
from optimizer import lookahead

"-------------------------------------------------------------------------------------------------"
Single_GPU_training = True

if Single_GPU_training == True:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print('boolean value to see if GPU is ready:', tc.cuda.is_available())
print('number of GPU is', tc.cuda.device_count())
print(tc.cuda.get_device_name(0))
use_cuda = True
training_start = time.perf_counter()

"""""""""""""""""""""""""""""""""""""""""""""
0. Configure all parameter
"""""""""""""""""""""""""""""""""""""""""""""
# --------------------------- configuration of support parameters --------------------------- #
batch_size = 8
EPOCH_NUM = 50
accumulation_steps = int(8//batch_size)

Maintain_in_plane_Size = False # stand for whether we want the output image has same size or NOT(e.g. larger size) as input image, e.g. set as Ture when apply for MRI motion artifact reduction, or applying through-plane downsampling MRI SR reconstruction.
Use_kspace_loss = False # stand for using K-space MSE loss in the total loss function
Use_SSIM_L1_Loss = True # stand for whether we want use SSIM L1 loss in the total loss function
Use_Gradient_Map_L1_Loss = False # stand for whether we want use gradient map L1 loss in the total loss function
Use_Gram_Matrix_L1_Loss = False # stand for whether we want use gram matrix L1 loss(between SR and HR, for increasing texture similarity between SR and HR) in the total loss function
Use_Negative_TV_Loss = False # stand for whether we want to use "1/(total variation + 1.000e-10) loss"(on SR , for providing over smoothing)
Use_Negative_Trace_Loss = False # stand for whether we want to use "1/(trace(sr*hr) + 1.000e-10) loss"(on HR and SR, for increasing similarity between SR and HR)
Use_Gradient_Map_Guided_Pixel_Wise_Loss = False # stand for whether we want to use gradient map guided "attention weights" to multiply with pixel-wise loss. Can NOT be True if Use_SSIM_Map_Guided_Pixel_Wise_Loss is True
Use_SSIM_Map_Guided_Pixel_Wise_Loss = False # stand for whether we want to use SSIM map guided "attention weights" to multiply with pixel-wise loss. Can NOT be True if Use_Gradient_Map_Guided_Pixel_Wise_Loss is True
Amplify_Small_Value_In_Gradient_Map = True # stand for whether we want to amplify small values in gradient map to emphasize the information from gradient values which stand for texture
Amplify_High_Frequency_Value_In_K_Space_Loss = True # stand for whether we want to amplify high frequence loss values in k space loss
Use_Feature_Map_Loss = False # Stand for whether we want use feature map L1 loss in the total loss function
Use_saved_model = False # Load saved network weights
Freeze_random_seed = True # Freeze seed, so every time the network will have the same intialized weightes
Use_ssim_map = False # Use ssim map to calculate loss
Perform_Evaluation = True

if Use_Feature_Map_Loss == True:
    from torchvision.models import vgg19
    
    
# --------------------------- configuration of parameters for 2D_MRI_SR_Dual_Domain Reconstruct --------------------------- #

"The folder where to load the LR, HR data pair"
folder_data_training = 'D:/Hao/SR_data/HCP_data/2x1_folds_3d_downsize_sag_128x1/training/'
file_names_training = os.listdir(folder_data_training)

folder_data_validation = 'D:/Hao/SR_data/HCP_data/2x1_folds_3d_downsize_sag_128x1/validation/'
file_names_validation = os.listdir(folder_data_validation)

folder_data_evaluation = 'D:/Hao/SR_data/HCP_data/2x1_folds_3d_downsize_sag_128x1/evaluation/'
file_names_evaluation = os.listdir(folder_data_evaluation)

"The folder for log and results"
folder_log_path = 'D:/Hao/results/20210623_ResT_MultiScaleExtractor_128x1_2x1folds_3d_downsize_seed1_cosine_101_HCP300/'

"The folder of saved network parameters"
folder_saved_network = 'D:/Hao/results/20210619_conv_gMLP_128x1_2x1folds_3d_downsize_kspace_grad_sigma20_seed1_cosine_101_HCP300/'



args = {'use_HR_reference' : False, 
        'HR_reference_framework': 'gMLP_with_information_exchange',
        'basic_block' : 'MultiScaleExtrctor_ResT',
        'n_colors': 1, 'n_dim': 16, 'LR_image_size': 64,
        'type_of_upsampler': 'gMLP_based_upsampler',
        
        'seed': 1, 'optimizer': 'Adam', 'learning_rate_decay_method': 'cosine_learning_rate_decay',
        'use_learning_rate_warm_up': False, 'how_many_epoch_to_be_used_for_warm_up': 10, 'initial_learning_rate_after_warm_up': 0.0001}

args_loss_weight = {'feature_map_weight': 20, 'pixel_wise_weight': 100, 'k_space_weight': 10, 'ssim_weight': 50, \
                    'gradient_img_weight': 50, 'gradient_grd_weight': 10, 'k_space_branch_weight': 0.02, \
                    'wavelets_branch_weight': 5, 'gram_similarity_weight': 5, 'negative_total_variation_weight': 3, 'negative_trace_weight': 3,\
                    'ssim_component_weight': 2}


# args['use_HR_reference'] = True, stands for whether we select to use HR reference for MRI SR, e.g. True, False
# args['HR_reference_framework'] = 'gMLP_without_information_exchange', stands for what kind of HR reference framework we use, e.g. 'gMLP_without_information_exchange', 'gMLP_with_information_exchange', 'ResT_without_information_exchange', 'ResT_with_information_exchange'.
# args['basic_block'] = 'efficient_transformer', stands for the selected basic block of U-Net, e.g. 'efficient_transformer', 'gMLP', 'MultiScaleExtrctor_gMLP', 'MultiScaleExtrctor_ResT'
# args['n_colors'] = 1, stands for number of channels of input image, e.g. 1 for MRI image, 3 for RGB image.
# args['n_dim'] = 128, stands for number of dimension in the gMLP.
# args['type_of_upsampler'] = 'conv_based_upsampler', stands for type of upsampler, e.g. 'conv_based_upsampler', 'gMLP_based_upsampler'
# arg['optimizer'] = ['Adam'] # stand for which optimizer we want use for training, e.g. 'Adam', 'SGD_with_momentum', 'look_ahead'
# arg['learning_rate_decay_method'] = ['cosine_learning_rate_decay'] # stand for which learning rate decay method we want use for training, e.g. 'cosine_learning_rate_decay', 'multi_step_learning_rate', 'step_learning_rate', 'cosine_learning_rate_warm_restarts'


#if args['use_HR_reference'] == False:
"""
The data loading pipeline for ordinary multi-channel SISR MRI SR or RGB SISR:
"""
"""""""""""""""""""""""""""""""""""""""""""""
1.1.c. MRI HR and LR Data pair preprocessing training part
"""""""""""""""""""""""""""""""""""""""""""""
# =============================================================================
# h5py.version
# =============================================================================


num_low_resolution_mat_file = 0
num_high_resolution_groundtruth_mat_file = 0
num_reference_mat_file = 0

for idx_file in file_names_training:
    print(idx_file)
    if '.mat' in os.path.join(folder_data_training, idx_file):
        print('One more low resolution image set exist')
        num_low_resolution_mat_file = num_low_resolution_mat_file + 1
        print(os.path.join(folder_data_training, idx_file))
        file_data = h5py.File(os.path.join(folder_data_training, idx_file), 'r')
        data_low_resolution = file_data['LR'][:] #----- numpy array
        print('Training data: Shape of LR data is: ', np.shape(data_low_resolution))
        print(data_low_resolution.dtype)
        torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_low_resolution = torch_data_low_resolution.permute(0, 1, 3, 2)
        print('Training data: Shape of LR data in Torch is: ', np.shape(torch_data_low_resolution))
        if num_low_resolution_mat_file == 1:
            torch_data_low_resolution_sequence = torch_data_low_resolution
        elif num_low_resolution_mat_file > 1:
            print(num_low_resolution_mat_file)
            torch_data_low_resolution_sequence = tc.cat((torch_data_low_resolution_sequence, torch_data_low_resolution), 0)
        print('Training data: Shape of LR data sequence in Torch is: ', np.shape(torch_data_low_resolution_sequence))
#        elif 'HRGT_training' in os.path.join(folder_data_training, idx_file):
#            print('One more high resolution groundtruth image set exist')
        num_high_resolution_groundtruth_mat_file = num_high_resolution_groundtruth_mat_file + 1
#            print(os.path.join(folder_data_training, idx_file))
#            file_data_high_resolution_groundtruth = h5py.File(os.path.join(folder_data_training, idx_file), 'r')
        data_high_resolution_groundtruth = file_data['HRGT'][:] #----- numpy array
        print('Training data: Shape of HR data is: ', np.shape(data_high_resolution_groundtruth))
        torch_data_high_resolution_groundtruth = tc.from_numpy(data_high_resolution_groundtruth) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.permute(0, 1, 3, 2)
        print('Training data: Shape of HR data in Torch is: ', np.shape(torch_data_high_resolution_groundtruth))
        if num_high_resolution_groundtruth_mat_file == 1:
            torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth
        if num_high_resolution_groundtruth_mat_file > 1:
            print(num_high_resolution_groundtruth_mat_file)
            torch_data_high_resolution_groundtruth_sequence = tc.cat((torch_data_high_resolution_groundtruth_sequence, torch_data_high_resolution_groundtruth), 0)
        print('Training data: Shape of HR data sequence in Torch is: ', np.shape(torch_data_high_resolution_groundtruth_sequence))
        if args['use_HR_reference']:
#                print('One more reference image set exist')
            num_reference_mat_file = num_reference_mat_file + 1
#                print(os.path.join(folder_log_path, idx_file))
#                file_data_reference = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
            data_reference = file_data['REF'][:] #----- numpy array
            print(np.shape(data_reference))
            torch_data_reference = tc.from_numpy(data_reference) #----- torch type data could be read by tc.utils.data.TensorDataset
            "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
            torch_data_reference = torch_data_reference.permute(0, 1, 3, 2)
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
torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.float()
print(np.shape(torch_data_low_resolution_sequence))
torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.float()
print(np.shape(torch_data_high_resolution_groundtruth_sequence))
if args['use_HR_reference']:
    torch_data_reference_sequence = torch_data_reference_sequence.float()
    print(np.shape(torch_data_reference_sequence))
# num_training_samples = math.floor(torch_data_low_resolution_sequence.size(0))
print('All mat files have been concatenated into one tensor for each type, data is ready to be loaded!')

"""""""""""""""""""""""""""""""""""""""""""""
1.2.c. Load MRI HR and LR Data pair training part
"""""""""""""""""""""""""""""""""""""""""""""
torch_data_low_resolution_training_sequence = torch_data_low_resolution_sequence.float()
torch_data_high_resolution_groundtruth_training_sequence = torch_data_high_resolution_groundtruth_sequence.float()
if args['use_HR_reference']:
    torch_data_reference_training_sequence = torch_data_reference_sequence.float()
    trainset = tc.utils.data.TensorDataset(torch_data_low_resolution_training_sequence, torch_data_high_resolution_groundtruth_training_sequence, torch_data_reference_training_sequence)
    # tc.multiprocessing.freeze_support()
else:
    trainset = tc.utils.data.TensorDataset(torch_data_low_resolution_training_sequence, torch_data_high_resolution_groundtruth_training_sequence)

trainloader = tc.utils.data.DataLoader(
                    trainset, 
                    batch_size = batch_size,
                    shuffle = True, 
                    num_workers = 0,
                    pin_memory = False,
                    drop_last = True)

    #testset = tc.utils.data.TensorDataset(torch_data_low_resolution_test_sequence, torch_data_high_resolution_groundtruth_test_sequence)

    #testloader = tc.utils.data.DataLoader(
    #                    testset, 
    #                    batch_size = batch_size,
    #                    shuffle = True, 
    #                    num_workers = 0)

"""""""""""""""""""""""""""""""""""""""""""""
2.1.c. MRI HR and LR Validation Data pair preprocessing validation part
"""""""""""""""""""""""""""""""""""""""""""""
num_low_resolution_mat_file = 0
num_high_resolution_groundtruth_mat_file = 0
num_reference_mat_file = 0

for idx_file in file_names_validation:
    print(idx_file)
    if '.mat' in os.path.join(folder_data_validation, idx_file):
        print('One more low resolution image set exist')
        num_low_resolution_mat_file = num_low_resolution_mat_file + 1
        print(os.path.join(folder_data_validation, idx_file))
        file_data = h5py.File(os.path.join(folder_data_validation, idx_file), 'r')
        data_low_resolution = file_data['LR'][:] #----- numpy array
        print('Evaluation data: Shape of LR data is: ', np.shape(data_low_resolution))
        print(data_low_resolution.dtype)
        torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_low_resolution = torch_data_low_resolution.permute(0, 1, 3, 2)
        print('Evaluation data: Shape of LR data in Torch is: ', np.shape(torch_data_low_resolution))
        if num_low_resolution_mat_file == 1:
            torch_data_low_resolution_sequence = torch_data_low_resolution
        elif num_low_resolution_mat_file > 1:
            print(num_low_resolution_mat_file)
            torch_data_low_resolution_sequence = tc.cat((torch_data_low_resolution_sequence, torch_data_low_resolution), 0)
        print('Evaluation data: Shape of LR data sequence in Torch is: ', np.shape(torch_data_low_resolution_sequence))
#        elif 'HRGT_validation' in os.path.join(folder_data_validation, idx_file):
#            print('One more high resolution groundtruth image set exist')
        num_high_resolution_groundtruth_mat_file = num_high_resolution_groundtruth_mat_file + 1
#            print(os.path.join(folder_data_validation, idx_file))
#            file_data_high_resolution_groundtruth = h5py.File(os.path.join(folder_data_validation, idx_file), 'r')
        data_high_resolution_groundtruth = file_data['HRGT'][:] #----- numpy array
        print('Evaluation data: Shape of HR data is: ', np.shape(data_high_resolution_groundtruth))
        torch_data_high_resolution_groundtruth = tc.from_numpy(data_high_resolution_groundtruth) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.permute(0, 1, 3, 2)
        print('Evaluation data: Shape of HR data in Torch is: ', np.shape(torch_data_high_resolution_groundtruth))
        if num_high_resolution_groundtruth_mat_file == 1:
            torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth
        if num_high_resolution_groundtruth_mat_file > 1:
            print(num_high_resolution_groundtruth_mat_file)
            torch_data_high_resolution_groundtruth_sequence = tc.cat((torch_data_high_resolution_groundtruth_sequence, torch_data_high_resolution_groundtruth), 0)
        print('Evaluation data: Shape of HR data sequence in Torch is: ', np.shape(torch_data_high_resolution_groundtruth_sequence))
        if args['use_HR_reference']:
#                print('One more reference image set exist')
            num_reference_mat_file = num_reference_mat_file + 1
#                print(os.path.join(folder_log_path, idx_file))
#                file_data_reference = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
            data_reference = file_data['REF'][:] #----- numpy array
            print(np.shape(data_reference))
            torch_data_reference = tc.from_numpy(data_reference) #----- torch type data could be read by tc.utils.data.TensorDataset
            "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
            torch_data_reference = torch_data_reference.permute(0, 1, 3, 2)
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
torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.float() 
print(np.shape(torch_data_low_resolution_sequence))
torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.float()
print(np.shape(torch_data_high_resolution_groundtruth_sequence))
if args['use_HR_reference']:
    torch_data_reference_sequence = torch_data_reference_sequence.float()
    print(np.shape(torch_data_reference_sequence))
# num_validation_samples = math.floor(torch_data_low_resolution_sequence.size(0))
print('All mat files have been concatenated into one tensor for each type, data is ready to be loaded!')

"""""""""""""""""""""""""""""""""""""""""""""
2.2.c. Load MRI HR and LR Validation Data pair validation part
"""""""""""""""""""""""""""""""""""""""""""""
torch_data_low_resolution_validation_sequence = torch_data_low_resolution_sequence.float()
torch_data_high_resolution_groundtruth_validation_sequence = torch_data_high_resolution_groundtruth_sequence.float()
if args['use_HR_reference']:
    torch_data_reference_validation_sequence = torch_data_reference_sequence.float()
    validationset = tc.utils.data.TensorDataset(torch_data_low_resolution_validation_sequence, torch_data_high_resolution_groundtruth_validation_sequence, torch_data_reference_validation_sequence)
    # tc.multiprocessing.freeze_support()
else:
    # tc.multiprocessing.freeze_support()
    validationset = tc.utils.data.TensorDataset(torch_data_low_resolution_validation_sequence, torch_data_high_resolution_groundtruth_validation_sequence)

validationloader = tc.utils.data.DataLoader(
                        validationset, 
                        batch_size = batch_size,
                        shuffle = True, 
                        num_workers = 0,
                        pin_memory = False,
                        drop_last = True)

 

"""""""""""""""""""""""""""""""""""""""""""""""
3. Define network modules and architecture part
"""""""""""""""""""""""""""""""""""""""""""""""
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


"calculate gradient map for any input MRI image"
def calculate_gradient_map(n_colors, img):
    
    if n_colors == 1:
        # sobel operator
        vertical_edge_mask = tc.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0)
        horizontal_edge_mask = tc.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0)
        
        vertical_edge_mask = vertical_edge_mask.float().unsqueeze(0).cuda()
        horizontal_edge_mask = horizontal_edge_mask.float().unsqueeze(0).cuda()

        gradient_vertical_map = F.conv2d(img, vertical_edge_mask, padding = 1, stride = 1, groups = 1)
        gradient_horizontal_map = F.conv2d(img, horizontal_edge_mask, padding = 1, stride = 1, groups = 1)

        gradient_map = abs(gradient_vertical_map) + abs(gradient_horizontal_map)
    elif n_colors == 2:
        vertical_edge_mask = tc.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0)
        horizontal_edge_mask = tc.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0)
        vertical_edge_mask = tc.cat((vertical_edge_mask, vertical_edge_mask),0)
        horizontal_edge_mask = tc.cat((horizontal_edge_mask, horizontal_edge_mask),0)
        
        vertical_edge_mask = vertical_edge_mask.float().unsqueeze(0).cuda()
        horizontal_edge_mask = horizontal_edge_mask.float().unsqueeze(0).cuda()
        
        gradient_vertical_map = F.conv2d(img, vertical_edge_mask, padding = 1, stride = 1, groups = 1)
        gradient_horizontal_map = F.conv2d(img, horizontal_edge_mask, padding = 1, stride = 1, groups = 1)

        gradient_map = abs(gradient_vertical_map) + abs(gradient_horizontal_map)
    elif n_colors >= 3:   # For MRI image with number of channel = 3, we just stack 3 MRI image with number of channel = 1 together. 
        # 3D prewitt operator
        vertical_edge_mask = tc.Tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).unsqueeze(0)
        horizontal_edge_mask = tc.Tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).unsqueeze(0)
        vertical_edge_mask = tc.cat((vertical_edge_mask, vertical_edge_mask, vertical_edge_mask),0)
        horizontal_edge_mask = tc.cat((horizontal_edge_mask, horizontal_edge_mask, horizontal_edge_mask),0)
        through_plane_edge_mask = horizontal_edge_mask.permute(1,2,0)
        
        vertical_edge_mask = vertical_edge_mask.float().unsqueeze(0).unsqueeze(0).cuda()
        horizontal_edge_mask = horizontal_edge_mask.float().unsqueeze(0).unsqueeze(0).cuda()
        through_plane_edge_mask = through_plane_edge_mask.float().unsqueeze(0).unsqueeze(0).cuda()

        gradient_vertical_map = F.conv3d(img.unsqueeze(1), vertical_edge_mask, padding = 1, stride = 1, groups = 1)
        gradient_horizontal_map = F.conv3d(img.unsqueeze(1), horizontal_edge_mask, padding = 1, stride = 1, groups = 1)
        gradient_through_plane_map = F.conv3d(img.unsqueeze(1), through_plane_edge_mask, padding = 1, stride = 1, groups = 1)

        gradient_map = abs(gradient_vertical_map) + abs(gradient_horizontal_map) + abs(gradient_through_plane_map)
    else:
        raise SystemExit('Error: Dimension of gradient operator is not correct!')

    
#    gradient_map = tc.cat((gradient_vertical_map, gradient_horizontal_map),1)

    if Amplify_Small_Value_In_Gradient_Map == True:
#        gradient_map = (1 - tc.exp(-2.5 * abs(gradient_map)))*(gradient_map/abs(gradient_map)) # 1 - exp(-ax), a = 2.5
        gradient_map = 1 - tc.exp(-2.5 * gradient_map)

    return gradient_map


"""(Not used yet in this code)Calculate PSNR for MRI image in shape (N, C, H, W)"""
def calc_psnr_for_mri_image(img1, img2):
    ### args:
        # img1: pytorch tensor, shape is [N, C, H, W]
        # img2: pytorch tensor, shape is [N, C, H, W]

    diff = tc.add(img1, -img2)
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
        time_domain_result = tc.fft.ifftn(x, dim = (-3,-2,-1))
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



class MultiScaleExtractor(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(MultiScaleExtractor, self).__init__()
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding = 1)
        self.conv5 = nn.Conv2d(channel_in, channel_out, kernel_size=5, stride=1, padding = 2)
        self.conv7 = nn.Conv2d(channel_in, channel_out, kernel_size=7, stride=1, padding = 3)
        self.relu = nn.ReLU()
        self.fuser = nn.Conv2d(channel_out*3, channel_out, kernel_size=1, stride=1, padding=0)
        
    def forward(self,x):
        x1 = self.relu(self.conv3(x))
        x2 = self.relu(self.conv5(x))
        x3 = self.relu(self.conv7(x))
        y = self.fuser(tc.cat((x1,x2,x3), dim=1))
        return x+y
        
        

"gMLP or aMLP, which is another 'pure MLP' or 'pure MLP with tiny attention' module. See paper: '2021.Pay Attention to MLPs' for more info."
class gMLPVision(nn.Module):
    def __init__(
        self,
        *,
        image_size = 32,
        patch_size = 1,
        dim = 512,          # Input feature's patch embedding dimension. See figure 1 in paper: 2021.Pay Attention to MLPs.
        depth = 2,          # Number of gMLP layers, L. See figure 1 in paper: 2021.Pay Attention to MLPs.
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


class gMLPVisionInfoExchange(nn.Module):
    def __init__(
        self,
        *,
        image_size = 32,
        patch_size = 1,
        dim = 512,          # Input feature's patch embedding dimension. See figure 1 in paper: 2021.Pay Attention to MLPs.
        depth = 2,          # Number of gMLP layers, L. See figure 1 in paper: 2021.Pay Attention to MLPs.
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
        self.depth = depth

        dim_ff = dim * ff_mult      # Hidden dimension?
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding: Generate N tokens, where N is the num_patches, equals to h*w/(patch_size**2). Each token is a vector with dimension (1, dim).
        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(channels * patch_size ** 2, dim)
        )   # Size of the output from this module is (batch_size, num_patches, dim).

        self.prob_survival = prob_survival

        self.gmlp_residual_block = []
        for i in range(depth):
            # I have to add .to('cuda') here, otherwise it will lead to strang problem "Tensor for argument #3 ‘mat2’ is on CPU, but expected it to be on GPU" later when calling gmlp_residual_block.
            self.gmlp_residual_block.append( GatingMlpResidualBlockInfoExchange(dim, dim_ff, num_patches, attn_dim).to('cuda') )

    def feature_mapping(self, x):
        x = x.view(
            x.size(0),
            int(self.image_size / self.patch_size),
            int(self.image_size / self.patch_size),
            self.dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x, injected_key):
        x = self.to_patch_embed(x)      # (B, C, H, W) --> (B, N, dim) where N equal to H*W/(patch_size**2)
        """ layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival) """
        output_key_for_hr_reference_branch = []
        if injected_key == None or injected_key[0] == None:
            for i in range(self.depth):
                x, output_key = self.gmlp_residual_block[i](x, None)        # (B, N, dim) --> (B, N, dim)
                output_key_for_hr_reference_branch.append(output_key)       # append output_key from MRI LR information.
        else:
            for i in range(self.depth):
                x, output_key = self.gmlp_residual_block[i](x, injected_key[i])     # (B, N, dim) --> (B, N, dim)
                output_key_for_hr_reference_branch.append(output_key)       # append output_key from MRI HR Ref information, although such output_key will NOT be used.
        x = self.feature_mapping(x)     # (B, N, dim) --> (B, dim, H/patch_size, W/patch_size)
        return x, output_key_for_hr_reference_branch

# helper classes
class AttentionForGmlpInfoExchange(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal
        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x, injected_key = None):
        device = x.device

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        if injected_key != None:
            k = injected_key
        sim = tc.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = tc.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -tc.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = tc.einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out), k

class SpatialGatingUnitInfoExchange(nn.Module):
    def __init__(self, dim, dim_seq, attn_dim = None, causal = False):
        super().__init__()
        self.causal = causal
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)
        self.attn = AttentionForGmlpInfoExchange(dim * 2, dim, attn_dim, causal) if exists(attn_dim) else None
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x, injected_key = None):
        device = x.device

        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            mask = tc.ones(weight.shape[:2], device = device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.)

        gate = F.conv1d(gate, weight, bias)

        if exists(self.attn):
            outcome, output_key = self.attn(x, injected_key)
            gate += outcome
        return gate * res, output_key

class GatingMlpResidualBlockInfoExchange(nn.Module):
    def __init__(self, dim, dim_ff, num_patches, attn_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_1 = nn.Linear(dim, dim_ff*2)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnitInfoExchange(dim_ff, num_patches, attn_dim)
        self.proj_2 = nn.Linear(dim_ff, dim)

    def forward(self, x, injected_key = None):
        res = self.norm(x)
        res = self.proj_1(res)
        res = self.activation(res)
        res, output_key = self.spatial_gating_unit(res, injected_key)
        res = self.proj_2(res)
        return res + x, output_key


"EfficientTransformerBlock in ResTransformer. See paper: 2021.ResT:An Efficient Transformer for Visual Recognition"
class EfficientTransformerBlock(nn.Module):
    # EMSA block in ResTransformer. See section 2.2.Efficient Transformer Block and equation (5) in paper: 2021.ResT:An Efficient Transformer for Visual Recognition
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EMSA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            sr_ratio=sr_ratio, apply_transform=apply_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpInEfficientTransformerBlock(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, H, W):
        # Following code stands for equation (5) in paper:2021.ResT:An Efficient Transformer for Visual Recognition.
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x

class MlpInEfficientTransformerBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class EMSA(nn.Module):
    # Effiecient Multi-head Self-Attention module. See figure 3 and equation (4) in paper:2021.ResT:An Efficient Transformer for Visual Recognition.
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 sr_ratio=1,
                 apply_transform=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio+1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class PatchEmbed(nn.Module):
    # Image to Patch Embedding. See section 2.3. Patch Embedding in paper:2021.ResT:An Efficient Transformer for Visual Recognition.
    def __init__(self, patch_size=16, in_ch=3, out_ch=768, with_pos=False, for_upsampling = False):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.for_upsampling = for_upsampling
        if for_upsampling == False:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size+1, stride=patch_size, padding=patch_size // 2)
        else: # for_upsampling == True
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size+1, stride=patch_size//2, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(out_ch)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        if self.for_upsampling == False:
            H, W = H // self.patch_size[0], W // self.patch_size[1]
        else: # self.for_upsampling == True
            H, W = H // (self.patch_size[0]//2), W // (self.patch_size[1]//2)
        return x, (H, W)

class PA(nn.Module):
    # Pixel-wise attention positioning encoding.
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Following code stands for equation (8) in paper:2021.ResT:An Efficient Transformer for Visual Recognition.
        return x * self.sigmoid(self.pa_conv(x))

class GL(nn.Module):
    # Group linear positioning encoding.
    def __init__(self, dim):
        super().__init__()
        self.gl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        # Following code stands for equation (7) in paper:2021.ResT:An Efficient Transformer for Visual Recognition.
        return x + self.gl_conv(x)

class BasicStem(nn.Module):
    # Convert input data from (B, 1, H, W) to (B, C, H/2, W/2)
    def __init__(self, in_ch=1, out_ch=32, with_pos=True):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        #self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        #self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        #x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x
    
    
    
"Upsampler Module, implemented by employeed of sub-pixel conv"
class Upsampler(nn.Sequential):
    """
    Upsampling/Upscale module, used as last part of "SR reconstruction network model" if the network model employ the "post-upsampling mode".
    Beware the actual upsampling approach is "sub-pixel conv" (which is nn.PixelShuffle() in Pytorch) which was proposed in
    paper: "2016. Real-Time single image and video super-resolution using an efficient sub-pixel convolutional neural network".
    Such sub-pixel conv actually constructs F ∗ S^2 feature maps of dimensions H ×W are reshaped into F feature maps of dimensions H ∗ S × W ∗ S, 
    where S is the upsampling factor.
    """
    def __init__(self, n_feats, reduce_number_of_channels_in_half = False):
        super(Upsampler, self).__init__()
        if reduce_number_of_channels_in_half == False:
            self.upsampler = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats, kernel_size = 3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(n_feats, n_feats, kernel_size = 3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor = 2)
            ])
        else: # reduce_number_of_channels_in_half == True
            self.upsampler = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats//2, kernel_size = 3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(n_feats//2, n_feats//2, kernel_size = 3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor = 2)
            ])

    def forward(self, x):
        upsampled_x = self.upsampler(x)
        return upsampled_x



"gMLP based U-Net for Super Resolution MRI"
class U_Net_Based_MRI_SR_Transformer_MLP_2D(nn.Module):
    """
    U-Net framework in this code, consists of each gMLP as one layer in U-Net.
    """
    def __init__(self, args):
        super(U_Net_Based_MRI_SR_Transformer_MLP_2D, self).__init__()

        n_colors = args['n_colors'] # number of channels going of input of entire model.
        LR_image_size = args['LR_image_size'] # size of LR image.
        n_dim = args['n_dim'] # number of dimension we expect to use for gMLP.
        type_of_upsampler = args['type_of_upsampler']
        
        self.fisrt_u_net_layer = gMLPVision(image_size = LR_image_size, patch_size = 2, dim = n_dim, channels = n_colors, attn_dim = 2)
        self.second_u_net_layer = gMLPVision(image_size = LR_image_size//2, patch_size = 2, dim = n_dim*4, channels = n_dim, attn_dim = 2)
        self.third_u_net_layer = gMLPVision(image_size = LR_image_size//4, patch_size = 2, dim = n_dim*16, channels = n_dim*4, attn_dim = 2)
        self.forth_u_net_layer = gMLPVision(image_size = LR_image_size//8, patch_size = 2, dim = n_dim*64, channels = n_dim*16, attn_dim = 2)

        if type_of_upsampler == 'conv_based_upsampler':
            self.fisrt_upsampling_layer = Upsampler(n_feats = n_dim)
            self.second_upsampling_layer = Upsampler(n_feats = n_dim*2, reduce_number_of_channels_in_half = True)
            self.third_upsampling_layer = Upsampler(n_feats = n_dim*2, reduce_number_of_channels_in_half = True)
            self.forth_upsampling_layer = Upsampler(n_feats = n_dim*2, reduce_number_of_channels_in_half = True)
            if Maintain_in_plane_Size == True:
                self.last_channel_reduce_layer = nn.Conv2d(n_dim, n_colors, kernel_size = 3, padding=1, stride=1)
            else:
                self.fifth_upsampling_layer = Upsampler(n_feats = n_dim, reduce_number_of_channels_in_half = False)
                self.last_channel_reduce_layer = nn.Conv2d(n_dim, n_colors, kernel_size = 3, padding=1, stride=1)
        elif type_of_upsampler == 'gMLP_based_upsampler':
            self.fisrt_upsampling_layer = nn.Sequential(*[
                gMLPVision(image_size = LR_image_size//16, patch_size = 1, dim = 64*n_dim, channels = 64*n_dim, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            self.second_upsampling_layer = nn.Sequential(*[
                gMLPVision(image_size = LR_image_size//8, patch_size = 1, dim = 16*n_dim, channels = 32*n_dim, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            self.third_upsampling_layer = nn.Sequential(*[
                gMLPVision(image_size = LR_image_size//4, patch_size = 1, dim = 4*n_dim, channels = 8*n_dim, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            self.forth_upsampling_layer = nn.Sequential(*[
                gMLPVision(image_size = LR_image_size//2, patch_size = 1, dim = n_dim, channels = 2*n_dim, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            if Maintain_in_plane_Size == True:
                self.last_channel_reduce_layer = gMLPVision(image_size = LR_image_size, patch_size = 1, dim = n_colors, channels = n_dim, attn_dim = 2)
            else:
                self.fifth_upsampling_layer = nn.Sequential(*[
                    gMLPVision(image_size = LR_image_size, patch_size = 1, dim = n_dim//4, channels = n_dim//4, attn_dim = 2),
                    nn.PixelShuffle(upscale_factor = 2)
                ])
                self.last_channel_reduce_layer = gMLPVision(image_size = LR_image_size*2, patch_size = 1, dim = n_colors, channels = n_dim//16, attn_dim = 2)
        else:
            raise ValueError("Not supported type of upsampler!")

    def forward(self, x):
        # U-Net framework.
        
        x1 = self.fisrt_u_net_layer(x)
        x2 = self.second_u_net_layer(x1)
        x3 = self.third_u_net_layer(x2)
        x4 = self.forth_u_net_layer(x3)

        y1 = self.fisrt_upsampling_layer(x4)
        y2 = self.second_upsampling_layer(tc.cat((x3, y1), dim = 1))
        y3 = self.third_upsampling_layer(tc.cat((x2, y2), dim = 1))
        y4 = self.forth_upsampling_layer(tc.cat((x1, y3), dim = 1))
        if Maintain_in_plane_Size == True:
            sr = self.last_channel_reduce_layer(y4)
        else:
            y5 = self.fifth_upsampling_layer(y4)
            sr = self.last_channel_reduce_layer(y5)
        """    
        x1 = self.fisrt_u_net_layer(x)
        x2 = self.second_u_net_layer(x1)
        x3 = self.third_u_net_layer(x2)

        y1 = self.second_upsampling_layer(x3)
        y2 = self.third_upsampling_layer(tc.cat((x2, y1), dim = 1))
        y3 = self.forth_upsampling_layer(tc.cat((x1, y2), dim = 1))
        if Maintain_in_plane_Size == True:
            sr = self.last_channel_reduce_layer(y3)
        else:
            y4 = self.fifth_upsampling_layer(y3)
            sr = self.last_channel_reduce_layer(y4)    
        """
        return sr


"gMLP based U-Net for Super Resolution MRI"
class U_Net_Based_MRI_SR_Transformer_Conv_MLP_2D(nn.Module):
    """
    U-Net framework in this code, consists of each gMLP as one layer in U-Net.
    """
    def __init__(self, args):
        super(U_Net_Based_MRI_SR_Transformer_Conv_MLP_2D, self).__init__()

        n_colors = args['n_colors'] # number of channels going of input of entire model.
        LR_image_size = args['LR_image_size'] # size of LR image.
        n_dim = args['n_dim'] # number of dimension we expect to use for gMLP.
        type_of_upsampler = args['type_of_upsampler']
        
        self.fisrt_u_net_layer = nn.Sequential(*[
            MultiScaleExtractor(n_colors, n_dim//4),
            gMLPVision(image_size = LR_image_size, patch_size = 2, dim = n_dim, channels = n_dim//4, attn_dim = 2)
            ])
        self.second_u_net_layer = nn.Sequential(*[
            MultiScaleExtractor(n_dim, n_dim),
            gMLPVision(image_size = LR_image_size//2, patch_size = 2, dim = n_dim*4, channels = n_dim, attn_dim = 2)
            ])
        self.third_u_net_layer = nn.Sequential(*[
            MultiScaleExtractor(n_dim*4, n_dim*4),
            gMLPVision(image_size = LR_image_size//4, patch_size = 2, dim = n_dim*16, channels = n_dim*4, attn_dim = 2)
            ])
        self.forth_u_net_layer = nn.Sequential(*[
            MultiScaleExtractor(n_dim*16, n_dim*16),
            gMLPVision(image_size = LR_image_size//8, patch_size = 2, dim = n_dim*64, channels = n_dim*16, attn_dim = 2)
            ])

        if type_of_upsampler == 'conv_based_upsampler':
            self.fisrt_upsampling_layer = Upsampler(n_feats = n_dim)
            self.second_upsampling_layer = Upsampler(n_feats = n_dim*2, reduce_number_of_channels_in_half = True)
            self.third_upsampling_layer = Upsampler(n_feats = n_dim*2, reduce_number_of_channels_in_half = True)
            self.forth_upsampling_layer = Upsampler(n_feats = n_dim*2, reduce_number_of_channels_in_half = True)
            if Maintain_in_plane_Size == True:
                self.last_channel_reduce_layer = nn.Conv2d(n_dim, n_colors, kernel_size = 3, padding=1, stride=1)
            else:
                self.fifth_upsampling_layer = Upsampler(n_feats = n_dim, reduce_number_of_channels_in_half = False)
                self.last_channel_reduce_layer = nn.Conv2d(n_dim, n_colors, kernel_size = 3, padding=1, stride=1)
        elif type_of_upsampler == 'gMLP_based_upsampler':
            self.fisrt_upsampling_layer = nn.Sequential(*[
                MultiScaleExtractor(n_dim*64, n_dim*64),
                gMLPVision(image_size = LR_image_size//16, patch_size = 1, dim = 64*n_dim, channels = 64*n_dim, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            self.second_upsampling_layer = nn.Sequential(*[
                MultiScaleExtractor(n_dim*32, n_dim*16),
                gMLPVision(image_size = LR_image_size//8, patch_size = 1, dim = 16*n_dim, channels = 16*n_dim, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            self.third_upsampling_layer = nn.Sequential(*[
                MultiScaleExtractor(n_dim*8, n_dim*4),
                gMLPVision(image_size = LR_image_size//4, patch_size = 1, dim = 4*n_dim, channels = 4*n_dim, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            self.forth_upsampling_layer = nn.Sequential(*[
                MultiScaleExtractor(n_dim*2, n_dim),
                gMLPVision(image_size = LR_image_size//2, patch_size = 1, dim = n_dim, channels = n_dim, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            if Maintain_in_plane_Size == True:
                self.last_channel_reduce_layer = gMLPVision(image_size = LR_image_size, patch_size = 1, dim = n_colors, channels = n_dim, attn_dim = 2)
            else:
                self.fifth_upsampling_layer = nn.Sequential(*[
                    MultiScaleExtractor(n_dim//4, n_dim//4),
                    gMLPVision(image_size = LR_image_size, patch_size = 1, dim = n_dim//4, channels = n_dim//4, attn_dim = 2),
                    nn.PixelShuffle(upscale_factor = 2)
                ])
#                self.last_channel_reduce_layer = gMLPVision(image_size = LR_image_size*2, patch_size = 1, dim = n_colors, channels = n_dim//16, attn_dim = 2)
        else:
            raise ValueError("Not supported type of upsampler!")

    def forward(self, x):
        # U-Net framework.
        
        x1 = self.fisrt_u_net_layer(x)
        x2 = self.second_u_net_layer(x1)
        x3 = self.third_u_net_layer(x2)
        x4 = self.forth_u_net_layer(x3)

        y1 = self.fisrt_upsampling_layer(x4)
        y2 = self.second_upsampling_layer(tc.cat((x3, y1), dim = 1))
        y3 = self.third_upsampling_layer(tc.cat((x2, y2), dim = 1))
        y4 = self.forth_upsampling_layer(tc.cat((x1, y3), dim = 1))
        if Maintain_in_plane_Size == True:
            sr = self.last_channel_reduce_layer(y4)
        else:
            sr = self.fifth_upsampling_layer(y4)
#            sr = self.last_channel_reduce_layer(y5)
        """    
        x1 = self.fisrt_u_net_layer(x)
        x2 = self.second_u_net_layer(x1)
        x3 = self.third_u_net_layer(x2)

        y1 = self.second_upsampling_layer(x3)
        y2 = self.third_upsampling_layer(tc.cat((x2, y1), dim = 1))
        y3 = self.forth_upsampling_layer(tc.cat((x1, y2), dim = 1))
        if Maintain_in_plane_Size == True:
            sr = self.last_channel_reduce_layer(y3)
        else:
            y4 = self.fifth_upsampling_layer(y3)
            sr = self.last_channel_reduce_layer(y4)    
        """
        return sr



"HR reference gMLP based U-Net for Super Resolution MRI"
class HR_Reference_U_Net_Based_MRI_SR_Transformer_MLP_2D(nn.Module):
    """
    U-Net framework in this code, consists of each gMLP as one layer in U-Net, with HR reference as auxiliary branch.
    """
    def __init__(self, args):
        super(HR_Reference_U_Net_Based_MRI_SR_Transformer_MLP_2D, self).__init__()

        n_colors = args['n_colors'] # number of channels going of input of entire model.
        LR_image_size = args['LR_image_size'] # size of LR image.
        type_of_upsampler = args['type_of_upsampler']
        self.HR_reference_framework = args['HR_reference_framework']

        if self.HR_reference_framework == 'gMLP_without_information_exchange':
            self.fisrt_u_net_layer = gMLPVision(image_size = LR_image_size, patch_size = 2, dim = 16, channels = n_colors, attn_dim = 2)
        elif self.HR_reference_framework == 'gMLP_with_information_exchange':
            self.fisrt_u_net_layer = gMLPVisionInfoExchange(image_size = LR_image_size, patch_size = 2, dim = 16, channels = n_colors, attn_dim = 2)
        self.second_u_net_layer = gMLPVision(image_size = LR_image_size//2, patch_size = 2, dim = 64, channels = 16, attn_dim = 2)
        self.third_u_net_layer = gMLPVision(image_size = LR_image_size//4, patch_size = 2, dim = 256, channels = 64, attn_dim = 2)
        self.forth_u_net_layer = gMLPVision(image_size = LR_image_size//8, patch_size = 2, dim = 1024, channels = 256, attn_dim = 2)

        if type_of_upsampler == 'conv_based_upsampler':
            self.fisrt_upsampling_layer = Upsampler(n_feats = 1024)
            self.second_upsampling_layer = Upsampler(n_feats = 256*2, reduce_number_of_channels_in_half = True)
            self.third_upsampling_layer = Upsampler(n_feats = 64*2, reduce_number_of_channels_in_half = True)
            self.forth_upsampling_layer = Upsampler(n_feats = 16*2, reduce_number_of_channels_in_half = True)
            if Maintain_in_plane_Size == True:
                self.last_channel_reduce_layer = nn.Conv2d(4, n_colors, kernel_size = 3, padding=1, stride=1)
            else:
                self.fifth_upsampling_layer = Upsampler(n_feats = 4, reduce_number_of_channels_in_half = False)
                self.last_channel_reduce_layer = nn.Conv2d(1, n_colors, kernel_size = 3, padding=1, stride=1)
        elif type_of_upsampler == 'gMLP_based_upsampler':
            self.fisrt_upsampling_layer = nn.Sequential(*[
                gMLPVision(image_size = LR_image_size//16, patch_size = 1, dim = 1024, channels = 1024, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            self.second_upsampling_layer = nn.Sequential(*[
                gMLPVision(image_size = LR_image_size//8, patch_size = 1, dim = 256, channels = 2*256, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            self.third_upsampling_layer = nn.Sequential(*[
                gMLPVision(image_size = LR_image_size//4, patch_size = 1, dim = 64, channels = 2*64, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            self.forth_upsampling_layer = nn.Sequential(*[
                gMLPVision(image_size = LR_image_size//2, patch_size = 1, dim = 16, channels = 2*16, attn_dim = 2),
                nn.PixelShuffle(upscale_factor = 2)
            ])
            if Maintain_in_plane_Size == True:
                self.last_channel_reduce_layer = gMLPVision(image_size = LR_image_size, patch_size = 1, dim = n_colors, channels = 4, attn_dim = 2)
            else:
                if self.HR_reference_framework == 'gMLP_without_information_exchange':
                    self.fifth_upsampling_layer_gMLP_layer = gMLPVision(image_size = LR_image_size, patch_size = 1, dim = 4, channels = 4, attn_dim = 2)
                elif self.HR_reference_framework == 'gMLP_with_information_exchange':
                    self.fifth_upsampling_layer_gMLP_layer = gMLPVisionInfoExchange(image_size = LR_image_size, patch_size = 1, dim = 4, channels = 4, attn_dim = 2)
                self.fifth_upsampling_layer_upsampling_layer = nn.PixelShuffle(upscale_factor = 2)
                self.last_channel_reduce_layer = gMLPVision(image_size = LR_image_size*2, patch_size = 1, dim = n_colors, channels = 1, attn_dim = 2)
        else:
            raise ValueError("Not supported type of upsampler!")

        if self.HR_reference_framework == 'gMLP_without_information_exchange':
            if Maintain_in_plane_Size == True:
                raise ValueError("Size of HR reference and LR are same, not supported yet!")
            else:
                self.first_layer_in_reference_branch_for_unet_encoder = gMLPVision(image_size = LR_image_size*2, patch_size = 2, dim = n_colors, channels = n_colors, attn_dim = 2)
                self.second_layer_in_reference_branch_for_unet_encoder = gMLPVision(image_size = LR_image_size, patch_size = 2, dim = 16, channels = n_colors, attn_dim = 2)
                self.last_layer_in_reference_branch_for_unet_decoder = gMLPVision(image_size = LR_image_size, patch_size = 1, dim = 4, channels = n_colors, attn_dim = 2)
        elif self.HR_reference_framework == 'gMLP_with_information_exchange':
            if Maintain_in_plane_Size == True:
                raise ValueError("Size of HR reference and LR are same, not supported yet!")
            else:
                self.first_layer_in_reference_branch_for_unet_encoder = gMLPVision(image_size = LR_image_size*2, patch_size = 2, dim = n_colors, channels = n_colors, attn_dim = 2)
                self.second_layer_in_reference_branch_for_unet_encoder = gMLPVisionInfoExchange(image_size = LR_image_size, patch_size = 2, dim = 16, channels = n_colors, attn_dim = 2)
                self.last_layer_in_reference_branch_for_unet_decoder = gMLPVisionInfoExchange(image_size = LR_image_size, patch_size = 1, dim = 4, channels = n_colors, attn_dim = 2)

    def forward(self, x, hr_reference):
        # First layer of MRI HR reference branch in U-Net framework.
        hr_reference1 = self.first_layer_in_reference_branch_for_unet_encoder(hr_reference)

        # MRI LR branch(main branch) in U-Net framework.
        if self.HR_reference_framework == 'gMLP_without_information_exchange':
            x1 = self.fisrt_u_net_layer(x)
            hr_reference2 = self.second_layer_in_reference_branch_for_unet_encoder(hr_reference1)
        elif self.HR_reference_framework == 'gMLP_with_information_exchange':
            x1, list_of_key_from_lr_info_first_downsampling_layer = self.fisrt_u_net_layer(x, injected_key = None)
            hr_reference2, _ = self.second_layer_in_reference_branch_for_unet_encoder(hr_reference1, list_of_key_from_lr_info_first_downsampling_layer)
        x2 = self.second_u_net_layer(x1 + hr_reference2)
        x3 = self.third_u_net_layer(x2)
        x4 = self.forth_u_net_layer(x3)

        y1 = self.fisrt_upsampling_layer(x4)
        y2 = self.second_upsampling_layer(tc.cat((x3, y1), dim = 1))
        y3 = self.third_upsampling_layer(tc.cat((x2, y2), dim = 1))
        y4 = self.forth_upsampling_layer(tc.cat((x1, y3), dim = 1))
        if Maintain_in_plane_Size == True:
            """ sr = self.last_channel_reduce_layer(y4) """
            sr = y4
        else:
            if self.HR_reference_framework == 'gMLP_without_information_exchange':
                y5 = self.fifth_upsampling_layer_gMLP_layer(y4)
                hr_reference3 = self.last_layer_in_reference_branch_for_unet_decoder(hr_reference1)
            elif self.HR_reference_framework == 'gMLP_with_information_exchange':
                y5, list_of_key_from_lr_info_fifth_upsampling_layer = self.fifth_upsampling_layer_gMLP_layer(y4, injected_key = [None]*2)
                hr_reference3, _ = self.last_layer_in_reference_branch_for_unet_decoder(hr_reference1, list_of_key_from_lr_info_fifth_upsampling_layer)
            y5 = self.fifth_upsampling_layer_upsampling_layer(y5 + hr_reference3)
            """ sr = self.last_channel_reduce_layer(y5) """
            sr = y5
        return sr



"Efficient Transformer based U-Net for Super Resolution MRI. See paper 2021.ResT:An Efficient Transformer for Visual Recognition for detail"
class Efficient_Transformer_Based_MRI_SR_Transformer_MLP_2D(nn.Module):
    """
    Input arguments:
        embed_dims: Stands for number of output channel from last effiecient transformer block and number of input channel for next effiecient transformer block.
        num_heads: Stands for how many heads are used in efficient multi-head self-attention block.
        depths: Stands for how many EfficientTransformerBlock are used in each stage. See figure 2 of paper:2021.ResT:An Efficient Transformer for Visual Recognition for detail(depth is L1, L2, L3, L4 in figure 2).
    """
    def __init__(self, args, embed_dims=[16, 64, 256, 1024, 1024, 256, 64, 16],
                 num_heads=[1, 2, 4, 8, 8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4, 4, 4, 4, 4], 
                 qkv_bias=False, qk_scale=None,
                 depths=[2, 2, 2, 2, 2, 2, 2, 2], sr_ratios=[8, 4, 2, 1, 1, 2, 4, 8],
                 norm_layer=nn.LayerNorm, apply_transform=True):
        super().__init__()
        self.depths = depths
        self.apply_transform = apply_transform
        self.type_of_upsampler = args['type_of_upsampler']

        # Efficient transformer block based encoder.
        self.stem = BasicStem(in_ch=args['n_colors'], out_ch=embed_dims[0], with_pos=True)  # See figure 2 of ResT paper for more detail regarding stem.

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True) # Patch embedding for stage 2.
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True) # Patch embedding for stage 3.
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True) # Patch embedding for stage 4.
        
        
        self.stage1 = nn.ModuleList([
            EfficientTransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])

        self.stage2 = nn.ModuleList([
            EfficientTransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform)
            for i in range(self.depths[1])])

        self.stage3 = nn.ModuleList([
            EfficientTransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform)
            for i in range(self.depths[2])])

        self.stage4 = nn.ModuleList([
            EfficientTransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform)
            for i in range(self.depths[3])])


        if self.type_of_upsampler == 'conv_based_upsampler':
            self.fisrt_upsampling_layer = Upsampler(n_feats = embed_dims[4])
            self.second_upsampling_layer = Upsampler(n_feats = embed_dims[5]*2, reduce_number_of_channels_in_half = True)
            self.third_upsampling_layer = Upsampler(n_feats = embed_dims[6]*2, reduce_number_of_channels_in_half = True)
            self.forth_upsampling_layer = Upsampler(n_feats = embed_dims[7]*2, reduce_number_of_channels_in_half = True)
            if Maintain_in_plane_Size == True:
                self.last_channel_reduce_layer = nn.Conv2d(embed_dims[7]//4, args['n_colors'], kernel_size = 3, padding=1, stride=1)
            else:
                self.fifth_upsampling_layer = Upsampler(n_feats = embed_dims[7]//4, reduce_number_of_channels_in_half = False)
                self.last_channel_reduce_layer = nn.Conv2d(embed_dims[7]//16, args['n_colors'], kernel_size = 3, padding=1, stride=1)
        else:
            # Efficient transformer block based decoder.
            self.patch_embed_upsampling_1 = PatchEmbed(patch_size=2, in_ch=embed_dims[4], out_ch=embed_dims[4], with_pos=True, for_upsampling = True) # Patch embedding for upsampling stage 1.
            self.patch_embed_upsampling_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[5]*2, out_ch=embed_dims[5], with_pos=True, for_upsampling = True) # Patch embedding for upsampling stage 2.
            self.patch_embed_upsampling_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[6]*2, out_ch=embed_dims[6], with_pos=True, for_upsampling = True) # Patch embedding for upsampling stage 3.
            self.patch_embed_upsampling_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[7]*2, out_ch=embed_dims[7], with_pos=True, for_upsampling = True) # Patch embedding for upsampling stage 4.
            self.patch_embed_upsampling_5 = PatchEmbed(patch_size=2, in_ch=embed_dims[7]//4, out_ch=embed_dims[7]//4, with_pos=True, for_upsampling = True) # Patch embedding for upsampling stage 5.
            
            self.pixel_shuffle_upsampler_1 = nn.PixelShuffle(upscale_factor = 2)
            self.pixel_shuffle_upsampler_2 = nn.PixelShuffle(upscale_factor = 2)
            self.pixel_shuffle_upsampler_3 = nn.PixelShuffle(upscale_factor = 2)
            self.pixel_shuffle_upsampler_4 = nn.PixelShuffle(upscale_factor = 2)
            self.pixel_shuffle_upsampler_5 = nn.PixelShuffle(upscale_factor = 2)

            self.upsampling_stage_1 = nn.ModuleList([
                EfficientTransformerBlock(embed_dims[4], num_heads[4], mlp_ratios[4], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[4], apply_transform=apply_transform)
                for i in range(self.depths[4])])

            self.upsampling_stage_2 = nn.ModuleList([
                EfficientTransformerBlock(embed_dims[5], num_heads[5], mlp_ratios[5], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[5], apply_transform=apply_transform)
                for i in range(self.depths[5])])

            self.upsampling_stage_3 = nn.ModuleList([
                EfficientTransformerBlock(embed_dims[6], num_heads[6], mlp_ratios[6], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[6], apply_transform=apply_transform)
                for i in range(self.depths[6])])

            self.upsampling_stage_4 = nn.ModuleList([
                EfficientTransformerBlock(embed_dims[7], num_heads[7], mlp_ratios[7], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[7], apply_transform=apply_transform)
                for i in range(self.depths[7])])

            self.upsampling_stage_5 = nn.ModuleList([
                EfficientTransformerBlock(embed_dims[7]//4, num_heads[7], mlp_ratios[7], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[7], apply_transform=apply_transform)
                for i in range(self.depths[7])])

    def forward(self, x):
        lr = x.clone()
        x = self.stem(x)
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        # encoder(downsampling) stage 1
        for blk in self.stage1: # blk stands for every efficient transformer block in currect stage.
            x = blk(x, H, W)
        x_downsampling_1 = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # encoder(downsampling) stage 2
        x, (H, W) = self.patch_embed_2(x_downsampling_1)
        for blk in self.stage2: # blk stands for every efficient transformer block in currect stage.
            x = blk(x, H, W)
        x_downsampling_2 = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # encoder(downsampling) stage 3
        x, (H, W) = self.patch_embed_3(x_downsampling_2)
        for blk in self.stage3: # blk stands for every efficient transformer block in currect stage.
            x = blk(x, H, W)
        x_downsampling_3 = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # encoder(downsampling) stage 4
        x, (H, W) = self.patch_embed_4(x_downsampling_3)
        for blk in self.stage4: # blk stands for every efficient transformer block in currect stage.
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        

        if self.type_of_upsampler == 'conv_based_upsampler':

            x_upampling_1 = self.fisrt_upsampling_layer(x)
            x_upampling_2 = self.second_upsampling_layer(tc.cat((x_upampling_1,x_downsampling_3),dim=1))
            x_upampling_3 = self.third_upsampling_layer(tc.cat((x_upampling_2,x_downsampling_2),dim=1))
            x_upampling_4 = self.forth_upsampling_layer(tc.cat((x_upampling_3,x_downsampling_1),dim=1))
            if Maintain_in_plane_Size == True:
                sr = self.last_channel_reduce_layer(x_upampling_4)
            else:
                x_upampling_5 = self.fifth_upsampling_layer(x_upampling_4)
                sr = self.last_channel_reduce_layer(x_upampling_5)

            
        else:            
            # decoder(upampling) stage 1
            x, (H, W) = self.patch_embed_upsampling_1(x)
            for blk in self.upsampling_stage_1: # blk stands for every efficient transformer block in currect stage.
                x = blk(x, H, W)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            x_upampling_1 = self.pixel_shuffle_upsampler_1(x)

            # decoder(upampling) stage 2
            x, (H, W) = self.patch_embed_upsampling_2(tc.cat((x_upampling_1, x_downsampling_3), dim = 1))
            for blk in self.upsampling_stage_2: # blk stands for every efficient transformer block in currect stage.
                x = blk(x, H, W)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            x_upampling_2 = self.pixel_shuffle_upsampler_2(x)

            # decoder(upampling) stage 3
            x, (H, W) = self.patch_embed_upsampling_3(tc.cat((x_upampling_2, x_downsampling_2), dim = 1))
            for blk in self.upsampling_stage_3: # blk stands for every efficient transformer block in currect stage.
                x = blk(x, H, W)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            x_upampling_3 = self.pixel_shuffle_upsampler_3(x)

            # decoder(upampling) stage 4
            x, (H, W) = self.patch_embed_upsampling_4(tc.cat((x_upampling_3, x_downsampling_1), dim = 1))
            for blk in self.upsampling_stage_4: # blk stands for every efficient transformer block in currect stage.
                x = blk(x, H, W)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            x_upampling_4 = self.pixel_shuffle_upsampler_4(x)

            """ x_upampling_4 = x_upampling_4 + lr  # If we want to just recover the "residual of sr" rather than the sr directly, then need this line of code. """

            # decoder(upampling) stage 5
            x, (H, W) = self.patch_embed_upsampling_5(x_upampling_4)
            for blk in self.upsampling_stage_5: # blk stands for every efficient transformer block in currect stage.
                x = blk(x, H, W)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            sr = self.pixel_shuffle_upsampler_5(x)
            

        return sr
    
    
"Efficient Transformer based U-Net for Super Resolution MRI. See paper 2021.ResT:An Efficient Transformer for Visual Recognition for detail"
class Efficient_Transformer_Based_MRI_SR_Conv_Transformer_MLP_2D(nn.Module):
    """
    Input arguments:
        embed_dims: Stands for number of output channel from last effiecient transformer block and number of input channel for next effiecient transformer block.
        num_heads: Stands for how many heads are used in efficient multi-head self-attention block.
        depths: Stands for how many EfficientTransformerBlock are used in each stage. See figure 2 of paper:2021.ResT:An Efficient Transformer for Visual Recognition for detail(depth is L1, L2, L3, L4 in figure 2).
    """
    def __init__(self, args, embed_dims=[16, 64, 256, 1024, 1024, 256, 64, 16],
                 num_heads=[1, 2, 4, 8, 8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4, 4, 4, 4, 4], 
                 qkv_bias=False, qk_scale=None,
                 depths=[2, 2, 2, 2, 2, 2, 2, 2], sr_ratios=[8, 4, 2, 1, 1, 2, 4, 8],
                 norm_layer=nn.LayerNorm, apply_transform=True):
        super().__init__()
        self.depths = depths
        self.apply_transform = apply_transform
        self.type_of_upsampler = args['type_of_upsampler']

        # Efficient transformer block based encoder.
        self.stem = BasicStem(in_ch=args['n_colors'], out_ch=embed_dims[0], with_pos=True)  # See figure 2 of ResT paper for more detail regarding stem.

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True) # Patch embedding for stage 2.
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True) # Patch embedding for stage 3.
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True) # Patch embedding for stage 4.

        self.stage1 = nn.ModuleList([
            EfficientTransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])
        
        self.MultiScaleExtractor_encoder_stage2 = nn.Sequential(*[MultiScaleExtractor(embed_dims[0], embed_dims[0])])
        self.stage2 = nn.ModuleList([
            EfficientTransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform)
            for i in range(self.depths[1])])
        
        self.MultiScaleExtractor_encoder_stage3 = nn.Sequential(*[MultiScaleExtractor(embed_dims[1], embed_dims[1])])
        self.stage3 = nn.ModuleList([
            EfficientTransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform)
            for i in range(self.depths[2])])

        self.MultiScaleExtractor_encoder_stage4 = nn.Sequential(*[MultiScaleExtractor(embed_dims[2], embed_dims[2])])
        self.stage4 = nn.ModuleList([
            EfficientTransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform)
            for i in range(self.depths[3])])


        if self.type_of_upsampler == 'conv_based_upsampler':
            self.fisrt_upsampling_layer = Upsampler(n_feats = embed_dims[4])
            self.second_upsampling_layer = Upsampler(n_feats = embed_dims[5]*2, reduce_number_of_channels_in_half = True)
            self.third_upsampling_layer = Upsampler(n_feats = embed_dims[6]*2, reduce_number_of_channels_in_half = True)
            self.forth_upsampling_layer = Upsampler(n_feats = embed_dims[7]*2, reduce_number_of_channels_in_half = True)
            if Maintain_in_plane_Size == True:
                self.last_channel_reduce_layer = nn.Conv2d(embed_dims[7]//4, args['n_colors'], kernel_size = 3, padding=1, stride=1)
            else:
                self.fifth_upsampling_layer = Upsampler(n_feats = embed_dims[7]//4, reduce_number_of_channels_in_half = False)
                self.last_channel_reduce_layer = nn.Conv2d(embed_dims[7]//16, args['n_colors'], kernel_size = 3, padding=1, stride=1)
        else:
            # Efficient transformer block based decoder.
            self.MultiScaleExtractor_dencoder_stage1 = nn.Sequential(*[MultiScaleExtractor(embed_dims[4], embed_dims[4])])
            self.MultiScaleExtractor_dencoder_stage2 = nn.Sequential(*[MultiScaleExtractor(embed_dims[5]*2, embed_dims[5])])
            self.MultiScaleExtractor_dencoder_stage3 = nn.Sequential(*[MultiScaleExtractor(embed_dims[6]*2, embed_dims[6])])
            self.MultiScaleExtractor_dencoder_stage4 = nn.Sequential(*[MultiScaleExtractor(embed_dims[7]*2, embed_dims[7])])
            self.MultiScaleExtractor_dencoder_stage5 = nn.Sequential(*[MultiScaleExtractor(embed_dims[7]//4, embed_dims[7]//4)])
            self.patch_embed_upsampling_1 = PatchEmbed(patch_size=2, in_ch=embed_dims[4], out_ch=embed_dims[4], with_pos=True, for_upsampling = True) # Patch embedding for upsampling stage 1.
            self.patch_embed_upsampling_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[5], out_ch=embed_dims[5], with_pos=True, for_upsampling = True) # Patch embedding for upsampling stage 2.
            self.patch_embed_upsampling_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[6], out_ch=embed_dims[6], with_pos=True, for_upsampling = True) # Patch embedding for upsampling stage 3.
            self.patch_embed_upsampling_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[7], out_ch=embed_dims[7], with_pos=True, for_upsampling = True) # Patch embedding for upsampling stage 4.
            self.patch_embed_upsampling_5 = PatchEmbed(patch_size=2, in_ch=embed_dims[7]//4, out_ch=embed_dims[7]//4, with_pos=True, for_upsampling = True) # Patch embedding for upsampling stage 5.
            
            self.pixel_shuffle_upsampler_1 = nn.PixelShuffle(upscale_factor = 2)
            self.pixel_shuffle_upsampler_2 = nn.PixelShuffle(upscale_factor = 2)
            self.pixel_shuffle_upsampler_3 = nn.PixelShuffle(upscale_factor = 2)
            self.pixel_shuffle_upsampler_4 = nn.PixelShuffle(upscale_factor = 2)
            self.pixel_shuffle_upsampler_5 = nn.PixelShuffle(upscale_factor = 2)

            self.upsampling_stage_1 = nn.ModuleList([
                EfficientTransformerBlock(embed_dims[4], num_heads[4], mlp_ratios[4], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[4], apply_transform=apply_transform)
                for i in range(self.depths[4])])

            self.upsampling_stage_2 = nn.ModuleList([
                EfficientTransformerBlock(embed_dims[5], num_heads[5], mlp_ratios[5], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[5], apply_transform=apply_transform)
                for i in range(self.depths[5])])

            self.upsampling_stage_3 = nn.ModuleList([
                EfficientTransformerBlock(embed_dims[6], num_heads[6], mlp_ratios[6], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[6], apply_transform=apply_transform)
                for i in range(self.depths[6])])

            self.upsampling_stage_4 = nn.ModuleList([
                EfficientTransformerBlock(embed_dims[7], num_heads[7], mlp_ratios[7], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[7], apply_transform=apply_transform)
                for i in range(self.depths[7])])

            self.upsampling_stage_5 = nn.ModuleList([
                EfficientTransformerBlock(embed_dims[7]//4, num_heads[7], mlp_ratios[7], qkv_bias, qk_scale, norm_layer=norm_layer, sr_ratio=sr_ratios[7], apply_transform=apply_transform)
                for i in range(self.depths[7])])

    def forward(self, x):
        lr = x.clone()
        x = self.stem(x)
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        # encoder(downsampling) stage 1
        for blk in self.stage1: # blk stands for every efficient transformer block in currect stage.
            x = blk(x, H, W)
        x_downsampling_1 = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # encoder(downsampling) stage 2
        x, (H, W) = self.patch_embed_2(self.MultiScaleExtractor_encoder_stage2(x_downsampling_1))
        for blk in self.stage2: # blk stands for every efficient transformer block in currect stage.
            x = blk(x, H, W)
        x_downsampling_2 = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # encoder(downsampling) stage 3
        x, (H, W) = self.patch_embed_3(self.MultiScaleExtractor_encoder_stage3(x_downsampling_2))
        for blk in self.stage3: # blk stands for every efficient transformer block in currect stage.
            x = blk(x, H, W)
        x_downsampling_3 = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # encoder(downsampling) stage 4
        x, (H, W) = self.patch_embed_4(self.MultiScaleExtractor_encoder_stage4(x_downsampling_3))
        for blk in self.stage4: # blk stands for every efficient transformer block in currect stage.
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        

        if self.type_of_upsampler == 'conv_based_upsampler':

            x_upampling_1 = self.fisrt_upsampling_layer(x)
            x_upampling_2 = self.second_upsampling_layer(tc.cat((x_upampling_1,x_downsampling_3),dim=1))
            x_upampling_3 = self.third_upsampling_layer(tc.cat((x_upampling_2,x_downsampling_2),dim=1))
            x_upampling_4 = self.forth_upsampling_layer(tc.cat((x_upampling_3,x_downsampling_1),dim=1))
            if Maintain_in_plane_Size == True:
                sr = self.last_channel_reduce_layer(x_upampling_4)
            else:
                x_upampling_5 = self.fifth_upsampling_layer(x_upampling_4)
                sr = self.last_channel_reduce_layer(x_upampling_5)

            
        else:            
            # decoder(upampling) stage 1
            x, (H, W) = self.patch_embed_upsampling_1(self.MultiScaleExtractor_dencoder_stage1(x))
            for blk in self.upsampling_stage_1: # blk stands for every efficient transformer block in currect stage.
                x = blk(x, H, W)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            x_upampling_1 = self.pixel_shuffle_upsampler_1(x)

            # decoder(upampling) stage 2
            x, (H, W) = self.patch_embed_upsampling_2(self.MultiScaleExtractor_dencoder_stage2(tc.cat((x_upampling_1, x_downsampling_3), dim = 1)))
            for blk in self.upsampling_stage_2: # blk stands for every efficient transformer block in currect stage.
                x = blk(x, H, W)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            x_upampling_2 = self.pixel_shuffle_upsampler_2(x)

            # decoder(upampling) stage 3
            x, (H, W) = self.patch_embed_upsampling_3(self.MultiScaleExtractor_dencoder_stage3(tc.cat((x_upampling_2, x_downsampling_2), dim = 1)))
            for blk in self.upsampling_stage_3: # blk stands for every efficient transformer block in currect stage.
                x = blk(x, H, W)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            x_upampling_3 = self.pixel_shuffle_upsampler_3(x)

            # decoder(upampling) stage 4
            x, (H, W) = self.patch_embed_upsampling_4(self.MultiScaleExtractor_dencoder_stage4(tc.cat((x_upampling_3, x_downsampling_1), dim = 1)))
            for blk in self.upsampling_stage_4: # blk stands for every efficient transformer block in currect stage.
                x = blk(x, H, W)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            x_upampling_4 = self.pixel_shuffle_upsampler_4(x)

            """ x_upampling_4 = x_upampling_4 + lr  # If we want to just recover the "residual of sr" rather than the sr directly, then need this line of code. """

            # decoder(upampling) stage 5
            x, (H, W) = self.patch_embed_upsampling_5(self.MultiScaleExtractor_dencoder_stage5(x_upampling_4))
            for blk in self.upsampling_stage_5: # blk stands for every efficient transformer block in currect stage.
                x = blk(x, H, W)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            sr = self.pixel_shuffle_upsampler_5(x)
            

        return sr    


    
device=tc.device("cuda" if use_cuda else "cpu")

if Freeze_random_seed == True:
    tc.manual_seed(args['seed'])
#    tc.backend.cudnn.deterministic = True
#    tc.backend.cudnn.benchmark = False
    print('Seed is frozen!')

if args['use_HR_reference'] == False:
    if args['basic_block'] == 'efficient_transformer':
        our_model_mri_sr_2d = Efficient_Transformer_Based_MRI_SR_Transformer_MLP_2D(args)
    elif args['basic_block'] == 'gMLP':
        our_model_mri_sr_2d = U_Net_Based_MRI_SR_Transformer_MLP_2D(args)
    elif args['basic_block'] == 'MultiScaleExtrctor_gMLP':
        our_model_mri_sr_2d = U_Net_Based_MRI_SR_Transformer_Conv_MLP_2D(args)
    elif args['basic_block'] == 'MultiScaleExtrctor_ResT':
        our_model_mri_sr_2d = Efficient_Transformer_Based_MRI_SR_Conv_Transformer_MLP_2D(args)
else:   # args['use_HR_reference'] == True:
    if args['basic_block'] == 'efficient_transformer':
        pass
    elif args['basic_block'] == 'gMLP':
        our_model_mri_sr_2d = HR_Reference_U_Net_Based_MRI_SR_Transformer_MLP_2D(args)
        
        
if Use_saved_model == True:
    parameter_file = open(os.path.join(folder_saved_network, 'min_validation_loss_network_parameter.pkl'), 'rb')
    min_validation_loss_model_wts = pickle.load(parameter_file)
    parameter_file.close()
    our_model_mri_sr_2d.load_state_dict(min_validation_loss_model_wts)
    print("Saved model loaded")
    
# Weight initialization using He initialization.
""" for m in our_rcan_mri_sr_2d.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in') """

if tc.cuda.device_count()>1:
    our_model_mri_sr_2d=nn.DataParallel(our_model_mri_sr_2d)
our_model_mri_sr_2d.to(device)

print('this is our model: ', our_model_mri_sr_2d)

if Use_Feature_Map_Loss == True:
    feature_extractor = FeatureExtractor().to(device)
    print('this is our FeatureExtractor: ', feature_extractor)
if Use_kspace_loss == True:
    fft_k_space = FFT_K_SPACE().to(device)
    print('this is our FFT_K_SPACE: ', fft_k_space)


"""""""""""""""""""""""""""""""""""""""
4. Setup optimization algorithm part
"""""""""""""""""""""""""""""""""""""""
"set an optimizer"
if args['optimizer'] == 'look_ahead':
    base_opt = opt.Adam(our_model_mri_sr_2d.parameters(), lr=1e-3, betas=(0.9, 0.999)) #----- use Adam algorithm as based optimizer A
    optimizer = lookahead.Lookahead(base_opt, k=5, alpha=0.5) # Initialize Lookahead
elif args['optimizer'] == 'Adam':
    optimizer = opt.Adam(our_model_mri_sr_2d.parameters(), lr = args['initial_learning_rate_after_warm_up'], eps = 1e-08, weight_decay = 1e-5)    #----- use Adam algorithm for all parameters of our_classifier
elif args['optimizer'] == 'SGD_with_momentum':
    optimizer = opt.SGD(our_model_mri_sr_2d.parameters(), lr = args['initial_learning_rate_after_warm_up'], momentum=0.9, weight_decay = 1e-9)    #----- use SGD algorithm for all parameters of our_lenet, by learning rate 0.01 and Momentum is 0.9
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

loss_function_SmoothL1 = nn.SmoothL1Loss().to(device)       #----- smooth L1 loss

loss_function_L1 = nn.L1Loss().to(device)       #----- L1 loss

loss_function_Charbonnier = L1_Charbonnier_Loss().to(device)        #----- L1 Charbonnier loss

# loss_function_CE = nn.CrossEntropyLoss().to(device)

if Use_ssim_map == True:
    SSIM_function = pytorch_ssim_map.SSIM().to(device)       #----- ssim map calculation
else:
    SSIM_function = pytorch_ssim_l1_org.SSIM().to(device)       #----- ssim calculation

if Use_Negative_TV_Loss == True:
    negative_tv_loss = NegativeTVLoss(negative_tv_loss_weight = args_loss_weight['negative_total_variation_weight']).to(device)

if Use_Negative_Trace_Loss == True:
    negative_trace_loss = NegativeTraceLoss(negative_trace_loss_weight = args_loss_weight['negative_trace_weight']).to(device)

# =============================================================================
# print('The loss function is L1Loss')
# loss_function = nn.L1Loss(size_average = False).to(device) 
# =============================================================================
# =============================================================================
# print('The loss function is CrossEntropyLoss')
# loss_function = nn.CrossEntropyLoss().to(device)        #----- here use cross entropy loss
# =============================================================================


"""""""""""""""""""""""""""
5. Train the U_Net_Based_MRI_SR_Transformer_MLP_2D part
"""""""""""""""""""""""""""
"Train the U_Net_Based_MRI_SR_Transformer_MLP_2D"
tc.set_num_threads(10)  #----- Sets the number of OpenMP threads used for parallelizing CPU operations

best_ssim = 0.0 # Initialization of best_ssim = 0.0
best_psnr = 0.0
min_validation_loss = 0.0
best_ssim_epoch = 0
best_psnr_epoch = 0
min_validation_loss_epoch = 0

f = open(os.path.join(folder_log_path, 'log.txt'), 'w')
for epoch in range(EPOCH_NUM):
    "Set training mode"
    our_model_mri_sr_2d.train()
# =============================================================================
#     print('This is the ', epoch, ' epoch')
# =============================================================================
    running_loss = 0.0

    batch_number_test = 0
    pixel_wise_loss_test = []
    k_space_freq_loss_test = []
    ssim_loss_test = []
    gradient_img_loss_test = []
    ssim_test = []
    psnr_test = []

    batch_number_training = 0
    pixel_wise_loss_training = []
    k_space_freq_loss_training = []
    ssim_loss_training = []
    gradient_img_loss_training = []
    loss_training = []
    ssim_training = []
    psnr_training = []
    
    batch_with_nan = []
    
    if Use_Feature_Map_Loss == True:
        feature_map_loss_test = 0.0
        feature_map_loss_training = 0.0

    if Use_Gram_Matrix_L1_Loss == True:
        gram_similarity_between_img_loss_training = 0.0
        gram_similarity_between_img_loss_test = 0.0
    
    if Use_Negative_TV_Loss == True:
        negative_total_variation_for_img_loss_training = 0.0
        negative_total_variation_for_img_loss_test = 0.0
    
    if Use_Negative_Trace_Loss == True:
        negative_trace_for_img_loss_training = 0.0
        negative_trace_for_img_loss_test = 0.0

    k_space_branch_k_space_loss_training = 0.0
    k_space_branch_k_space_loss_test = 0.0
    gradient_grad_loss_training = 0.0
    gradient_grad_loss_test = 0.0
    wavelets_high_frequency_components_branch_high_frequency_loss_training = 0.0
    wavelets_high_frequency_components_branch_high_frequency_loss_test = 0.0
    
    "Save the training loss for each epoch"
    if (epoch == 0):
#        f = open(os.path.join(folder_log_path, 'log.txt'), 'w')
        f.write('Code Version: 1.0.1\n')
        f.write('The configuration of parameters:\n')
        f.write('batch_size is: %d\n' % batch_size)
        f.write('EPOCH_NUM is: %d\n' % EPOCH_NUM)
        f.write('Maintain_in_plane_Size is: %s\n' % Maintain_in_plane_Size)
        f.write('Freeze_random_seed is: %s\n' % Freeze_random_seed)
        f.write('Use_saved_model is: %s\n' % Use_saved_model)
        f.write('Use_kspace_Loss is: %s\n' % Use_kspace_loss)
        f.write('Use_SSIM_L1_Loss is: %s\n' % Use_SSIM_L1_Loss)
        f.write('Use_Gradient_Map_L1_Loss is: %s\n' % Use_Gradient_Map_L1_Loss)
        f.write('Use_Gram_Matrix_L1_Loss is: %s\n' % Use_Gram_Matrix_L1_Loss)
        f.write('Use_Negative_TV_Loss is: %s\n' % Use_Negative_TV_Loss)
        f.write('Use_Negative_Trace_Loss is: %s\n' % Use_Negative_Trace_Loss)
        f.write('Amplify_Small_Value_In_Gradient_Map: %s\n' % Amplify_Small_Value_In_Gradient_Map)
        f.write('Amplify_High_Frequency_Value_In_K_Space_Loss: %s\n' % Amplify_High_Frequency_Value_In_K_Space_Loss)
        f.write('Use_ssim_map is: %s\n' % Use_ssim_map)
        f.write('args is: %s\n' % args)
        f.write('args of loss weight is: %s\n' % args_loss_weight)
        f.write('------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        f.write(' \n')

    # If using warm up, call the scheduler of warm up now
    if args['use_learning_rate_warm_up'] == True:
        scheduler.step(epoch)
    learning_rate = optimizer.param_groups[0]['lr']
    print('learning rate for epoch %d is : %f' % (epoch, optimizer.param_groups[0]['lr']))
    
    optimizer.zero_grad()
    
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
#        optimizer.zero_grad()
        # print('The optimizer has been cleared' )

        """
        if args['conv_layer_type'] == 'deformable_conv':
            tc.cuda.empty_cache()
        """
        
        if args['use_HR_reference'] == False:
            "load input data"
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            # print('The data have been loaded' )

            "forward prop"
            # outputs = our_resnext(inputs).double() #-- numpy arrays are 64-bit floating point and will be converted to torch.DoubleTensor standardly. Now, if you use them with your model, you'll need to make sure that your model parameters are also Double
            img_outputs = our_model_mri_sr_2d(inputs) #-- or using default float as type, however remember to cast the input from Double to Float            
            # print(outputs.size())
            # print('the forward pass has been went')
        elif args['use_HR_reference'] == True:
            "load input data"
            inputs, labels, references = data
            inputs, labels, references = Variable(inputs).to(device), Variable(labels).to(device), Variable(references).to(device)
            # print('The data have been loaded' )

            "forward prop"
            # outputs = our_resnext(inputs).double() #-- numpy arrays are 64-bit floating point and will be converted to torch.DoubleTensor standardly. Now, if you use them with your model, you'll need to make sure that your model parameters are also Double
            img_outputs = our_model_mri_sr_2d(inputs, references) #-- or using default float as type, however remember to cast the input from Double to Float            
            # print(outputs.size())
            # print('the forward pass has been went')

        if Use_Feature_Map_Loss == True:
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
        
        if Use_kspace_loss == True:
            SR_freq = fft_k_space(img_outputs)

            HR_freq = fft_k_space(labels)

        "calculate the gradients for all Variables during back prop"
        "vgg loss + pixel MSE loss + fft frequency loss, and we use weight_decay in Adam so that is L2 regularization"
        if Use_Feature_Map_Loss == True:
            feature_map_loss = args_loss_weight['feature_map_weight']*loss_function_MSE(SR_features, HR_features)
            feature_map_loss_training.append(feature_map_loss.item())    # Only save the value of feature_map_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage
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
        pixel_wise_loss_training.append(pixel_wise_loss.item())    # Only save the value of pixel_wise_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage 
#            print("pixel_wise_loss: ", pixel_wise_loss)
        if Use_kspace_loss == True:
            if Amplify_High_Frequency_Value_In_K_Space_Loss == True:
                k_space_freq_loss = args_loss_weight['k_space_weight']*(loss_function_MSE(
                    create_2d_Gaussian_weights(window_size = SR_freq.shape[2], num_of_samples = SR_freq.shape[0], channel = SR_freq.shape[1]).to(device)*SR_freq.real, 
                    create_2d_Gaussian_weights(window_size = HR_freq.shape[2], num_of_samples = HR_freq.shape[0], channel = HR_freq.shape[1]).to(device)*HR_freq.real) + 
                    loss_function_MSE(
                        create_2d_Gaussian_weights(window_size = SR_freq.shape[2], num_of_samples = SR_freq.shape[0], channel = SR_freq.shape[1]).to(device)*SR_freq.imag, 
                        create_2d_Gaussian_weights(window_size = HR_freq.shape[2], num_of_samples = HR_freq.shape[0], channel = HR_freq.shape[1]).to(device)*HR_freq.imag))
            else:
                k_space_freq_loss = args_loss_weight['k_space_weight']*(loss_function_MSE(SR_freq.real, HR_freq.real) + loss_function_MSE(SR_freq.imag, HR_freq.imag))
            k_space_freq_loss_training.append(k_space_freq_loss.item())
#            print(loss_function_MSE(SR_freq[:,:,:,:,0], HR_freq[:,:,:,:,0]))
#            print(loss_function_MSE(SR_freq[:,:,:,:,1], HR_freq[:,:,:,:,1]))
#            print("k_space_freq_loss: ", k_space_freq_loss.item())

#        HR_ssim_weighted, HR_ssim = SSIM_function(labels, labels)
#        SR_ssim_weighted, SR_ssim = SSIM_function(img_outputs, labels)
        HR_ssim = SSIM_function(labels, labels)
        SR_ssim = SSIM_function(img_outputs, labels)

        if Use_SSIM_L1_Loss == True:
            ssim_loss = args_loss_weight['ssim_weight']*loss_function_L1(SR_ssim.pow(args_loss_weight['ssim_component_weight']), HR_ssim.pow(args_loss_weight['ssim_component_weight']))
            ssim_loss_training.append(ssim_loss.item())    # Only save the value of ssim_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage 
        ssim_training.append(SR_ssim.item())    # Accumulation of SR_ssim in training over all batches in one epoch, will be used to calculate the average value of SR SSIM for one epoch.
        psnr_training.append(calc_psnr_for_mri_image(img_outputs, labels).item())
        
        if Use_Gradient_Map_L1_Loss == True:
            gradient_img_loss = args_loss_weight['gradient_img_weight']*loss_function_L1(calculate_gradient_map(args['n_colors'], img_outputs), calculate_gradient_map(args['n_colors'], labels))
            gradient_img_loss_training.append(gradient_img_loss.item())  # Only save the value of gradient_img_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage

        "TODO: The coefficient of this loss function needs to be adjusted"
        if Use_Gram_Matrix_L1_Loss == True:
            gram_similarity_between_img_loss = args_loss_weight['gram_similarity_weight']*loss_function_L1(calculate_gram_matrix(img_outputs), calculate_gram_matrix(labels))
            gram_similarity_between_img_loss_training.append(gram_similarity_between_img_loss.item())    # Only save the value of gram_similarity_between_img_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage

        if Use_Negative_TV_Loss == True:
            negative_total_variation_for_img_loss = negative_tv_loss(img_outputs)
            negative_total_variation_for_img_loss_training.append(negative_total_variation_for_img_loss.item())    # Only save the value of gram_similarity_between_img_loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage
        
        if Use_Negative_Trace_Loss == True:
            negative_trace_for_img_loss = negative_trace_loss(img_outputs, labels)
            negative_trace_for_img_loss_training.append(negative_trace_for_img_loss.item())


        if tc.isnan(SR_ssim) or tc.isnan(HR_ssim):
            batch_with_nan.append(i)
            continue

#            print('gradient_loss: ', gradient_map_loss)

#            loss = pixel_wise_loss + ssim_loss
        loss = pixel_wise_loss
        
        if Use_Feature_Map_Loss == True:
            loss = loss + feature_map_loss

#            if ssim_loss < 0.5:
#                loss = ssim_loss + feature_map_loss + pixel_wise_loss
#                print('ssim_loss')
#            else:
#                loss = pixel_wise_loss + feature_map_loss
        if Use_kspace_loss == True:
            if tc.isnan(k_space_freq_loss) != 1:
                loss = loss + k_space_freq_loss
        
        if Use_SSIM_L1_Loss == True:
            if tc.isnan(ssim_loss) != 1:
                loss = loss + ssim_loss

        if Use_Gradient_Map_L1_Loss == True:
            if tc.isnan(gradient_img_loss) != 1:
                loss = loss + gradient_img_loss

        if Use_Gram_Matrix_L1_Loss == True:
            loss = loss + gram_similarity_between_img_loss
        
        if Use_Negative_TV_Loss == True:
            loss = loss + negative_total_variation_for_img_loss
        
        if Use_Negative_Trace_Loss == True:
            loss = loss + negative_trace_for_img_loss


#            print('loss: ', loss)
#            loss = feature_map_loss + pixel_wise_loss + k_space_freq_loss
            # print('the loss has been checked')


            "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"
#            if tc.isnan(loss) == 1: #- loss == 'NaN':
#                break
            "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"

        loss_training.append(loss.item())  # Only save the value of loss(rather than saving the entire graph), otherwise the GPU memory may not be enough for usage    
        loss = loss / accumulation_steps
        "back prop"
        loss.backward()
            # print('the backward pass has gone')
        
        if (i+1) % accumulation_steps == 0:             
            optimizer.step()                            
            optimizer.zero_grad()
#        "update all Variables by using newly fetched gradients"
#        optimizer.step() 
        """print('learning rate: %f' % (optimizer.param_groups[0]['lr']))"""
        # print('the all Variables have been updated')

        "print log info"
        running_loss += loss.data
        if i % 1000 == 0 and i!=0: #----- print log info every 1000 batch
            if i == 0:
                print('[%d, %5d] loss: %.3f' \
                      % (epoch, i, running_loss))
            else:
                print('[%d, %5d] loss: %.3f' \
                      % (epoch, i, running_loss / 1000))
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

    
    learning_rate = optimizer.param_groups[0]['lr']
    
    # If NOT using warm up, call the normal scheduler now
    if args['use_learning_rate_warm_up'] == False:
#        print('learning rate for epoch %d is : %f' % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()
#        print('learning rate for next epoch is : %f' % (optimizer.param_groups[0]['lr']))

    batch_number_training = i+1 # Calculate for the current epoch, how many batches are used.

    "Set evaluation Mode"    
    our_model_mri_sr_2d.eval()    
    with tc.no_grad():
        for i, data in enumerate(validationloader, 0):

            if args['use_HR_reference'] == False:
                "load input data"
                inputs, labels = data
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

                SR_img_test = our_model_mri_sr_2d(inputs)
            elif args['use_HR_reference'] == True:
                "load input data"
                inputs, labels, references = data
                inputs, labels, references = Variable(inputs).to(device), Variable(labels).to(device), Variable(references).to(device)

                SR_img_test = our_model_mri_sr_2d(inputs, references)

            if Use_Feature_Map_Loss == True:
                SR_test_copies = tc.cat((SR_img_test, SR_img_test, SR_img_test), 1)
                # print(SR_copies.size())
                SR_test_features = feature_extractor(SR_test_copies)
                # print(SR_features.size())

                HR_test_copies = tc.cat((labels, labels, labels), 1)
                # print(HR_copies.size())
                HR_test_features = feature_extractor(HR_test_copies)
                # print(HR_features.size())

            if Use_kspace_loss == True:
                SR_test_freq = fft_k_space(SR_img_test)

                HR_test_freq = fft_k_space(labels)


            "calculate the gradients for all Variables during back prop"
            "vgg loss + pixel MSE loss + fft frequency loss, and we use weight_decay in Adam so that is L2 regularization"
            if Use_Feature_Map_Loss == True:
                feature_map_loss_test.append((args_loss_weight['feature_map_weight']*loss_function_MSE(SR_test_features, HR_test_features)).item())
#                print("feature_map_loss_test: ", feature_map_loss_test)

            if Use_Gradient_Map_Guided_Pixel_Wise_Loss == True:
                gradient_map_guided_weight_for_pixel_wise_loss = GradientMapGuidedWeightForPixelWiseLoss()
                gradient_map_difference_weight_matrix = gradient_map_guided_weight_for_pixel_wise_loss(SR_img_test, labels)
                pixel_wise_loss_test.append((args_loss_weight['pixel_wise_weight']*loss_function_L1(gradient_map_difference_weight_matrix*SR_img_test, gradient_map_difference_weight_matrix*labels)).item())
            elif Use_SSIM_Map_Guided_Pixel_Wise_Loss == True:
                ssim_map_guided_weight_for_pixel_wise_loss = SSIMMapGuidedWeightForPixelWiseLoss()
                ssim_map_difference_weight_matrix = ssim_map_guided_weight_for_pixel_wise_loss(SR_img_test, labels)
                pixel_wise_loss_test.append((args_loss_weight['pixel_wise_weight']*loss_function_L1(ssim_map_difference_weight_matrix*SR_img_test, ssim_map_difference_weight_matrix*labels)).item())
            else:
                pixel_wise_loss_test.append((args_loss_weight['pixel_wise_weight']*loss_function_L1(SR_img_test, labels)).item())
#                print("pixel_wise_loss_test: ", pixel_wise_loss_test)

            if Use_kspace_loss == True:
                if Amplify_High_Frequency_Value_In_K_Space_Loss == True:
                    k_space_freq_loss_test.append((args_loss_weight['k_space_weight']*(loss_function_MSE(
                        create_2d_Gaussian_weights(window_size = SR_test_freq.shape[2], num_of_samples = SR_test_freq.shape[0], channel = SR_test_freq.shape[1]).to(device)*SR_test_freq.real, 
                        create_2d_Gaussian_weights(window_size = HR_test_freq.shape[2], num_of_samples = HR_test_freq.shape[0], channel = HR_test_freq.shape[1]).to(device)*HR_test_freq.real) + 
                        loss_function_MSE(
                            create_2d_Gaussian_weights(window_size = SR_test_freq.shape[2], num_of_samples = SR_test_freq.shape[0], channel = SR_test_freq.shape[1]).to(device)*SR_test_freq.imag, 
                            create_2d_Gaussian_weights(window_size = HR_test_freq.shape[2], num_of_samples = HR_test_freq.shape[0], channel = HR_test_freq.shape[1]).to(device)*HR_test_freq.imag))).item())
                else:
                    k_space_freq_loss_test.append((args_loss_weight['k_space_weight']*(loss_function_MSE(SR_test_freq.real, HR_test_freq.real)+loss_function_MSE(SR_test_freq.imag, HR_test_freq.imag))).item())
#                print("k_space_freq_loss_test: ", k_space_freq_loss_test)


#            HR_ssim_test_weighted, HR_ssim_test = SSIM_function(labels,labels)
#            SR_ssim_test_weighted, SR_ssim_test = SSIM_function(SR_img_test, labels)
            HR_ssim_test = SSIM_function(labels,labels)
            SR_ssim_test = SSIM_function(SR_img_test, labels)
            if tc.isnan(SR_ssim_test):
                continue
            if Use_SSIM_L1_Loss == True:
                ssim_loss_test.append((args_loss_weight['ssim_weight']*loss_function_L1(SR_ssim_test.pow(args_loss_weight['ssim_component_weight']), HR_ssim_test.pow(args_loss_weight['ssim_component_weight']))).item())
            ssim_test.append(SR_ssim_test.item())   # Accumulation of SR_ssim in testing over all batches in one epoch, will be used to calculate the average value of SR SSIM for one epoch.
#                print("ssim_loss_test: ", ssim_loss_test)
            psnr_test.append((calc_psnr_for_mri_image(SR_img_test, labels)).item())

            if Use_Gradient_Map_L1_Loss == True:
                gradient_img_loss_test.append((args_loss_weight['gradient_img_weight']*loss_function_L1(calculate_gradient_map(args['n_colors'], SR_img_test), calculate_gradient_map(args['n_colors'], labels))).item())
#                print('gradient_loss_test: ', gradient_map_loss_test)

            if Use_Gram_Matrix_L1_Loss == True:
                gram_similarity_between_img_loss_test.append((args_loss_weight['gram_similarity_weight']*loss_function_L1(calculate_gram_matrix(SR_img_test), calculate_gram_matrix(labels))).item())

            if Use_Negative_TV_Loss == True:
                negative_total_variation_for_img_loss_test.append((negative_tv_loss(SR_img_test)).item())

            if Use_Negative_Trace_Loss == True:
                negative_trace_for_img_loss_test.append((negative_trace_loss(SR_img_test, labels)).item())


        loss_test = pixel_wise_loss_test 
        if Use_Feature_Map_Loss == True:
            loss_test = np.sum([loss_test, feature_map_loss_test],axis=0)
#                print('loss_test: ', loss_test)

#            if tc.isnan(k_space_freq_loss_test) != 1:
        if Use_kspace_loss == True:
            loss_test = np.sum([loss_test, k_space_freq_loss_test],axis=0)

#            if tc.isnan(ssim_loss_test) != 1 and Use_SSIM_L1_Loss == True:
        if Use_SSIM_L1_Loss == True:
            loss_test = np.sum([loss_test, ssim_loss_test],axis=0)

#            if tc.isnan(gradient_img_loss_test) != 1 and Use_Gradient_Map_L1_Loss == True:
        if Use_Gradient_Map_L1_Loss == True:
            loss_test = np.sum([loss_test, gradient_img_loss_test],axis=0)

        if Use_Gram_Matrix_L1_Loss == True:
            loss_test = np.sum([loss_test, gram_similarity_between_img_loss_test],axis=0)
            
        if Use_Negative_TV_Loss == True:
            loss_test = np.sum([loss_test, negative_total_variation_for_img_loss_test],axis=0)
            
        if Use_Negative_Trace_Loss == True:
            loss_test = np.sum([loss_test, negative_trace_for_img_loss_test],axis=0)


        batch_number_test = i+1
        ssim_test = np.mean(ssim_test) # Calculate avergae SR SSIM over all validation data samples in one epoch.
        print("ssim_test: ", ssim_test)
        psnr_test = np.mean(psnr_test)
        print("psnr_test: ", psnr_test)
        pixel_wise_loss_test = np.mean(pixel_wise_loss_test)
        print("pixel_wise_loss_test: ", pixel_wise_loss_test)
        if Use_Feature_Map_Loss == True:
            feature_map_loss_test = np.mean(feature_map_loss_test)
            print("feature_map_loss_test: ", feature_map_loss_test)
        if Use_kspace_loss == True:
            k_space_freq_loss_test = np.mean(k_space_freq_loss_test)
            print("k_space_freq_loss_test: ", k_space_freq_loss_test)
        if Use_SSIM_L1_Loss == True:
            ssim_loss_test = np.mean(ssim_loss_test)
            print("ssim_loss_test: ", ssim_loss_test)
        if Use_Gradient_Map_L1_Loss == True:
            gradient_img_loss_test = np.mean(gradient_img_loss_test)
            print('gradient_img_loss_test: ', gradient_img_loss_test)
        if Use_Gram_Matrix_L1_Loss == True:
            gram_similarity_between_img_loss_test = np.mean(gram_similarity_between_img_loss_test)
            print('gram_similarity_between_img_loss_test: ', gram_similarity_between_img_loss_test)
        if Use_Negative_TV_Loss == True:
            negative_total_variation_for_img_loss_test = np.mean(negative_total_variation_for_img_loss_test)
            print('negative_total_variation_for_img_loss_test: ', negative_total_variation_for_img_loss_test)
        if Use_Negative_Trace_Loss == True:
            negative_trace_for_img_loss_test = np.mean(negative_trace_for_img_loss_test)
            print('negative_trace_for_img_loss_test: ', negative_trace_for_img_loss_test)
        loss_test = np.mean(loss_test)
        print('loss_test: ', loss_test)
        print("*************************************")
    
    ssim_training = np.mean(ssim_training) # Calculate avergae SR SSIM over all training data samples in one epoch.
    print("ssim_training: ", ssim_training)
    psnr_training = np.mean(psnr_training)
    print("psnr_training: ", psnr_training)
    training_loss_for_current_epoch = np.mean(loss_training)
    print("training_loss: ", training_loss_for_current_epoch)
    pixel_wise_loss_for_current_epoch = np.mean(pixel_wise_loss_training)
    print("pixel_wise_loss_training: ", pixel_wise_loss_for_current_epoch)
    if Use_Feature_Map_Loss == True:
        feature_map_loss_for_current_epoch = np.mean(feature_map_loss_training)
        print("feature_map_loss_training: ", feature_map_loss_for_current_epoch)
    if Use_SSIM_L1_Loss == True:
        ssim_loss_for_current_epoch = np.mean(ssim_loss_training)
        print("ssim_loss_for_training: ", ssim_loss_for_current_epoch)
    if Use_Gradient_Map_L1_Loss == True:
        gradient_img_loss_for_current_epoch = np.mean(gradient_img_loss_training)
        print("gradient_img_loss_training: ", gradient_img_loss_for_current_epoch)
    if Use_kspace_loss == True:
        k_space_freq_loss_for_current_epoch = np.mean(k_space_freq_loss_training)
        print("k_space_freq_loss_training: ", k_space_freq_loss_for_current_epoch)
    if Use_Gram_Matrix_L1_Loss == True:
        gram_similarity_between_img_loss_for_current_epoch = np.mean(gram_similarity_between_img_loss_training)
        print("gram_similarity_loss_training: ", gram_similarity_between_img_loss_for_current_epoch)
    if Use_Negative_TV_Loss == True:
        negative_total_variation_for_img_loss_for_current_epoch = np.mean(negative_total_variation_for_img_loss_training)
        print("negative_total_variation_loss_training: ", negative_total_variation_for_img_loss_for_current_epoch)
    if Use_Negative_Trace_Loss == True:
        negative_trace_for_img_loss_for_current_epoch = np.mean(negative_trace_for_img_loss_training)
        print("negative_trace_loss_training: ", negative_trace_for_img_loss_for_current_epoch)


    "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"    
#    if tc.isnan(loss) == 1: #- loss == 'NaN':
#        break
    "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"


    "Save the weights of network model when it achieves best average SR SSIM over all validation data samples in one epoch"
    print("*************************************")
    print("best_ssim:", best_ssim)
    print("validation_ssim:", ssim_test)
    if ssim_test > best_ssim:
        best_ssim = ssim_test
        best_ssim_model_wts = copy.deepcopy(our_model_mri_sr_2d.state_dict())
        parameter_file = open(os.path.join(folder_log_path, 'best_ssim_network_parameter.pkl'), 'wb')
        pickle.dump(best_ssim_model_wts, parameter_file)
        parameter_file.close()
        best_ssim_epoch = epoch
    print("best_psnr:", best_psnr)
    print("validation_psnr:", psnr_test)
    if psnr_test > best_psnr:
        best_psnr = psnr_test
        best_psnr_model_wts = copy.deepcopy(our_model_mri_sr_2d.state_dict())
        parameter_file = open(os.path.join(folder_log_path, 'best_psnr_network_parameter.pkl'), 'wb')
        pickle.dump(best_psnr_model_wts, parameter_file)
        parameter_file.close()
        best_psnr_epoch = epoch
    print("min_validation_loss:",min_validation_loss)
    print("validation_loss:",loss_test)
    print("*************************************")
    if epoch==0:
        min_validation_loss=loss_test
        min_validation_loss_epoch = epoch
    elif min_validation_loss>loss_test:
        min_validation_loss=loss_test
        min_validation_loss_model_wts = copy.deepcopy(our_model_mri_sr_2d.state_dict())
        parameter_file = open(os.path.join(folder_log_path, 'min_validation_loss_network_parameter.pkl'), 'wb')
        pickle.dump(min_validation_loss_model_wts, parameter_file)
        parameter_file.close()
        min_validation_loss_epoch = epoch
        
    last_model_wts = copy.deepcopy(our_model_mri_sr_2d.state_dict())
    parameter_file = open(os.path.join(folder_log_path, 'last_network_parameter.pkl'), 'wb')
    pickle.dump(last_model_wts, parameter_file)
    parameter_file.close()


    
#    f.write('Training Loss:')
#    f.write('\n')
    f.write('Best SR SSIM for validation data has been achieved at epoch : %d' % (best_ssim_epoch))
    f.write('\n')
    f.write('Best SR PSNR for validation data has been achieved at epoch : %d' % (best_psnr_epoch))
    f.write('\n')
    f.write('Minimal validation loss has been achieved at epoch : %d' % (min_validation_loss_epoch))
    f.write('\n')
    f.write('Learning rate for epoch %d is : %f' % (epoch, learning_rate))
    f.write('\n')
    f.write('The batches with NaN for epoch %d is : %s' % (epoch, batch_with_nan))
    f.write('\n')
    f.write('Training SSIM for epoch %d is : %f' % (epoch, ssim_training))
    f.write('\n')
    f.write('Training PSNR for epoch %d is : %f' % (epoch, psnr_training))
    f.write('\n')
    f.write('The pixel_wise_loss for epoch %d is : %f' % (epoch, pixel_wise_loss_for_current_epoch))
    f.write('\n')
    
    if Use_Feature_Map_Loss == True:
        f.write('The feature_map_loss for epoch %d is : %f' % (epoch, feature_map_loss_for_current_epoch))
        f.write('\n')
    if Use_kspace_loss == True:
        f.write('The k_space_freq_loss for epoch %d is : %f' % (epoch, k_space_freq_loss_for_current_epoch))
        f.write('\n')
    if Use_SSIM_L1_Loss == True:
        f.write('The ssim_loss for epoch %d is : %f' % (epoch, ssim_loss_for_current_epoch))
        f.write('\n')
    if Use_Gradient_Map_L1_Loss == True:
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
    f.write('The training_loss for epoch %d is : %f' % (epoch, training_loss_for_current_epoch))
    f.write('\n')
    f.write(' \n')


#    f.write('Validation Loss:')
#    f.write('\n')
    f.write('Validation SSIM for epoch %d is : %f' % (epoch, ssim_test))
    f.write('\n')
    f.write('Validation PSNR for epoch %d is : %f' % (epoch, psnr_test))
    f.write('\n')
    f.write('The pixel_wise_loss_validation for epoch %d is : %f' % (epoch, pixel_wise_loss_test))
    f.write('\n')
    if Use_Feature_Map_Loss == True:
        f.write('The feature_map_loss_validation for epoch %d is : %f' % (epoch, feature_map_loss_test))
        f.write('\n')
    if Use_kspace_loss == True:
        f.write('The k_space_freq_loss_validation for epoch %d is : %f' % (epoch, k_space_freq_loss_test))
        f.write('\n')
    if Use_SSIM_L1_Loss == True:
        f.write('The ssim_loss_validation for epoch %d is : %f' % (epoch, ssim_loss_test))
        f.write('\n')
    if Use_Gradient_Map_L1_Loss == True:
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
    f.write('The validation_loss for epoch %d is : %f' % (epoch, loss_test))
    f.write('\n')
    f.write('----------------------------------------------------------------------------------')
    f.write(' \n')
    f.write(' \n')
    """
    if (epoch == EPOCH_NUM - 1):
        f.close()
    """
    
"Reload best weight parameters into the network model, which will be used for evaludation in the following part"
our_model_mri_sr_2d.load_state_dict(min_validation_loss_model_wts)
training_end = time.perf_counter()
running_time = training_end - training_start
print('The training time in minute is: ', running_time/60) 
print("training complete")
f.write('Total training time is: %f' % (running_time))
f.write('\n')



if Perform_Evaluation:
    "Evaluation(Test)"
    with tc.no_grad():
        our_model_mri_sr_2d.eval()
        f.write('Test started: \n')
    
        """
        The data loading pipeline for ordinary multi-channel SISR MRI SR or RGB SISR:
        """
        # =============================================================================
        # h5py.version
        # ===========================================================================
        "The folder where to load the training LR, HR data pair"

#    file_names_evaluation = os.listdir(folder_data_evaluation)

        for idx_file in file_names_evaluation:
            print(idx_file)
            if '.mat' in os.path.join(folder_data_evaluation, idx_file):
                print('One more low resolution image set exist')
                print(os.path.join(folder_data_evaluation, idx_file))
                file_data = h5py.File(os.path.join(folder_data_evaluation, idx_file), 'r')
                data_low_resolution = file_data['LR'][:] #----- numpy array
                print('Testing data: Shape of LR data is: ', np.shape(data_low_resolution))
                print(data_low_resolution.dtype)
                torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_low_resolution = torch_data_low_resolution.permute(0, 1, 3, 2)
                print('Testing data: Shape of LR data in Torch is: ', np.shape(torch_data_low_resolution))
                torch_data_low_resolution_sequence = torch_data_low_resolution
                print('Testing data: Shape of LR data sequence in Torch is: ', np.shape(torch_data_low_resolution_sequence))
                """
                data_high_resolution_groundtruth = file_data['HRGT'][:] #----- numpy array
                print('Testing data: Shape of HR data is: ', np.shape(data_high_resolution_groundtruth))
                torch_data_high_resolution_groundtruth = tc.from_numpy(data_high_resolution_groundtruth) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.permute(0, 1, 3, 2)
                print('Testing data: Shape of HR data in Torch is: ', np.shape(torch_data_high_resolution_groundtruth))
                torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth
                print('Testing data: Shape of HR data sequence in Torch is: ', np.shape(torch_data_high_resolution_groundtruth_sequence))
                """
                if args['use_HR_reference']:
#                print('One more reference image set exist')
#                print(os.path.join(folder_log_path, idx_file))
#                file_data_reference = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                    data_reference = file_data['REF'][:] #----- numpy array
                    print(np.shape(data_reference))
                    torch_data_reference = tc.from_numpy(data_reference) #----- torch type data could be read by tc.utils.data.TensorDataset
                    "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                    torch_data_reference = torch_data_reference.permute(0, 1, 3, 2)
                    print(np.shape(torch_data_reference))
                    torch_data_reference_sequence = torch_data_reference
                    print(np.shape(torch_data_reference_sequence))
                    torch_data_reference_sequence = torch_data_reference_sequence.float()
                torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.float()
                
                
                print(np.shape(torch_data_low_resolution_sequence))
                """
                torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.float()
                print(np.shape(torch_data_high_resolution_groundtruth_sequence))
                """

                """""""""""""""""""""""""""""""""""""""""""""
                3.2.c. Load MRI HR and LR Evaluation(Test) Data pair evaluation(test) part
                """""""""""""""""""""""""""""""""""""""""""""
                torch_data_low_resolution_eval_sequence = torch_data_low_resolution_sequence.float()
                if args['use_HR_reference'] == True:
                    torch_data_reference_eval_sequence = torch_data_reference_sequence.float()
                    testset = tc.utils.data.TensorDataset(torch_data_low_resolution_eval_sequence, torch_data_reference_eval_sequence)
                else:    
                    """
                    torch_data_high_resolution_groundtruth_eval_sequence = torch_data_high_resolution_groundtruth_sequence.float()
                    testset = tc.utils.data.TensorDataset(torch_data_low_resolution_eval_sequence, torch_data_high_resolution_groundtruth_eval_sequence)
                    """
                    testset = tc.utils.data.TensorDataset(torch_data_low_resolution_eval_sequence)
                    
                testloader = tc.utils.data.DataLoader(
                                testset, 
                                batch_size = batch_size,
                                shuffle = False, 
                                num_workers = 0,
                                pin_memory = False)
 
    
                prediction_start = time.perf_counter()
                "predict the SR MRI image by using testing LR image data and save them"
                for i, testing_data_2 in enumerate(testloader, 0):

                    if args['use_HR_reference'] == False:
                        LR_images_test = testing_data_2[0]
                        """
                        LR_images_test, HR_images_test = testing_data_2
                        HR_images_test = HR_images_test.type(tc.FloatTensor)
                        """
                        img_outputs = our_model_mri_sr_2d(Variable(LR_images_test).type(tc.FloatTensor).to(device))
                
                    elif args['use_HR_reference'] == True:
                        LR_images_test, references_test = testing_data_2
                        LR_images_test, references_test = Variable(LR_images_test).to(device), Variable(references_test).to(device)
                        img_outputs = our_model_mri_sr_2d(LR_images_test, references_test)

                #----- skip display "the last batch for one epoch test data" and skip "all the batches expect the batch in the middle"
                    SR_images_tensor_test = img_outputs.data.cpu()

                    if i==0:
                        SR_img_eval_tensor = SR_images_tensor_test
                        
                    else:
                        SR_img_eval_tensor = tc.cat((SR_img_eval_tensor, SR_images_tensor_test), 0)
                        
                prediction_end = time.perf_counter()
                prediction_time = prediction_end - prediction_start
                f.write('Prediction time for dataset %s is %f s \n' % (idx_file, prediction_time))
                SR_images_test = SR_img_eval_tensor.numpy()

                scipy.io.savemat(os.path.join(folder_log_path, 'test_results', os.path.splitext(idx_file)[0]+'_SR_test_image_ssim.mat'), mdict = {'SR_test_image' : SR_images_test})


print("the predicting of generated SR image by using testing samples complete")
f.close()
