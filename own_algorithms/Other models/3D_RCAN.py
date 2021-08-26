# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:53:29 2021

@author: AG_Dental
"""



import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
from math import exp
import numpy as np
import h5py
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
EPOCH_NUM = 0
accumulation_steps = int(8//batch_size)

Maintain_matrix_Size = False # stand for whether we want the output image has same size or NOT(e.g. larger size) as input image, e.g. set as Ture when apply for MRI motion artifact reduction, or applying through-plane downsampling MRI SR reconstruction.
Use_kspace_loss = False # stand for using K-space MSE loss in the total loss function
Use_SSIM_L1_Loss = True # stand for whether we want use SSIM L1 loss in the total loss function
Use_Gradient_Map_L1_Loss = False # stand for whether we want use gradient map L1 loss in the total loss function
Use_Gradient_Map_Guided_Pixel_Wise_Loss = False # stand for whether we want to use gradient map guided "attention weights" to multiply with pixel-wise loss. Can NOT be True if Use_SSIM_Map_Guided_Pixel_Wise_Loss is True
Use_SSIM_Map_Guided_Pixel_Wise_Loss = False # stand for whether we want to use SSIM map guided "attention weights" to multiply with pixel-wise loss. Can NOT be True if Use_Gradient_Map_Guided_Pixel_Wise_Loss is True
Amplify_Small_Value_In_Gradient_Map = False # stand for whether we want to amplify small values in gradient map to emphasize the information from gradient values which stand for texture
Amplify_High_Frequency_Value_In_K_Space_Loss = False # stand for whether we want to amplify high frequence loss values in k space loss
Use_Feature_Map_Loss = False # Stand for whether we want use feature map L1 loss in the total loss function
Use_saved_model = True # Load saved network weights
Freeze_random_seed = True # Freeze seed, so every time the network will have the same intialized weightes
Use_ssim_map = False # Use ssim map to calculate loss
Perform_Evaluation = True
print_loss_per_batch = 1000


if Use_Feature_Map_Loss == True:
    from torchvision.models import vgg19
    
    
# --------------------------- configuration of parameters for 2D_MRI_SR_Dual_Domain Reconstruct --------------------------- #

"The folder where to load the LR, HR data pair"
# folder_data_training = 'D:/Hao/SR_data/HCP_data/2x1_folds_3d_downsize_sag_128x1/training/'
folder_data_training = 'D:/Hao/SR_data/HCP_data/2x1_folds_3d_downsize_sag_128x1/training/'
file_names_training = os.listdir(folder_data_training)

# folder_data_validation = 'D:/Hao/SR_data/HCP_data/2x1_folds_3d_downsize_sag_128x1/validation/'
folder_data_validation = 'D:/Hao/SR_data/HCP_data/2x1_folds_3d_downsize_sag_128x1/validation/'
file_names_validation = os.listdir(folder_data_validation)

# folder_data_evaluation = 'D:/Hao/SR_data/HCP_data/2x1_folds_3d_downsize_sag_128x1/evaluation/'
folder_data_evaluation = 'D:/Hao/SR_data/HCP_data/2x1_folds_3d_downsize_sag_128x1/evaluation/'
file_names_evaluation = os.listdir(folder_data_evaluation)

"The folder for log and results"
folder_log_path = 'D:/Hao/results/20210814_2d_SRCNN_128x1_3d_downsize_seed1_cosine_101_HCP300/'

"The folder of saved network parameters"
folder_saved_network = 'D:/Hao/results/20210814_2d_SRCNN_128x1_3d_downsize_seed1_cosine_101_HCP300/'



args = {'in_colors': 3, 'out_colors': 6,'n_resgroups': 5, 'n_rcablocks': 5, 'n_feats': 64, 'reduction': 16, 'scale': 2, 
        'seed': 1, 'optimizer': 'Adam', 'learning_rate_decay_method': 'cosine_learning_rate_decay',
        'use_learning_rate_warm_up': False, 'how_many_epoch_to_be_used_for_warm_up': 10, 'initial_learning_rate_after_warm_up': 0.0001}

args_loss_weight = {'feature_map_weight': 20, 'pixel_wise_weight': 100, 'k_space_weight': 10, 'ssim_weight': 50, \
                    'gradient_img_weight': 50, 'gradient_grd_weight': 10, 'k_space_branch_weight': 0.02, \
                    'wavelets_branch_weight': 5, 'ssim_component_weight': 2}





# args['n_colors'] = 1, stands for number of channels of input image, e.g. 1 for MRI image, 3 for RGB image.
# args['n_feats'] = 128, stands for how many "number of channels" for feature map going through model
# args['scale'] = 2, stands for scale factor used in one upsampler, e.g. 2, 4
# arg['optimizer'] = ['Adam'] # stand for which optimizer we want use for training, e.g. 'Adam', 'SGD_with_momentum', 'look_ahead'
# arg['learning_rate_decay_method'] = ['cosine_learning_rate_decay'] # stand for which learning rate decay method we want use for training, e.g. 'cosine_learning_rate_decay', 'multi_step_learning_rate', 'step_learning_rate', 'cosine_learning_rate_warm_restarts'


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
    if 'data1.mat' in os.path.join(folder_data_training, idx_file):
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
    else:
        print('other type NOT support for now')

"""Note for 2D matrix data with dimension H x W, everytime before loading into trainset and testset, we have to adapt the dimension into format N x C X H x W. Cause all
the dimension of input/output are using N x C x H x W. N denotes number of data, C denotes number of channels, H means height, W stays width"""        
torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.float()
print(np.shape(torch_data_low_resolution_sequence))
torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.float()
print(np.shape(torch_data_high_resolution_groundtruth_sequence))
# num_training_samples = math.floor(torch_data_low_resolution_sequence.size(0))
print('All mat files have been concatenated into one tensor for each type, data is ready to be loaded!')

"""""""""""""""""""""""""""""""""""""""""""""
1.2.c. Load MRI HR and LR Data pair training part
"""""""""""""""""""""""""""""""""""""""""""""
torch_data_low_resolution_training_sequence = torch_data_low_resolution_sequence.float()
torch_data_high_resolution_groundtruth_training_sequence = torch_data_high_resolution_groundtruth_sequence.float()
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
    if 'data1.mat' in os.path.join(folder_data_validation, idx_file):
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
    else:
        print('other type NOT support for now')

"""Note for 2D matrix data with dimension H x W, everytime before loading into trainset and testset, we have to adapt the dimension into format N x C X H x W. Cause all
the dimension of input/output are using N x C x H x W. N denotes number of data, C denotes number of channels, H means height, W stays width"""        
torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.float() 
print(np.shape(torch_data_low_resolution_sequence))
torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.float()
print(np.shape(torch_data_high_resolution_groundtruth_sequence))
print('All mat files have been concatenated into one tensor for each type, data is ready to be loaded!')

"""""""""""""""""""""""""""""""""""""""""""""
2.2.c. Load MRI HR and LR Validation Data pair validation part
"""""""""""""""""""""""""""""""""""""""""""""
torch_data_low_resolution_validation_sequence = torch_data_low_resolution_sequence.float()
torch_data_high_resolution_groundtruth_validation_sequence = torch_data_high_resolution_groundtruth_sequence.float()
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





"""""""""""""""""""""""""""""""""""""""
 Definition of 3D RCAN
"""""""""""""""""""""""""""""""""""""""

"default conv layer"
def default_conv(in_channels, out_channels, kernel_size, bias = True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


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
        self.avg_pool = nn.AdaptiveAvgPool3d(1) # global average pooling(GAP), output size is 1 for each channel
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True), # W_d in CA
                nn.ReLU(inplace=True), # ReLU in CA
                nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True), # W_u in CA
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
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, use_channel_and_spatial_attention_inside_RCAB = False, 
        channel_and_spatial_attention_framework = 'CBAM', channel_and_spatial_attention_mode = 'sequential_mode'):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2): # conv --> ReLU --> conv
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: 
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction)) # use CA Layer

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



"Residual Channel Attention Network (RCAN) based Dual Domain Network for Super Resolution MRI"
class RCAN_3D(nn.Module):
    """
    RCAN(Deep Residual Channel Attention Network) = RIR(Residual in Residual module) + Upsampler Module.
    See bottom figure in figure 2 of original RCAN paper
    """
    def __init__(self, args, not_use_last_conv_to_change_num_channels_to_n_colors = False):
        super(RCAN_3D, self).__init__()
        if args['type_of_network'] == 'image_single_domain':
            self.type_of_network = 'image_single_domain'

        conv = default_conv
        act = nn.ReLU(True)
        

        n_resgroups = args['n_resgroups'] # number of RGs in RIR/RCAN
        n_rcablocks = args['n_rcablocks'] # number of RCABs in one RG
        in_colors = args['in_colors'] # number of channels going of input of entire model
        out_colors = args['out_colors'] # number of channels going of output of entire model
        n_feats = args['n_feats'] # number of feature maps/channels going through the entire model
        kernel_size = 3 # conv filter size used for all conv in RCAN
        reduction = args['reduction'] # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        scale = args['scale'] # resize factor, e.g. 2, 4
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
        modules_head = [conv(in_colors, n_feats, kernel_size)] # the first conv layer in RCAN, show in figure 2 of RCAN paper

        # define commonly use pretail module
        modules_pretail = [conv(n_feats, n_feats, kernel_size)]


        # define body module for different type of network. 
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_rcablocks=n_rcablocks,
                use_channel_and_spatial_attention_inside_RCAB = use_channel_and_spatial_attention_inside_RCAB, 
                channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, 
                channel_and_spatial_attention_mode = channel_and_spatial_attention_mode) \
            for _ in range(n_resgroups)]

        # define tail module. The last stage is upsampling module and one more conv layer, show in figure 2 of RCAN paper
        modules_tail = []
        if (Maintain_matrix_Size == False):
            modules_tail.append(
                Upsampler(conv, scale, n_feats, use_channel_and_spatial_attention_inside_upsampler = use_channel_and_spatial_attention_inside_upsampler, 
                          channel_and_spatial_attention_framework = channel_and_spatial_attention_framework, channel_and_spatial_attention_mode = channel_and_spatial_attention_mode))
        modules_tail.append(conv(n_feats, out_colors, kernel_size))

        # Add a downsize converter by using conv layer.
        # The purpose of original RCAN designed in original RCAN paper is to upscale scale times of LR image, the output from RIR has same size as input LR image. However, here in our task
        # we actually want to maintain output(SR) and input(LR) same as, it means we need to downsize the image before leave it into upscale module. That is the reason we need extra conv layer to perform 
        # down size with scale time first before going through upscale with scale time

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.pretail = nn.Sequential(*modules_pretail)
        self.tail = nn.Sequential(*modules_tail)

        self.relu = nn.ReLU()

    def forward(self, x):
        # do NOT understand why need this, may NOT be useful for us 
        """ x = self.sub_mean(x) """
        x = self.head(x) # input image goes through first conv layer
        res = self.body(x) # data goes through sveral ResidualGroups
        res = self.pretail(res)
        res += x # long skip connection of RIR
#        if (Maintain_in_plne_Size == True):
#            res = self.down_size_converter(res) # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output
        x = self.tail(res) # data goes through upsampling module and one more conv layer
#        x = self.relu(x)
        # do NOT understand why need this, may NOT be useful for us
        """ x = self.add_mean(x) """
        return x

    
    
    
device=tc.device("cuda" if use_cuda else "cpu")

if Freeze_random_seed == True:
    tc.manual_seed(args['seed'])
#    tc.backend.cudnn.deterministic = True
#    tc.backend.cudnn.benchmark = False
    print('Seed is frozen!')
 
    
our_model_mri_sr_2d = RCAN_3D(args)


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
        f.write('Maintain_matrix_Size is: %s\n' % Maintain_matrix_Size)
        f.write('Freeze_random_seed is: %s\n' % Freeze_random_seed)
        f.write('Use_saved_model is: %s\n' % Use_saved_model)
        f.write('Use_kspace_Loss is: %s\n' % Use_kspace_loss)
        f.write('Use_SSIM_L1_Loss is: %s\n' % Use_SSIM_L1_Loss)
        f.write('Use_Gradient_Map_L1_Loss is: %s\n' % Use_Gradient_Map_L1_Loss)
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
        
        "load input data"
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        # print('The data have been loaded' )

        "forward prop"
        # outputs = our_resnext(inputs).double() #-- numpy arrays are 64-bit floating point and will be converted to torch.DoubleTensor standardly. Now, if you use them with your model, you'll need to make sure that your model parameters are also Double
        img_outputs = our_model_mri_sr_2d(inputs) #-- or using default float as type, however remember to cast the input from Double to Float            
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
        if i % print_loss_per_batch == 0 and i!=0: #----- print log info every 1000 batch
            if i == 0:
                print('[%d, %5d] loss: %.3f' \
                      % (epoch, i, running_loss))
            else:
                print('[%d, %5d] loss: %.3f' \
                      % (epoch, i, running_loss / print_loss_per_batch))
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


            "load input data"
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            SR_img_test = our_model_mri_sr_2d(inputs)
            
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

                    LR_images_test = testing_data_2[0]
                    img_outputs = our_model_mri_sr_2d(Variable(LR_images_test).type(tc.FloatTensor).to(device))

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


