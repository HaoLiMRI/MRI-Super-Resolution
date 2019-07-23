#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:40:06 2019

@author: HaoLi
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


"-------------------------------------------------------------------------------------------------"
print('boolean value to see if GPU is ready:', tc.cuda.is_available())
print('number of GPU is', tc.cuda.device_count())
print(tc.cuda.get_device_name(0))
use_cuda = True #-- boolean to choose GPU

"Note folder log path is changed for 2D_Gated_Dilated_ResNeXt_Based_MRI_SR_v2"
folder_log_path = '/home/HaoLi/SR/Data/whole_image_training/'
file_names = os.listdir(folder_log_path)

for idx_file in file_names:
    print(idx_file)
    if 'LR' in os.path.join(folder_log_path, idx_file):
        print(os.path.join(folder_log_path, idx_file))
        file_data_lr_whole = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
        data_lr_whole = file_data_lr_whole['LR'][:] #----- numpy array
        print(np.shape(data_lr_whole))
        torch_data_lr_whole = tc.from_numpy(data_lr_whole) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension\
        changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_lr_whole = torch_data_lr_whole.permute(0, 2, 1)
# =============================================================================
#         torch_data_low_resolution = tc.t(torch_data_low_resolution)
# =============================================================================
# =============================================================================
#         torch_data_low_resolution = torch_data_low_resolution.type(tc.DoubleTensor)
# =============================================================================
        print(np.shape(torch_data_lr_whole))
    else:
        print('Other types not supported')
        
torch_data_lr_whole = torch_data_lr_whole.unsqueeze(1)    
print(np.shape(torch_data_lr_whole))
        
before = time.clock()
SR_whole = our_resnext(Variable(torch_data_lr_whole).type(tc.FloatTensor).to(device))
after = time.clock()
print('Process done! Processing time: ',(after-before))
SR_whole = SR_whole.cpu().squeeze(1)
SR_whole = SR_whole.detach().numpy()
scipy.io.savemat('/home/HaoLi/SR/Data/whole_image_training/SR_whole1.mat',mdict={'SR_whole':SR_whole})

        
        
        