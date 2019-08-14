# -*- coding: utf-8 -*-
"-------------------------------------------------------------------------------------------------"
"""
2D_Gated_U-ResNeXt_Based_MRI_SR Reconstruct
"""
"""
Author: chisyliu@hotmail.com *
        hao.li@med.uni-heidelberg.de *
        
        * Both authors contribute equally
Version: 4.5.0
"""
"-------------------------------------------------------------------------------------------------"
"""
This is the current version we are working on, in 20190727
This is a demo code of 2D_Gated_U-ResNeXt_Based_MRI_SR.
    0) This is the 2D version of Gated_U-ResNeXt_Based_MRI_SR, in this version we have following items:
        a) Optional module(exist or NOT): a VGG feature extractor is added before the main "CNN based Reconstruct network"(so the input to "CNN based Reconstruct network" is feature map of LR image)
        b) Optional module(either one exist): either Pixel-Wise MSE loss or Pixel-Wise L1 loss
        c) Optional module(exist or NOT): weighted k space loss(fft loss)
        d) Optional module(exist or NOT): weighted VGG loss
        e) Optional module(exist or NOT): weighted L1 Regularization
        f) Optional module(): ssim_loss
        g) Optional module(): log_ssim_loss
        h) Optional module(): ms-ssim_loss
	i) add option to use lookahead optimizer
        loss function = Pixel-Wise MSE loss(or Pixel-Wise L1 loss) + weighted VGG loss + weighted k space loss + log_ssim_loss + weighted L1 Regularization
       
        
       NOT like v1 there uses the layer of 3D image/video as channel directly. In this 2D v2 version, the data format has been changed. The input data
       is just 64 x 64 2D matrix rather than 64 x 64 x 64, we already collapse all the 64 layers into only one layer in the data tailing and noise filtering processing
    1) ResNeXt[8] in theory, works better than ResNet[1](which works as almost convex problem[2]), almost equals to performance of DenseNet[10] as comparaion of 6.figure of[9] especially on the following items:
        a) 
        b) 
    2) used batch normalization(BN, normalization over all training data in one mini-batch)[7] to make the algorithm converge fast, due to some possible reason:
        --by making the shape of objective function more symmetric which means where ever from to start searching, the converge speed should be similar rather\
        than somewhere converge too slow.
        --by making the activation to be followed by same 'mean' and 'variance' in one mini-batch, thus allocate larger scale for the gradient of activation which\
        should be smaller, to prevent vanishing of gradient problem
        --if we do NOT use batch normalization on each of layer, then the bias and variance of input of each layer will be highly dependent on previous layer's\
        weights/parameters, this could introduce very complex nonlinearity, which makes the network hard to train and might overfitting. By introducing BN, the bias\
        and varaince of input of each layer will be indepedent on previous layer's weights/parameters, so de-composite the complexity of network thus easy to train\
        and slightly avoid overfitting
    3) In this version, the "cardinality" is implemented by using property "group" in PyTorch for each conv layer
    4) In our code, we used 3D conv layers within ResNeXt architecture. In case all the parameter "C" set as 1, the group are off and the model is just back to 3D ResNet based MRI SR
    5) In out code, we also added "Epsilon-gate" structure to make chance of reducing the size of our 3D ResNeXt based MRI SR model(adaptively decide if the residual remain or not to reduce the size of ResNeXt),
       by using the same way of implement "Epsilon-gate" added residual block in Epsilon-ResNet[18]. "Epsilon-gate" could be implemented by using multiplication between sparse promoting fucntion (which 
       consists of 4 ReLU) and the input into sparse promoting function itself(which is output from previous layer), as written in [18] and [19]. In case all the "gate_in_use" 
       parameters set as False, the Epsilon-gates are off and the model is just back to 3D ResNeXt based MRI SR
    6) In out code, we also added Dilated-ResNet[20] module. There are already several examples of using Dilated conv, e.g. [22], [23], [24]
       Dilated-ResNet[20] and Dilated-CNN[21] are proposed in 2017, 2015. Dilated convolution actually implements convolution with holes, is used to replace the downsampling(pooling layer).
       By doing this, Dilated-ResNet increases the resolution of output feature maps(cause prevents the elimination of spatial acuity, so it has higher chance to achieve better accuracy) 
       without reducing the receptive field of individual neurons(just use  stride = 1 for pooling layer/"removing subsampling" could also maintain higher resolution of feature maps, however
       this reduces the receptive field in subsequent layers). In other words, Dilated conv could support the exponetially increasing of receptive field, but unlike pooling, 
       Dilated conv not loss any resolution.
    7) 2018.08.23. Added He Kaiming's initialization[3]
       how to use initilization for Pytorch? https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    
    @TODO:
        1) need to consider adding the "selection of weights of model which has best performance for validation/test loss", try to catch the best test loss in the training loop and save the checkpoint.\
           Check how it is designed according to [17]
        2) consider dropout layer to prevent overfitting[6]
        4) consider added L2 regularization to avoid overfitting,
        5) consider to switch from Adam in the early training iterations to SGD in the late training iterations, inspired from [4].
           By doing this, it may enhance the generilization ability of network on performance in test/dev data
        6) consider making ResNet50, by adding “bottleneck” building block reducing the consumption of computation by using
           multiple smalll size conv filter to replace one big size conv filter[1]
        7) might consider just using residual link in the shallow layers rather than using residual link in both shallow layers(e.g. first 30% layers) 
           and deep layers(e.g. the deeper 70% layers), the reason is only residual link in shallow layers are actually passing the value actively 
           according to [5]
        8) might consider jointly using WideResNet[11] structure to generate more feature maps(channels) for each layer, and ResNeXt structure to take multi-"cardinality"
        9) might also consider SGDR[12] to replace the normal SGD to optimize.
        10)might also consider changing the order of connection, from "Conv --> BN --> ReLU"(normal connection) to "BN --> ReLU --> Conv"(so called full pre-activation)[13]
           The authors of ResNet[1] found out the performance increased if order of connection changed to full pre-activation in [13]. However, note the BN should always be
           placed before ReLU or other activation functions, due that "BN is used to produce activations function with the desired distribution"[14] so it has to be before 
           activate function
        11)might also see if the SpikingResNet[15] structure could be working for MRI SR Reconstruction
"""
"-------------------------------------------------------------------------------------------------"
"""Reference: 
    [1] 2015 Deep Residual Learning for Image Recognition. download here: https://arxiv.org/pdf/1512.03385.pdf"
        tutorial online: https://icml.cc/2016/tutorials/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf
    [2] 2017 Convergence Analysis of Two-layer Neural Networks with ReLU Activation. https://arxiv.org/pdf/1705.09886.pdf
        presentation: https://www.youtube.com/watch?v=pWxSUZWOctk
    [3] 2015 Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. https://arxiv.org/pdf/1502.01852.pdf
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
    [11]2016 Wide Residual Networks. https://arxiv.org/abs/1605.07146
        source code: https://github.com/szagoruyko/wide-residual-networks
    [12]2016 SGDR: Stochastic Gradient Descent with Warm Restarts. https://arxiv.org/abs/1608.03983
    [13]2016 Identity Mappings in Deep Residual Networks. https://arxiv.org/abs/1603.05027
    [14]2015 Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/pdf/1502.03167.pdf
    [15]2018 Spiking Deep Residual Network. https://arxiv.org/abs/1805.01352
    [16]2017 Multi-scale brain MRI super-resolution using deep 3D convolutional networks. 
    [17]2018 Brain MRI super resolution using 3D deep densely connected neural networks.
    [18]2018 Learning Strict Identity Mappings in Deep Residual Networks. https://arxiv.org/abs/1804.01661
    [19]2015 Highway Networks. https://arxiv.org/abs/1505.00387
    [20]2017 Dilated Residual Networks. https://arxiv.org/abs/1705.09914
    [21]2015 Multi-Scale Context Aggregation by Dilated Convolutions. https://arxiv.org/abs/1511.07122
    [22]2018 CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes.
        https://arxiv.org/abs/1802.10062
    [23]2017 Dilated Recurrent Neural Networks. https://papers.nips.cc/paper/6613-dilated-recurrent-neural-networks.pdf
    [24]2017 Dilated convolution neural network with LeakyReLU for environmental sound classification
        https://www.researchgate.net/publication/320886596_Dilated_convolution_neural_network_with_LeakyReLU_for_environmental_sound_classification
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


batch_size = 16
EPOCH_NUM = 150
SELECTED_BATCH_FOR_PLOT_AND_SAVE_MAT_FILE = 10
Feature_Extractor_in_Front_of_Network = False
Use_Batch_Norm = False
Use_Transpose_Conv_as_Upsampling_Approach = False
Use_Lookahead_Optimizer = False

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
    if 'LR_training' in os.path.join(folder_log_path, idx_file):
        print('One more low resolution image set exist')
        num_low_resolution_mat_file = num_low_resolution_mat_file + 1
        print(os.path.join(folder_log_path, idx_file))
        file_data_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
        data_low_resolution = file_data_low_resolution['LR'][:] #----- numpy array
        print(np.shape(data_low_resolution))
        torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension\
        changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
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
    elif 'HRGT_training' in os.path.join(folder_log_path, idx_file):
        print('One more high resolution groundtruth image set exist')
        num_high_resolution_groundtruth_mat_file = num_high_resolution_groundtruth_mat_file + 1
        print(os.path.join(folder_log_path, idx_file))
        file_data_high_resolution_groundtruth = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
        data_high_resolution_groundtruth = file_data_high_resolution_groundtruth['HRGT'][:] #----- numpy array
        print(np.shape(data_high_resolution_groundtruth))
        torch_data_high_resolution_groundtruth = tc.from_numpy(data_high_resolution_groundtruth) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension\
        changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
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
2. Load MRI HR and LR Data pair part
"""""""""""""""""""""""""""""""""""""""""""""
torch_data_low_resolution_training_sequence = torch_data_low_resolution_sequence.float()
torch_data_high_resolution_groundtruth_training_sequence = torch_data_high_resolution_groundtruth_sequence.float()

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





"""""""""""""""""""""""""""""""""""""""
3. Define ResNeXt architecture part
"""""""""""""""""""""""""""""""""""""""
"Note: all size of in and out channels, stride, padding, etc, could refer to table 1 from Kaiming[1], size of each layer output depends on input data"
"Residual block, as submodule"

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
    

class Normal_Residual_Block(nn.Module): #----- In this version, the "cardinality" is implemented by using property "group" in PyTorch for each conv layer
    def __init__(self, number_in_channel, number_out_channel, stride = 1, linear_projection = None, num_of_group = 32, gate_in_use = True, EPSILON = 1, Use_Batch_Norm = True): #----- num_of_group is "cardinality"
        super(Normal_Residual_Block, self).__init__()     #----- Call the constructor of base class explicitly
        if (Use_Batch_Norm == True):
            self.normal_path = nn.Sequential(
                nn.Conv2d(number_in_channel, number_out_channel, 3, stride, 1, bias = False), #---- No need to make bias learnable, due using BatchNorm
                nn.BatchNorm2d(number_out_channel),
                nn.ReLU(inplace = True), #-----inplace = True could overwrite the input of ReLU by using output to save memory(ReLU only needs output to calculate gradient )
                nn.Conv2d(number_out_channel, number_out_channel, 3, 1, 1, bias = False, groups = num_of_group),
                nn.BatchNorm2d(number_out_channel))
        else:
            self.normal_path = nn.Sequential(
                nn.Conv2d(number_in_channel, number_out_channel, 3, stride, 1, bias = False), #---- No need to make bias learnable, due using BatchNorm
                nn.ReLU(inplace = True), #-----inplace = True could overwrite the input of ReLU by using output to save memory(ReLU only needs output to calculate gradient )
                nn.Conv2d(number_out_channel, number_out_channel, 3, 1, 1, bias = False, groups = num_of_group))
        self.linear_projection_on_shortcut_path = linear_projection
        self.gate_in_use = gate_in_use
        self.epsilon = EPSILON
        
    def forward(self, x):
        out = self.normal_path(x)
        # residual = x if self.shortcut_path is None else self.shortcut_path(x)
        if self.linear_projection_on_shortcut_path == None:
            residual_link = x
        else:
            residual_link = self.linear_projection_on_shortcut_path(x)
        
        "Gate on or off"
        if self.gate_in_use == True:
            y = tc.max(tc.nn.functional.relu((out - self.epsilon), inplace = True) + tc.nn.functional.relu(((-1) * out - self.epsilon), inplace = True))
# =============================================================================
#             print(y)
# =============================================================================
            gate_on_off_result = tc.nn.functional.relu(tc.nn.functional.relu(y * (-10000000) + 1) * (-10000000) + 1)
            out = gate_on_off_result * out + residual_link
        else:
            out = out + residual_link
        return F.relu(out)


class BottleNeck_Residual_Block(nn.Module): #----- In this version, the "cardinality" is implemented by using property "group" in PyTorch for each conv layer
    "Note: number_out_channel must be even number"
    def __init__(self, number_in_channel, number_out_channel, stride = 1, linear_projection = None, num_of_group = 32, gate_in_use = True, EPSILON = 1, Use_Batch_Norm = True): #----- num_of_group is "cardinality"
        super(BottleNeck_Residual_Block, self).__init__()     #----- Call the constructor of base class explicitly
        if number_out_channel % 2 == 1:
            print('Warning: number_out_channel for Bottle Neck Residual Block is NOT even number')
        if (Use_Batch_Norm == True):
            self.normal_path = nn.Sequential(
                    nn.Conv2d(number_in_channel, int(number_out_channel/2), 1, 1, 0, bias = False), #---- No need to make bias learnable, due using BatchNorm
                    nn.BatchNorm2d(int(number_out_channel/2)),
                    nn.ReLU(inplace = True), #-----inplace = True could overwrite the input of ReLU by using output to save memory(ReLU only needs output to calculate gradient )
                    nn.Conv2d(int(number_out_channel/2), int(number_out_channel/2), 3, stride, padding = 1, groups = num_of_group, bias = False),
                    nn.BatchNorm2d(int(number_out_channel/2)),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(int(number_out_channel/2), number_out_channel, 1, 1, 0, bias = False),
                    nn.BatchNorm2d(number_out_channel))
        else:
            self.normal_path = nn.Sequential(
                    nn.Conv2d(number_in_channel, int(number_out_channel/2), 1, 1, 0, bias = False), #---- No need to make bias learnable, due using BatchNorm
                    nn.ReLU(inplace = True), #-----inplace = True could overwrite the input of ReLU by using output to save memory(ReLU only needs output to calculate gradient )
                    nn.Conv2d(int(number_out_channel/2), int(number_out_channel/2), 3, stride, padding = 1, groups = num_of_group, bias = False),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(int(number_out_channel/2), number_out_channel, 1, 1, 0, bias = False))                 
        self.linear_projection_on_shortcut_path = linear_projection
        self.gate_in_use = gate_in_use
        self.epsilon = EPSILON
        
    def forward(self, x):
        out = self.normal_path(x)
        #-- residual = x if self.shortcut_path is None else self.shortcut_path(x)
        if self.linear_projection_on_shortcut_path == None:
            residual_link = x
        else:
            residual_link = self.linear_projection_on_shortcut_path(x)        
        
        "Gate on or off"
        if self.gate_in_use == True:
            y = tc.max(tc.nn.functional.relu((out - self.epsilon), inplace = True) + tc.nn.functional.relu(((-1) * out - self.epsilon), inplace = True))
# =============================================================================
#             print(y)
# =============================================================================
            gate_on_off_result = tc.nn.functional.relu(tc.nn.functional.relu(y * (-10000000) + 1) * (-10000000) + 1)
            out = gate_on_off_result * out + residual_link
        else:
            out = out + residual_link
        return F.relu(out)
    
	
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale, Use_Transpose_Conv_as_Upsampling_Approach):
        super(UpsampleBLock, self).__init__()

        self.Use_Transpose_Conv_as_Upsampling_Approach = Use_Transpose_Conv_as_Upsampling_Approach
        kernel_size_for_trans_conv, stride_for_trans_conv, padding_for_trans_conv = {
            # W2=(W1−F+2P)/S+1, H2=(H1−F+2P)/S+1. For this particular configuration (kernel_size = 6, stride = 2, padding = 2) it means shriks 1/2 for normal conv layer, and expands/upscales 2 for transpose conv layers
            2: (6, 2, 2),
            #  For this particular configuration (kernel_size = 8, stride = 4, padding = 2) it means shriks 1/4 for normal conv layer, and expands/upscales 4 for transpose conv layers
            4: (8, 4, 2), 
            # For this particular configuration (kernel_size = 12, stride = 8, padding = 2) it means shriks 1/8 for normal conv layer, and expands/upscales 8 for transpose conv layers
            8: (12, 8, 2)
        }[up_scale]
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()
        self.trans_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size_for_trans_conv, stride=stride_for_trans_conv, padding=padding_for_trans_conv)
            
    def forward(self, x):
        if (self.Use_Transpose_Conv_as_Upsampling_Approach == False): # use sub-pixel conv as way of upsampling
            x = self.conv(x)
            x = self.pixel_shuffle(x)
            x = self.prelu(x)
        else: # transpose conv as way of upsampling
            x = self.trans_conv(x)
            x = self.prelu(x)

        return x
		
        
"2D_ResNeXt"
class ResNeXt_2D(nn.Module):                   #----- Define a Net class as derived class inherited from nn.Module
    "Residual_Block_Type is either class 'BottleNeck_Residual_Block' or class 'Normal_Residual_Block"
    def __init__(self, Residual_Block_Type, feature_extractor_in_front_bool, Use_Batch_Norm, Use_Transpose_Conv_as_Upsampling_Approach):                 #----- __init__ define the constructor of Net class, consist of declaration of components in network              
        super(ResNeXt_2D, self).__init__()     #----- Call the constructor of base class explicitly
        
        self.feature_extractor_in_front_bool = feature_extractor_in_front_bool
        self.Use_Batch_Norm = Use_Batch_Norm
        self.Use_Transpose_Conv_as_Upsampling_Approach = Use_Transpose_Conv_as_Upsampling_Approach
        if self.feature_extractor_in_front_bool == True:
            "declaration of the first non-residual normal block"
            if (self.Use_Batch_Norm == True):
                self.normal_block = nn.Sequential(FeatureExtractor(),
                nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            else:
                self.normal_block = nn.Sequential(FeatureExtractor(),
                nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False),
                nn.ReLU(inplace=True))
            
            "declaration of the rest parts which consist of residual blocks"
            self.part1 = self.make_residual_part(Residual_Block_Type, 64, 32, 3, 1, stride = 1, gate_in_use = False, Use_Batch_Norm = self.Use_Batch_Norm)
            self.part2 = self.make_residual_part(Residual_Block_Type, 32, 32, 4, 1, stride = 1, gate_in_use = False, Use_Batch_Norm = self.Use_Batch_Norm)
            self.part3 = self.make_residual_part(Residual_Block_Type, 32, 64, 6, 1, stride = 1, gate_in_use = True, Use_Batch_Norm = self.Use_Batch_Norm)
        else:
            "declaration of the first non-residual normal block"
            if (self.Use_Batch_Norm == True):
                self.normal_block = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))
            else:
                self.normal_block = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))
        self.part1 = self.make_residual_part(Residual_Block_Type, 16, 16, 3, 1, stride = 2, gate_in_use = False, Use_Batch_Norm = self.Use_Batch_Norm)
        self.part2 = self.make_residual_part(Residual_Block_Type, 16, 32, 4, 1, stride = 2, gate_in_use = False, Use_Batch_Norm = self.Use_Batch_Norm)
        self.part3 = self.make_residual_part(Residual_Block_Type, 32, 64, 6, 1, stride = 2, gate_in_use = False, Use_Batch_Norm = self.Use_Batch_Norm)
        self.part4 = self.make_residual_part(Residual_Block_Type, 64, 128, 3, 1, stride = 2, gate_in_use = False, Use_Batch_Norm = self.Use_Batch_Norm)
        self.part5 = self.make_residual_part(Residual_Block_Type, 128, 256, 3, 1, stride = 2, gate_in_use = True, Use_Batch_Norm = self.Use_Batch_Norm)
        self.part6 = self.make_residual_part(Residual_Block_Type, 256, 512, 3, 1, stride = 2, gate_in_use = True, Use_Batch_Norm = self.Use_Batch_Norm)
			
        self.part7 = nn.Sequential(UpsampleBLock(512, 2, Use_Transpose_Conv_as_Upsampling_Approach = self.Use_Transpose_Conv_as_Upsampling_Approach),
        nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = False))
        self.part8 = nn.Sequential(UpsampleBLock(256, 2, Use_Transpose_Conv_as_Upsampling_Approach = self.Use_Transpose_Conv_as_Upsampling_Approach),
        nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False))
        self.part9 = nn.Sequential(UpsampleBLock(128, 2, Use_Transpose_Conv_as_Upsampling_Approach = self.Use_Transpose_Conv_as_Upsampling_Approach),
        nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False))
        self.part10 = nn.Sequential(UpsampleBLock(64, 2, Use_Transpose_Conv_as_Upsampling_Approach = self.Use_Transpose_Conv_as_Upsampling_Approach),
        nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False))
        self.part11 = nn.Sequential(UpsampleBLock(32, 2, Use_Transpose_Conv_as_Upsampling_Approach = self.Use_Transpose_Conv_as_Upsampling_Approach),
        nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False))
        self.part12 = nn.Sequential(UpsampleBLock(16, 2, Use_Transpose_Conv_as_Upsampling_Approach = self.Use_Transpose_Conv_as_Upsampling_Approach),
        nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 3, stride = 1, padding = 1, bias = False))
        self.dropout = nn.Dropout(p=0.5)
	
# =============================================================================
#         "fully connected layers as regressor"
#         self.regressor = nn.Linear(in_features = 1024, out_features = num_classes)
# =============================================================================
        
        "single conv layers as regressor"
        self.regressor = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, stride = 1, padding = 1, bias = False)
        
        
        "initialization by using He's initialization"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# =============================================================================
#                 nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
# =============================================================================
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def make_residual_part(self, Residual_Block_Type, number_in_channel, number_out_channel, number_of_residual_blocks, num_of_group, stride = 1, padding = 0, gate_in_use = True, EPSILON = 1, Use_Batch_Norm = True):
        linear_projection = None        
        if (stride != 1) or (number_in_channel != number_out_channel): 
        #----- in case the stride is NOT 1, or number_in_channel is NOT same as number_out_channel, adapte the size and number of out channel of residual link
            if (Use_Batch_Norm == True):
                linear_projection = nn.Sequential(
                        nn.Conv2d(number_in_channel, number_out_channel, 1, stride, padding = 0, bias = False),
                        nn.BatchNorm2d(number_out_channel))
            else:
                linear_projection = nn.Conv2d(number_in_channel, number_out_channel, 1, stride, padding = 0, bias = False)
            
        parts = []
        parts.append(Residual_Block_Type(number_in_channel, number_out_channel, stride, linear_projection, num_of_group, gate_in_use, EPSILON, Use_Batch_Norm))

        for i in range(1, number_of_residual_blocks):
            print('in and out channels for this residual block is ', number_out_channel)
            parts.append(Residual_Block_Type(number_out_channel, number_out_channel, num_of_group = num_of_group, gate_in_use = gate_in_use, EPSILON = EPSILON, Use_Batch_Norm = Use_Batch_Norm))
        return nn.Sequential(*parts) #----- iteratively pass each element in the list, see https://stackoverflow.com/questions/3480184/unpack-a-list-in-python for detail
        
    "All the forward propagation behavier is implemented in this function"
    def forward(self, x):
        low_resolution_input = x
        if self.feature_extractor_in_front_bool == True:
            x = tc.cat((x, x, x), 1)
        x0 = self.normal_block(x) #----- x --> normal_block --> result stores back in x
#        print('Before part1: ',x0.size())
        x1 = self.part1(x0)
#        print('Part1: ',x1.size())
        x2 = self.part2(x1)
#        print('Part2: ',x2.size())
        x3 = self.part3(x2)
#        print('Part3: ',x3.size())
        x4 = self.part4(x3)
#        print('Part4: ',x4.size())
        x5 = self.part5(x4)
#        print('Part5: ',x5.size())
        x6 = self.part6(x5)
#        print('Part6: ',x6.size())

        x7 = self.part7(x6) + x5
#        print('Part7: ',x7.size())
        x8 = self.part8(x7) + x4
#        print('Part8: ',x8.size())
        x9 = self.part9(x8) + x3
#        print('Part9: ',x9.size())
        x10 = self.part10(x9) + x2
#        print('Part10: ',x10.size())
        x11 = self.part11(x10) + x1
#        print('Part11: ',x11.size())
#        sr_resolution_output = self.part12(x11) + low_resolution_input
	
        sr_resolution_output = self.dropout(self.part12(x11)) + low_resolution_input
#        print('sr_output: ',sr_resolution_output.size())
        
        "No needs for avg poolingbecause the output size from part5 is already N x 256 x 64 x 64(in 256 out channels/feature maps)"
        # x = F.avg_pool3d(x, kernel_size = 7) 
        
        # x = x.view(x.size()[0], -1)                     #----- Reshape the x to make it flat
        
        return sr_resolution_output #----- return the sr_resolution_output which is just right before passing the last activation function layer


        
"Residual_Block_Type is either class 'BottleNeck_Residual_Block' or 'class Normal_Residual_Block'"
device=tc.device("cuda" if use_cuda else "cpu")
# our_resnext = ResNeXt_2D(BottleNeck_Residual_Block, feature_extractor_in_front_bool = Feature_Extractor_in_Front_of_Network).to(device)
our_resnext = ResNeXt_2D(BottleNeck_Residual_Block, feature_extractor_in_front_bool = Feature_Extractor_in_Front_of_Network, Use_Batch_Norm = Use_Batch_Norm, Use_Transpose_Conv_as_Upsampling_Approach = Use_Transpose_Conv_as_Upsampling_Approach)
if tc.cuda.device_count()>1:
    our_resnext=nn.DataParallel(our_resnext)
our_resnext.to(device)





# our_resnext = ResNeXt_3D(Normal_Residual_Block).to(device)
print('this is our ResNeXt: ', our_resnext)

feature_extractor = FeatureExtractor().to(device)
print('this is our FeatureExtractor: ', feature_extractor)

fft_k_space = FFT_K_SPACE().to(device)
print('this is our FFT_K_SPACE: ', fft_k_space)


"""""""""""""""""""""""""""""""""""""""
4. Setup optimization algorithm part
"""""""""""""""""""""""""""""""""""""""
"set an optimizer"
if (Use_Lookahead_Optimizer):
    base_opt = opt.Adam(our_resnext.parameters(), lr=1e-3, betas=(0.9, 0.999)) #----- use Adam algorithm as based optimizer A
    optimizer = lookahead.Lookahead(base_opt, k=5, alpha=0.5) # Initialize Lookahead
else:
    # optimizer = opt.SGD(our_resnext.parameters(), lr = 0.0001, momentum=0.9, weight_decay = 1e-9)    #----- use SGD algorithm for all parameters of our_lenet, by learning rate 0.01 and Momentum is 0.9
    optimizer = opt.Adam(our_resnext.parameters(), lr = 0.0001, eps = 1e-08, weight_decay = 1e-5)    #----- use Adam algorithm for all parameters of our_classifier
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
5. Train the ResNeXt34 part
"""""""""""""""""""""""""""
"Train the ResNeXt34"
tc.set_num_threads(10)  #----- Sets the number of OpenMP threads used for parallelizing CPU operations

for epoch in range(EPOCH_NUM):
    "Set training mode"
    our_resnext.train()
# =============================================================================
#     print('This is the ', epoch, ' epoch')
# =============================================================================
    if (Use_Lookahead_Optimizer):
        optimizer.step()
    else:
        scheduler.step()
    
    running_loss = 0.0
    
    batch_number = 0
    feature_map_loss_test = 0
    pixel_wise_loss_test = 0
    k_space_freq_loss_test = 0
    ssim_loss_test = 0
    gradient_map_loss_test = 0
    test_loss_history = [0]
    
    for i, data in enumerate(trainloader, 0):
# =============================================================================
#         print('This is the ', i, ' batch for the ', epoch, ' epoch' )
# =============================================================================
        "load input data"
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        # print('The data have been loaded' )
        if i<num_training_samples*0.95/batch_size:
            
            "clear all stored gradients if there exist"
            optimizer.zero_grad()
            # print('The optimizer has been cleared' )
            
            "forward prop"
            # outputs = our_resnext(inputs).double() #-- numpy arrays are 64-bit floating point and will be converted to torch.DoubleTensor standardly. Now, if you use them with your model, you'll need to make sure that your model parameters are also Double
            outputs = our_resnext(inputs) #-- or using default float as type, however remember to cast the input from Double to Float
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
#            print("feature_map_loss: ", feature_map_loss)
        
            # pixel_wise_loss = 10*loss_function_MSE(outputs, labels)
            pixel_wise_loss = 100*loss_function_L1(outputs, labels)
#            print("pixel_wise_loss: ", pixel_wise_loss)

            k_space_freq_loss = 0.01*(loss_function_MSE(SR_freq[:,:,:,:,0], HR_freq[:,:,:,:,0])+loss_function_MSE(SR_freq[:,:,:,:,1], HR_freq[:,:,:,:,1]))
#            print(loss_function_MSE(SR_freq[:,:,:,:,0], HR_freq[:,:,:,:,0]))
#            print(loss_function_MSE(SR_freq[:,:,:,:,1], HR_freq[:,:,:,:,1]))
#            print("k_space_freq_loss: ", k_space_freq_loss)

            ssim_loss = loss_function_L1(SSIM_function(labels,labels),SSIM_function(outputs, labels))
#            print("ssim_loss: ", ssim_loss)

            gradient_map_loss = 100*loss_function_L1(calculate_gradient_map(outputs), calculate_gradient_map(labels))
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
        
            if tc.isnan(ssim_loss) != 1:
                loss = loss + ssim_loss

            if tc.isnan(gradient_map_loss) != 1:
                loss = loss + gradient_map_loss
        
#            print('loss: ', loss)
#            loss = feature_map_loss + pixel_wise_loss + k_space_freq_loss
            # print('the loss has been checked')

        
            "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"
#            if tc.isnan(loss) == 1: #- loss == 'NaN':
#                break
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
        else:        
            with tc.no_grad():
                "Set evaluation Mode"    
                our_resnext.eval()
                
                batch_number += 1
                                
                SR_test = our_resnext(inputs)
                
                SR_test_copies = tc.cat((SR_test, SR_test, SR_test), 1)
                # print(SR_copies.size())
                SR_test_features = feature_extractor(SR_test_copies)
                # print(SR_features.size())
                
                HR_test_copies = tc.cat((labels, labels, labels), 1)
                # print(HR_copies.size())
                HR_test_features = feature_extractor(HR_test_copies)
                # print(HR_features.size())
            
                SR_test_freq = fft_k_space(SR_test)
            
                HR_test_freq = fft_k_space(labels)
            
            
                "calculate the gradients for all Variables during back prop"
                "vgg loss + pixel MSE loss + fft frequency loss, and we use weight_decay in Adam so that is L2 regularization"
                feature_map_loss_test += 0.01*loss_function_MSE(SR_test_features, HR_test_features)
#               print("feature_map_loss_test: ", feature_map_loss_test)
                
                pixel_wise_loss_test += 100*loss_function_L1(SR_test, labels)
#                print("pixel_wise_loss_test: ", pixel_wise_loss_test)
                
                k_space_freq_loss_test += 0.01*(loss_function_MSE(SR_test_freq[:,:,:,:,0], HR_test_freq[:,:,:,:,0])+loss_function_MSE(SR_test_freq[:,:,:,:,1], HR_test_freq[:,:,:,:,1]))
#                print("k_space_freq_loss_test: ", k_space_freq_loss_test)
                
                ssim_loss_test += loss_function_L1(SSIM_function(labels,labels),SSIM_function(SR_test, labels))
#                print("ssim_loss_test: ", ssim_loss_test)
                
                gradient_map_loss_test += 100*loss_function_L1(calculate_gradient_map(SR_test), calculate_gradient_map(labels))
#                print('gradient_loss_test: ', gradient_map_loss_test)
                
                loss_test = pixel_wise_loss_test + feature_map_loss_test + k_space_freq_loss_test + ssim_loss_test + gradient_map_loss_test
#                print('loss_test: ', loss_test)
                
    feature_map_loss_test = feature_map_loss_test/batch_number
    print("feature_map_loss_test: ", feature_map_loss_test)
    pixel_wise_loss_test = pixel_wise_loss_test/batch_number
    print("pixel_wise_loss_test: ", pixel_wise_loss_test)
    k_space_freq_loss_test = k_space_freq_loss_test/batch_number
    print("k_space_freq_loss_test: ", k_space_freq_loss_test)
    ssim_loss_test = ssim_loss_test/batch_number
    print("ssim_loss_test: ", ssim_loss_test)
    gradient_map_loss_test = gradient_map_loss_test/batch_number
    print('gradient_loss_test: ', gradient_map_loss_test)
    loss_test = loss_test/batch_number
    print('loss_test: ', loss_test)

    "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"    
#    if tc.isnan(loss) == 1: #- loss == 'NaN':
#        break
    "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"
    
    
    "Save the training loss for each epoch"
    if (epoch == 0):
        f = open('result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_150_16_4folds_2d_test.txt', 'w')
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
    f.write('----------------------------------------------------------------------------------')
    f.write(' \n')
    f.write(' \n')
    if (epoch == EPOCH_NUM - 1):
        f.close()
            
print("training complete")






"Evaluation"
folder_log_path = '/home/HaoLi/SR/Data/'
file_names = os.listdir(folder_log_path)

num_low_resolution_mat_file = 0
num_high_resolution_groundtruth_mat_file = 0

for idx_file in file_names:
    print(idx_file)
    if 'LR_eval' in os.path.join(folder_log_path, idx_file):
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
    elif 'HRGT_eval' in os.path.join(folder_log_path, idx_file):
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

new_batch_size_for_checking = 16

testloader = tc.utils.data.DataLoader(
                    testset, 
                    batch_size = batch_size,
                    shuffle = False, 
                    num_workers = 0)        


"Resetup the batch size for train and test data, to avoid the errorCUDA out of memory"
trainloader = tc.utils.data.DataLoader(
                    trainset, 
                    batch_size = new_batch_size_for_checking,
                    shuffle = True, 
                    num_workers = 0)

with tc.no_grad():
    our_resnext.eval()
    "exam the generated SR MRI image by using training LR image data and save them"
    # =============================================================================
    # for data in testloader:
    # =============================================================================
    for i, training_data_2 in enumerate(trainloader, 0):
        
        #    LR_images_training, HR_images_training = training_data_2
        
        # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
        #    outputs = our_rcan_mri_sr_2d(Variable(LR_images_training).type(tc.FloatTensor).to(device))
        # =============================================================================
        #     print(outputs.data.size())
        # =============================================================================
        
        if (i == math.floor((torch_data_low_resolution_training_sequence.size(0)/new_batch_size_for_checking)/2)): 
            
            LR_images_training, HR_images_training = training_data_2
            # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
            outputs = our_resnext(Variable(LR_images_training).type(tc.FloatTensor).to(device))
        
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
            scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_150_16_4folds_2d_test/HR_training_image.mat', mdict = {'HR_training_image' : HR_images_training.numpy()})
            scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_150_16_4folds_2d_test/LR_training_image.mat', mdict = {'LR_training_image' : LR_images_training.numpy()})
            scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_150_16_4folds_2d_test/SR_training_image.mat', mdict = {'SR_training_image' : SR_images_exam_train})
            
            
            
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
    
        #    LR_images_test, HR_images_test = testing_data_2
        # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
        #    outputs = our_rcan_mri_sr_2d(Variable(LR_images_test).type(tc.FloatTensor).to(device))
        # =============================================================================
        #     print(outputs.data.size())
        # =============================================================================
    
#       if (i == math.floor((torch_data_low_resolution_test_sequence.size(0)/new_batch_size_for_checking)/2)): 
    
        LR_images_test, HR_images_test = testing_data_2
        HR_images_test = HR_images_test.type(tc.FloatTensor)
        outputs = our_resnext(Variable(LR_images_test).type(tc.FloatTensor).to(device))
        
        #----- skip display "the last batch for one epoch test data" and skip "all the batches expect the batch in the middle"
        SR_images_tensor_test = outputs.data.cpu().squeeze(1)
        LR_images_tensor_test = LR_images_test.cpu().squeeze(1)
        HR_images_tensor_test = HR_images_test.cpu().squeeze(1)
        # HR_images_tensor = HR_images_temp.cpu().squeeze(1)
        # =============================================================================
        #         print(SR_images_tensor.size())
        # =============================================================================
        # print(HR_images_tensor.size())    
        
        # HR_images_test = HR_images_tensor.numpy()
        if i==0:
            SR_eval_tensor = SR_images_tensor_test
            LR_eval_tensor = LR_images_tensor_test
            HR_eval_tensor = HR_images_tensor_test
        else:
            SR_eval_tensor = tc.cat((SR_eval_tensor, SR_images_tensor_test), 0)
            LR_eval_tensor = tc.cat((LR_eval_tensor, LR_images_tensor_test), 0)
            HR_eval_tensor = tc.cat((HR_eval_tensor, HR_images_tensor_test), 0)
        
    SR_images_test = SR_eval_tensor.numpy()
    LR_images_test = LR_eval_tensor.numpy()
    HR_images_test = HR_eval_tensor.numpy()
    "save the .mat files for SR, HR and LR training images"
    scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_150_16_4folds_2d_test/SR_test_image.mat', mdict = {'SR_test_image' : SR_images_test})
    scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_150_16_4folds_2d_test/HR_test_image.mat', mdict = {'HR_test_image' : HR_images_test})
    scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_150_16_4folds_2d_test/LR_test_image.mat', mdict = {'LR_test_image' : LR_images_test})
        
    
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

















# =============================================================================
# "exam the generated SR MRI image by using training LR image data and save them"
# "initialization"
# i = 0
# for training_data_i in trainloader:
#     
#     if i == SELECTED_BATCH_FOR_PLOT_AND_SAVE_MAT_FILE:
#         LR_images_training_i, HR_images_training_i = training_data_i
#         print(LR_images_training_i.size())
#         print(HR_images_training_i.size())
#         
#         outputs = our_resnext(Variable(LR_images_training_i).type(tc.FloatTensor).to(device))
#         print(outputs.size())
#         
#         SR_images_tensor_training_i = outputs.data.cpu().squeeze(1)
#         print(SR_images_tensor_training_i.size())
#            
#         "save the .mat files for SR LR, HR training images"
#         scipy.io.savemat('./SR_data/HR_training_image.mat', mdict = {'HR_training_image' : HR_images_training_i.squeeze(1).numpy()})
#         scipy.io.savemat('./SR_data/LR_training_image.mat', mdict = {'LR_training_image' : LR_images_training_i.squeeze(1).numpy()})
#         scipy.io.savemat('./SR_data/SR_training_image.mat', mdict = {'SR_training_image' : SR_images_tensor_training_i.numpy()})
#         
#         for j in range(batch_size):
#             plt.imshow(SR_images_tensor_training_i.numpy()[j, :, :])
#             plt.savefig('./SR_plot/training_' + str(j) + '_SR_image.png')
#             plt.show()
#             plt.imshow(HR_images_training_i.squeeze(1).numpy()[j, :, :])
#             plt.savefig('./SR_plot/training_' + str(j) + '_HR_image.png')
#             plt.show()
#             plt.imshow(LR_images_training_i.squeeze(1).numpy()[j, :, :])
#             plt.savefig('./SR_plot/training_' + str(j) + '_LR_image.png')
#             plt.show()
#     
#     i = i + 1
# print("examination of generated SR image by using training samples complete")
# 
# 
# 
# 
# "predict the SR MRI image by using testing LR image data and save them"
# "initialization"
# i = 0
# for test_data_i in testloader:
#     
#     if i == SELECTED_BATCH_FOR_PLOT_AND_SAVE_MAT_FILE:
#         LR_images_test_i, HR_images_test_i = test_data_i
#         print(LR_images_test_i.size())
#         print(HR_images_test_i.size())
#         
#         outputs = our_resnext(Variable(LR_images_test_i).type(tc.FloatTensor).to(device))
#         print(outputs.size())
#         
#         SR_images_tensor_test_i = outputs.data.cpu().squeeze(1)
#         print(SR_images_tensor_test_i.size())
#            
#         "save the .mat files for SR LR, HR test images"
#         scipy.io.savemat('./SR_data/HR_test_image.mat', mdict = {'HR_test_image' : HR_images_test_i.squeeze(1).numpy()})
#         scipy.io.savemat('./SR_data/LR_test_image.mat', mdict = {'LR_test_image' : LR_images_test_i.squeeze(1).numpy()})
#         scipy.io.savemat('./SR_data/SR_test_image.mat', mdict = {'SR_test_image' : SR_images_tensor_test_i.numpy()})
#         
#         for j in range(batch_size):
#             plt.imshow(SR_images_tensor_test_i.numpy()[j, :, :])
#             plt.savefig('./SR_plot/test_' + str(j) + '_SR_image.png')
#             plt.show()
#             plt.imshow(HR_images_test_i.squeeze(1).numpy()[j, :, :])
#             plt.savefig('./SR_plot/test_' + str(j) + '_HR_image.png')
#             plt.show()
#             plt.imshow(LR_images_test_i.squeeze(1).numpy()[j, :, :])
#             plt.savefig('./SR_plot/test_' + str(j) + '_LR_image.png')
#             plt.show()
#     
#     i = i + 1
# print("the predicting of generated SR image by using testing samples complete")
# =============================================================================







#with tc.no_grad():
#    "Set evaluation mode"
#    our_resnext.eval()
#    "Resetup the batch size for train and test data, to avoid the errorCUDA out of memory"
#    new_batch_size_for_checking = 16
#    trainloader = tc.utils.data.DataLoader(
#                        trainset, 
#                        batch_size = new_batch_size_for_checking,
#                        shuffle = True, 
#                        num_workers = 0)

#    testloader = tc.utils.data.DataLoader(
#                        testset, 
#                        batch_size = new_batch_size_for_checking,
#                        shuffle = True, 
#                        num_workers = 0)


#    "exam the generated SR MRI image by using training LR image data and save them"
    # =============================================================================
    # for data in testloader:
    # =============================================================================
#    for i, training_data_2 in enumerate(trainloader, 0):
    
        #    LR_images_training, HR_images_training = training_data_2
    
        # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
        #    outputs = our_resnext(Variable(LR_images_training).type(tc.FloatTensor).to(device))
        # =============================================================================
        #     print(outputs.data.size())
        # =============================================================================
    
#        if (i == math.floor((torch_data_low_resolution_training_sequence.size(0)/new_batch_size_for_checking)/2)): 
        
#            LR_images_training, HR_images_training = training_data_2
            # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
#            outputs = our_resnext(Variable(LR_images_training).type(tc.FloatTensor).to(device))
        
            #----- skip display "the last batch for one epoch test data" and skip "all the batches expect the batch in the middle"
#            SR_images_tensor_training = outputs.data.cpu().squeeze(1)
            # HR_images_tensor = HR_images_temp.cpu().squeeze(1)
            # =============================================================================
            #         print(SR_images_tensor.size())
            # =============================================================================
            # print(HR_images_tensor.size())    
#            SR_images_exam_train = SR_images_tensor_training.numpy()
            # HR_images_test = HR_images_tensor.numpy()
        
#            "save the .mat files for SR LR, HR training images"
#            scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_200_32_4folds_2d_test/HR_training_image.mat', mdict = {'HR_training_image' : HR_images_training.numpy()})
#            scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_200_32_4folds_2d_test/LR_training_image.mat', mdict = {'LR_training_image' : LR_images_training.numpy()})
#            scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_200_32_4folds_2d_test/SR_training_image.mat', mdict = {'SR_training_image' : SR_images_exam_train})
        
        
    
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
    



#    "predict the SR MRI image by using testing LR image data and save them"
    # =============================================================================
    # for data in testloader:
    # =============================================================================
#    for i, testing_data_2 in enumerate(testloader, 0):
    
        #    LR_images_test, HR_images_test = testing_data_2
        # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
        #    outputs = our_resnext(Variable(LR_images_test).type(tc.FloatTensor).to(device))
        # =============================================================================
        #     print(outputs.data.size())
        # =============================================================================
    
#        if (i == math.floor((torch_data_low_resolution_test_sequence.size(0)/new_batch_size_for_checking)/2)): 
            
#            LR_images_test, HR_images_test = testing_data_2
            # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
#            outputs = our_resnext(Variable(LR_images_test).type(tc.FloatTensor).to(device))
        
            #----- skip display "the last batch for one epoch test data" and skip "all the batches expect the batch in the middle"
#            SR_images_tensor_test = outputs.data.cpu().squeeze(1)
            # HR_images_tensor = HR_images_temp.cpu().squeeze(1)
            # =============================================================================
            #         print(SR_images_tensor.size())
            # =============================================================================
            # print(HR_images_tensor.size())    
#            SR_images_test = SR_images_tensor_test.numpy()
            # HR_images_test = HR_images_tensor.numpy()
        
#            "save the .mat files for SR, HR and LR training images"
#            scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_200_32_4folds_2d_test/SR_test_image.mat', mdict = {'SR_test_image' : SR_images_test})
#            scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_200_32_4folds_2d_test/HR_test_image.mat', mdict = {'HR_test_image' : HR_images_test.numpy()})
#            scipy.io.savemat('/home/HaoLi/SR/Results/result_UResNeXt_l1_4ssim_gradient_bnf_tcf_laf_200_32_4folds_2d_test/LR_test_image.mat', mdict = {'LR_test_image' : LR_images_test.numpy()})
        
    
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

#    print("the predicting of generated SR image by using testing samples complete")





now = time.clock()

running_time = now - since
print('The spent time in minute is: ', running_time/60)
