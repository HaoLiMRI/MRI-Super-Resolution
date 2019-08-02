"-------------------------------------------------------------------------------------------------"
"""
2D_DDBPN_Based_MRI_SR Reconstruct (2D Dense Deep Back-Projection Network Based MRI Super-Resolution Image Reconstruction)
This is the code for DDBPN based MRI SR Reconstruction, DDBPN model proposed to generate super resolution(SR) image in the paper: 
2018. Deep Back-Projection Networks For Super-Resolution
https://arxiv.org/abs/1803.02735
"""
"""
Author: chisyliu@hotmail.com *
        hao.li@med.uni-heidelberg.de *
        
        * Both authors contribute equally
Version: 1.0.0
"""
"-------------------------------------------------------------------------------------------------"
"""
This is the current version we are working on, in 20190730
This is a demo code of 2D_DDBPN_Based_MRI_SR.
    0) This is the 2D version of 2D_DDBPN_Based_MRI_SR, in this version we have following items:
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
    [19]2018 Deep Back-Projection Networks For Super-Resolution. https://arxiv.org/abs/1803.02735
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

import pytorch_ssim
from optimizer import lookahead

"-------------------------------------------------------------------------------------------------"
print('boolean value to see if GPU is ready:', tc.cuda.is_available())
print('number of GPU is', tc.cuda.device_count())
print(tc.cuda.get_device_name(0))
use_cuda = True #-- boolean to choose GPU
since = time.clock()


batch_size = 32
EPOCH_NUM = 250
SELECTED_BATCH_FOR_PLOT_AND_SAVE_MAT_FILE = 10
Feature_Extractor_in_Front_of_Network = False
Use_Lookahead_Optimizer = True
Maintain_Same_Size = True # stand for whether we want the output SR Simage has same size or NOT(e.g. larger size) as input LR image


# --------------------------- configuration of parameters for RDN --------------------------- #
args = {'scale': 2, 'n_colors': 1}
# args['scale'] = 2, stands for "scale factor" for downsize/upsize in the half of our own 2D_RDN_Based_MRI_SR model
# args['n_colors'] = 1, stands for the number of channels of input LR image of our own 2D_RDN_Based_MRI_SR model


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
In the original paper which proposed DDBPN(2018. Deep Back-Projection Networks For Super-Resolution, mentioned as "original DBPN paper" in following), there are some important facts:
    1. DDBNP actually avoid using batch norm in the network. They point out: "Unlike the original DenseNets, we avoid dropout and batch norm, which are not suitable for SR, because they remove the range flexibility of the features [31]. Instead, we use 1 x 1 
       convolution layer as feature pooling and dimensional reduction [42, 12] before entering the projection unit"
    2. DDBNP take transpose conv for upsampling, NOT sub-pixel conv for upsampling as RDN use
    3. DDBPN follows the principle of "iterative up and downsampling", NOT follows the principle of "single upsampling" or "progressive upsampling" as RDN use and "predefined upsampling" as used in SRCNN
"""

"""""""""""""""""""""""""""""""""""""""
3. Define ResNeXt architecture part
"""""""""""""""""""""""""""""""""""""""
"Projection conv layer for either upsampling and downsampling"
def projection_conv(in_channels, out_channels, scale, up=True):
    """
    projection_conv: conv layer which either shriks or expands the size(height H, width W, of image) of image. 
    When "shriking", it is just normal conv layer. When "expanding", it is acutal the upsampling approach based on transpose convolution
    """
    kernel_size, stride, padding = {
        # W2=(W1−F+2P)/S+1, H2=(H1−F+2P)/S+1. For this particular configuration (kernel_size = 6, stride = 2, padding = 2) it means shriks 1/2 for normal conv layer, and expands/upscales 2 for transpose conv layers
        2: (6, 2, 2),
        #  For this particular configuration (kernel_size = 8, stride = 4, padding = 2) it means shriks 1/4 for normal conv layer, and expands/upscales 4 for transpose conv layers
        4: (8, 4, 2), 
        # For this particular configuration (kernel_size = 12, stride = 8, padding = 2) it means shriks 1/8 for normal conv layer, and expands/upscales 8 for transpose conv layers
        8: (12, 8, 2)
    }[scale]
    if up:
        # implementing upsampling using "transpose conv" which was proposed originally in paper: .
        # Note: It is different upsampling approach rather than "sub-pixel conv"
        conv_f = nn.ConvTranspose2d
    else:
        # conv layer, shrik the size of image
        conv_f = nn.Conv2d
    return conv_f(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )


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


"Dense projection unit"
class DenseProjection(nn.Module):
    """
    DenseProjection: Dense Up Projection Unit or Dense Down Projection Unit. It is as figure 4 in original DBPN paper. There are several Dense Projection Units in the DDBPN. 
    """
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck: # if bottleneck = True, pass through the 1x1 conv layer(dark grey block in figure 4 of DBPN paper)
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        # if up=True, conv_1, 2, 3 will be upsampling(transpose conv), downsizing(conv), upsampling(transpose conv), which are blue block, light green block, blue block in upper figure in figure 4 of DBPN paper; 
        # otherwise if up=False, conv_1, 2, 3 will be downsizing(conv), upsampling(transpose conv), downsizing(conv), which are light green block, blue block, light green block in bottom figure in figure 4 of DBPN paper
        self.conv_1 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.conv_2 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels)
        ])
        self.conv_3 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])

    def forward(self, x):
        """
        implement the flow of Dense Up/Down Projection Unit as figure 4 in original DBPN paper
        """
        # if bottleneck is True, pass into 1x1 conv(dark grey block)
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        # first pass into transpose conv(blue block) to upsampling if up = True, pass into conv(ligh green block) to downsize if up = False 
        a_0 = self.conv_1(x)
        # secondly pass into pass into conv(ligh green block) to downsize if up = Up, transpose conv(blue block) to upsampling if up = False
        b_0 = self.conv_2(a_0)
        # then minus output form 1x1 conv according to figure 4
        e = b_0.sub(x)
        # lately pass into transpose conv(blue block) to upsampling if up = True, pass into conv(ligh green block) to downsize if up = False
        a_1 = self.conv_3(e)
        # add the output from conv_1 according to figure 4
        out = a_0.add(a_1)

        return out


"Dense deep back projection network for MRI SR 2D"
class DDBPN_MRI_SR_2D(nn.Module):
    """
    Dense Deep Back-Projection Network(DDBPN) for MRI SR 2D. It is almost same as figure 5 in original DBPN paper
    """
    def __init__(self, args):
        """
        args: Set of specification of arguements, which have been defined outside ddbnp.py
              it includes several arguements, e.g.
              scale: upscale factor
              n_colors: the number of channels of input image, e.g. 3 for R,G,B
        """
        super(DDBPN_MRI_SR_2D, self).__init__()
        scale = args['scale']
        n_colors = args['n_colors']

        n0 = 128
        nr = 32
        self.depth = 6 # how many "up projection units" expected to be in DDBPN(number of "down projection units" will be depth - 1)

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        # build up "Initial Feature Extraction Module" in figure 5
        initial = [
            nn.Conv2d(n_colors, n0, 3, padding=1), # first 3x3 conv layer(light green block) in figure 5
            nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1), # second 1x1 conv layer(light green block) in figure 5
            nn.PReLU(nr)
        ]
        self.initial = nn.Sequential(*initial)

        # build up "Back-Projection stages Module" in figure 5， but with a little change that is "in our application here we consider adding extra downsize converter using conv layer which could shrik 
        # 1/scale of output from Back-Projection Stages, cause we need to build up the model which generates the final output image with same size as input image to the model
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(
                DenseProjection(channels, nr, scale, True, i > 1) # build up a list "upmodules" which contains several up projection units, e.g. 6 up projection units in case self.depth = 6
            )
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(
                DenseProjection(channels, nr, scale, False, i != 0) # build up a list "downmodules" which contains several down projection units, e.g. 5 down projection units in case self.depth = 6
            )
            channels += nr

        # Add a downsize converter by using conv layer.
        # The purpose of original RDN designed in original RDN paper is to upscale r times of LR image, the output from global residual learning module has same size as input image. However, here in our task
        # we actually want to maintain output(SR) and input(LR) same as, it means we need to downsize the image before leave it into upscale module. That is the reason we need extra conv layer to perform 
        # down size with r time first before going through upscale with r time
        if (scale == 2):
            # W2=(W1−F+2P)/S+1, H2=(H1−F+2P)/S+1.
            self.down_size_converter = nn.Conv2d(self.depth * nr, self.depth * nr, 3, padding=1, stride=2) 
        elif (scale == 4):
            self.down_size_converter = nn.Sequential(*[
                nn.Conv2d(self.depth * nr, self.depth * nr, 3, padding=1, stride=2),
                nn.Conv2d(self.depth * nr, self.depth * nr, 3, padding=1, stride=2)
            ])
        elif (scale == 8):
            self.down_size_converter = nn.Sequential(*[
                nn.Conv2d(self.depth * nr, self.depth * nr, 3, padding=1, stride=2),
                nn.Conv2d(self.depth * nr, self.depth * nr, 3, padding=1, stride=2),
                nn.Conv2d(self.depth * nr, self.depth * nr, 3, padding=1, stride=2)
            ])
        else:
            raise ValueError("scale must be 2, or 4 or 8.")

        # build up "Reconstruction Module" in figure 5
        reconstruction = [
            nn.Conv2d(self.depth * nr, n_colors, 3, padding=1) # 3x3 conv layer. the size of image has already been downsized as expected, so just downsize the number of channels to n_colors
        ]
        self.reconstruction = nn.Sequential(*reconstruction)

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1) """
        # ----------------------------------------------------------------------------------------------------------------------- #

    def forward(self, x):
        """
        implement the flow of Dense Deep Back-Projection Projection Network(DDBPN) as figure 5 in original DBPN paper
        """
        # substraction mean first, later add mean. I don't understand the purpose
        """ x = self.sub_mean(x) """

        # let the LR image to pass into initial feature extraction module
        x = self.initial(x)

        # ------------------------------------------------ implement the flow of "Back-Projection Stages" acoording to figure 5 ------------------------------------------------------- #
        h_list = []
        l_list = []
        for i in range(self.depth - 1): # data flow for the first "depth - 1, e.g. 5 when depth = 6" up projection units and "depth - 1, e.g. 5 when depth = 6" down projection units 
            if i == 0:
                l = x
            else:
                l = tc.cat(l_list, dim=1) # concatenate all the outputs from down projection units already passed through so far
            h_list.append(self.upmodules[i](l)) # leave the data which "concatenated all the outputs from down projection units already passed through so far" into current up projection unit in the list "upmodules"
            h = tc.cat(h_list, dim=1) # concatenate all the outputs from up projection units already passed through so far
            l_list.append(self.downmodules[i](h)) # leave the data which "concatenated all the outputs from up projection units already passed through so far" into current down projection unit in the list "downmodules"
        h_list.append(self.upmodules[-1](tc.cat(l_list, dim=1))) # last up projection units, concatenate all the outputs from all down projection units in list "upmodules", leave into the final up projection unit
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

		if (Maintain_Same_Size == True):
            downsized_concatenated_out = self.down_size_converter(tc.cat(h_list, dim=1)) # concatenate all the output from each up projection unit in list "upmodules" and leave them into down_size_converter module 
            out = self.reconstruction(downsized_concatenated_out) # into reconstruction module to lower the number of channels
        else:
            out = self.reconstruction(tc.cat(h_list, dim=1)) # into reconstruction module to lower the number of channels
        
        # substraction mean in the very early stage, now add mean. I don't understand the purpose
        """ out = self.add_mean(out) """

        return out



device=tc.device("cuda" if use_cuda else "cpu")
our_ddbpn_mri_sr_2d = DDBPN_MRI_SR_2D(args)
if tc.cuda.device_count()>1:
    our_ddbpn_mri_sr_2d=nn.DataParallel(our_ddbpn_mri_sr_2d)
our_ddbpn_mri_sr_2d.to(device)
print('this is our DDBPN_MRI_SR_2D: ', our_ddbpn_mri_sr_2d)

feature_extractor = FeatureExtractor().to(device)
print('this is our FeatureExtractor: ', feature_extractor)

fft_k_space = FFT_K_SPACE().to(device)
print('this is our FFT_K_SPACE: ', fft_k_space)


"""""""""""""""""""""""""""""""""""""""
4. Setup optimization algorithm part
"""""""""""""""""""""""""""""""""""""""
"set an optimizer"
if (Use_Lookahead_Optimizer):
    base_opt = opt.Adam(our_ddbpn_mri_sr_2d.parameters(), lr=1e-3, betas=(0.9, 0.999)) #----- use Adam algorithm as based optimizer A
    optimizer = lookahead.Lookahead(base_opt, k=5, alpha=0.5) # Initialize Lookahead
else:
    # optimizer = opt.SGD(our_resnext.parameters(), lr = 0.0001, momentum=0.9, weight_decay = 1e-9)    #----- use SGD algorithm for all parameters of our_lenet, by learning rate 0.01 and Momentum is 0.9
    optimizer = opt.Adam(our_ddbpn_mri_sr_2d.parameters(), lr = 0.0001, eps = 1e-08, weight_decay = 1e-9)    #----- use Adam algorithm for all parameters of our_classifier
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

"set a loss function"
print('The loss function is MSE')
loss_function_MSE = nn.MSELoss().to(device)        #----- here use MSE loss

print('The loss function is L1')
loss_function_L1 = nn.SmoothL1Loss().to(device)       #----- smooth L1 loss

# print('The loss function is Cross Entropy')
# loss_function_CE = nn.CrossEntropyLoss().to(device)

print('The loss function is MS-SSIM')
SSIM_function = pytorch_ssim.SSIM().to(device)       #----- ssim loss

# =============================================================================
# print('The loss function is L1Loss')
# loss_function = nn.L1Loss(size_average = False).to(device) 
# =============================================================================
# =============================================================================
# print('The loss function is CrossEntropyLoss')
# loss_function = nn.CrossEntropyLoss().to(device)        #----- here use cross entropy loss
# =============================================================================



"""""""""""""""""""""""""""
5. Train the our_ddbpn_mri_sr_2d part
"""""""""""""""""""""""""""
"Train the our_ddbpn_mri_sr_2d"
tc.set_num_threads(10)  #----- Sets the number of OpenMP threads used for parallelizing CPU operations
for epoch in range(EPOCH_NUM):
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
        # outputs = our_ddbpn_mri_sr_2d(inputs).double() #-- numpy arrays are 64-bit floating point and will be converted to torch.DoubleTensor standardly. Now, if you use them with your model, you'll need to make sure that your model parameters are also Double
        outputs = our_ddbpn_mri_sr_2d(inputs) #-- or using default float as type, however remember to cast the input from Double to Float
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
        feature_map_loss = 0.001*loss_function_MSE(SR_features, HR_features)
        # feature_map_loss = 0.000000001*loss_function_CE(SR_features, HR_features)
        print("feature_map_loss: ", feature_map_loss)
        # pixel_wise_loss = 10*loss_function_MSE(outputs, labels)
        pixel_wise_loss = 10*loss_function_L1(outputs, labels)
        
        print("pixel_wise_loss: ", pixel_wise_loss)
        k_space_freq_loss = 0.001*(loss_function_MSE(SR_freq[:,:,:,:,0], HR_freq[:,:,:,:,0])+loss_function_MSE(SR_freq[:,:,:,:,1], HR_freq[:,:,:,:,1]))
#        print(loss_function_MSE(SR_freq[:,:,:,:,0], HR_freq[:,:,:,:,0]))
#        print(loss_function_MSE(SR_freq[:,:,:,:,1], HR_freq[:,:,:,:,1]))
        print("k_space_freq_loss: ", k_space_freq_loss)
        ssim_loss = 1-SSIM_function(outputs, labels)
        print("ssim_loss: ", ssim_loss)
        
#        loss = pixel_wise_loss + ssim_loss
#        loss = pixel_wise_loss + feature_map_loss
        
        if ssim_loss < 0.5:
            loss = ssim_loss + feature_map_loss + pixel_wise_loss
            print('ssim_loss')
        else:
            loss = pixel_wise_loss + feature_map_loss
            
        if tc.isnan(k_space_freq_loss) != 1:
            loss = loss + k_space_freq_loss
        
#        if tc.isnan(ssim_loss) != 1:
#            loss = loss + ssim_loss
        
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
            k_space_freq_loss_for_current_epoch = k_space_freq_loss
            
            running_loss = 0.0
        
    "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"    
#    if tc.isnan(loss) == 1: #- loss == 'NaN':
#        break
    "added code to prevent 'NaN' in loss, just a work around but not final/correct solution"
    
    
    "Save the training loss for each epoch"
    if (epoch == 0):
        f = open('result_RDN_MRI_SR_2D_all_epoch.txt', 'w')
    f.write('The feature_map_loss for epoch %d  is : %f' % (epoch, feature_map_loss_for_current_epoch))
    f.write('\n')
    f.write('The pixel_wise_loss for epoch %d  is : %f' % (epoch, pixel_wise_loss_for_current_epoch))
    f.write('\n')
    f.write('The k_space_freq_loss for epoch %d  is : %f' % (epoch, k_space_freq_loss_for_current_epoch))
    f.write('\n')
    f.write('The ssim_loss for epoch %d  is : %f' % (epoch, ssim_loss_for_current_epoch))
    f.write('\n')
    f.write('The training loss for epoch %d  is : %f' % (epoch, training_loss_for_current_epoch))
    f.write('\n')
    f.write(' \n')
    if (epoch == EPOCH_NUM - 1):
        f.close()
            
print("training complete")










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
    outputs = our_ddbpn_mri_sr_2d(Variable(LR_images_training).type(tc.FloatTensor).to(device))
# =============================================================================
#     print(outputs.data.size())
# =============================================================================
    
    if (i == math.floor((torch_data_low_resolution_training_sequence.size(0)/new_batch_size_for_checking)/2)): 
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
        scipy.io.savemat('/home/HaoLi/SR/Results/HR_training_image.mat', mdict = {'HR_training_image' : HR_images_training.numpy()})
        scipy.io.savemat('/home/HaoLi/SR/Results/LR_training_image.mat', mdict = {'LR_training_image' : LR_images_training.numpy()})
        scipy.io.savemat('/home/HaoLi/SR/Results/SR_training_image.mat', mdict = {'SR_training_image' : SR_images_exam_train})
        
        
    
        for j in range(new_batch_size_for_checking):
            plt.imshow(SR_images_exam_train[j, :, :])
            plt.savefig('/home/HaoLi/SR/Results/training_' + str(j) + '_SR_image.png')
            plt.show()
            plt.imshow(HR_images_training[j, 0, :, :])
            plt.savefig('/home/HaoLi/SR/Results/training_' + str(j) + '_HR_image.png')
            plt.show()
            plt.imshow(LR_images_training[j, 0, :, :])
            plt.savefig('/home/HaoLi/SR/Results/training_' + str(j) + '_LR_image.png')
            plt.show()

print("examination of generated SR image by using training samples complete")




"predict the SR MRI image by using testing LR image data and save them"
# =============================================================================
# for data in testloader:
# =============================================================================
for i, testing_data_2 in enumerate(testloader, 0):
    
    LR_images_test, HR_images_test = testing_data_2
    # HR_images_temp = HR_images.type(tc.LongTensor).to(device)
    outputs = our_ddbpn_mri_sr_2d(Variable(LR_images_test).type(tc.FloatTensor).to(device))
# =============================================================================
#     print(outputs.data.size())
# =============================================================================
    
    if (i == math.floor((torch_data_low_resolution_test_sequence.size(0)/new_batch_size_for_checking)/2)): 
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
        scipy.io.savemat('/home/HaoLi/SR/Results/SR_test_image.mat', mdict = {'SR_test_image' : SR_images_test})
        scipy.io.savemat('/home/HaoLi/SR/Results/HR_test_image.mat', mdict = {'HR_test_image' : HR_images_test.numpy()})
        scipy.io.savemat('/home/HaoLi/SR/Results/LR_test_image.mat', mdict = {'LR_test_image' : LR_images_test.numpy()})
        
    
        for j in range(new_batch_size_for_checking):
            plt.imshow(SR_images_test[j, :, :])
            plt.savefig('/home/HaoLi/SR/Results/testing_' + str(j) + '_SR_image.png')
            plt.show()            
            plt.imshow(HR_images_test[j, 0, :, :])
            plt.savefig('/home/HaoLi/SR/Results/testing_' + str(j) + '_HR_image.png')
            plt.show()            
            plt.imshow(LR_images_test[j, 0, :, :])
            plt.savefig('/home/HaoLi/SR/Results/testing_' + str(j) + '_LR_image.png')
            plt.show()            

print("the predicting of generated SR image by using testing samples complete")





now = time.clock()

running_time = now - since
print('The spent time in minute is: ', running_time/60)
