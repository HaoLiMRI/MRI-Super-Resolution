"""
Because of different importances of three factors, 
LSSIM is calculated by sum up the weighted logarithms of luminance, contrast and structure factors.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    # define gaussian distribution for the window
    
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    # create window (convolution kernel) with pre-defined gaussian distribution
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _lssim(img1, img2, window, window_size, channel, size_average = True):
    # calculte LSSIM in each window
    # theory and equations can be found in: 
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.1939&rep=rep1&type=pdf
    
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)        # Mean of image1 in windows
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)        # Mean of image2 in windows

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq        # Variance of image1 
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq        # Variance of image2
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2         # Covariance of image1 and image2

                        #==============================================#
    C1 = 0.01**2        # should be (0.01*dynamic range)^2             #
    C2 = 0.03**2        # should be (0.03*dynamic range)^2             #
    C3 = C2/2           # since data is normalized, dynamic range is 1 #
                        #==============================================#
    
    weight_luminance = 1        # weighting of luminance_factor
    weight_contrast = 2         # weighting of contrast_factor
    weight_structure = 4        # weighting of structure_factor

    # calculate lunimance_factor, contrast_factor and structure_factor, and rescale to (0,1) to avoid negative values    
    luminance_factor = (2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)/2 + 0.5                                              
    contrast_factor = (2*torch.sqrt(sigma1_sq)*torch.sqrt(sigma2_sq)+C2)/(sigma1_sq + sigma2_sq + C2)/2 + 0.5       
    structure_factor = (sigma12 + C3)/(torch.sqrt(sigma1_sq)*torch.sqrt(sigma2_sq)+C3)/2 + 0.5                      
    print('Luminance: ',luminance_factor.mean(),'. log: ',torch.log(luminance_factor).mean())
    print('Contrast: ',contrast_factor.mean(),'. log: ',torch.log(contrast_factor).mean())
    print('Structure: ',structure_factor.mean(),'. log: ',torch.log(structure_factor).mean())

    lssim_map = 0
    
    if torch.isnan(torch.log(luminance_factor).mean())!=1 and  torch.isinf(torch.log(luminance_factor).mean())!=1:
        lssim_map = lssim_map + weight_luminance * torch.log(luminance_factor)
    
    if torch.isnan(torch.log(contrast_factor).mean())!=1 and  torch.isinf(torch.log(contrast_factor).mean())!=1:
        lssim_map = lssim_map + weight_contrast * torch.log(contrast_factor)
        
    if torch.isnan(torch.log(structure_factor).mean())!=1 and  torch.isinf(torch.log(structure_factor).mean())!=1:
        lssim_map = lssim_map + weight_structure * torch.log(structure_factor)

#    lssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))          # original ssim

    if size_average:
        return lssim_map.mean()
    else:
        return lssim_map.mean(1).mean(1).mean(1)

class LSSIM(torch.nn.Module):
    # calculate average of LSSIM from all windows
    
    def __init__(self, window_size = 11, size_average = True):
        super(LSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _lssim(img1, img2, window, self.window_size, channel, self.size_average)

def lssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _lssim(img1, img2, window, window_size, channel, size_average)
