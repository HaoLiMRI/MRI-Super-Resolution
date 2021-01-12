"""
Reference:
[1] 2020.Learning Texture Transformer Network for Image Super-Resolution
(https://arxiv.org/abs/2006.04139)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import MeanShift


"""
Class LTE implements the "Learnable Texture Extractor" module in figure 2 of [1]. Basically LTE just extract feature maps 
in different sizes from different layers of pre-trained VGG feature extractor.
"""
class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LTE, self).__init__()
        
        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad
        
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        """ x = self.sub_mean(x) """    # Not applicable for MRI SR?
        x = self.slice1(x)      # Go through the first 2 layers of vgg feature extractor.
        x_lv1 = x
        x = self.slice2(x)      # Go through the 3th to 7th layers of vgg feature extractor.
        x_lv2 = x
        x = self.slice3(x)      # Go through the 8th to 12th layers of vgg feature extractor.
        x_lv3 = x
        return x_lv1, x_lv2, x_lv3