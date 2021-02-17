# -*- coding: utf-8 -*-
"-------------------------------------------------------------------------------------------------"
"""
2D_Visual_Transformer_MRI_SR Reconstruction. TTSRMRI.
This is the code for "Visual Transformer based Network" for MRI SR Reconstruction. The 2D_Visual_Transformer_MRI_SR is modified from the TTSR which was original proposed 
in paper [1] accepted in CVPR 2020.

** The basic idea of 2D_Visual_Transformer_MRI_SR is, take the 3 scan layers LR MRI image together in 3 channels(make it similar structure as RGB image), into TTSTMRI 
network to reconstrcut 2x (for both H and W, e.g. size of LR MRI image is 64x64 and size of reconstructed SR MRI image is 128x128) SR MRI image, with the help from 
reference MRI image in same size as SR MRI image. Such refernece MRI image can be obtained from high resolution MRI image from the similar scenarios of measurements 
with same MRI scaning module as the MRI scaning module used for LR MRI image(e.g. T1) or from the same scenarios of measurements with different MRI scaning module(e.g. T2).
"""
"""
Author: chisyliu@hotmail.com *
        hao.li@med.uni-heidelberg.de *
        
        * Both authors contribute equally
Version: 1.3.0
"""

"""
This is the current version we are working on, in 20210216
This is a demo code of 2D_Visual_Transformer_MRI_SR(TTSR_MRI). in this version we have already support following items:
    1) added a new class "SFE_Downsample" in MainNet.py which downscale the LR by 1/2, thus to support 2x upscaling in the end.
    2) added support for "L1 Charbonnier loss", "gradient map loss", "ssim map loss" and "k space loss".
    3) support to load and process normal MRI image with number 0f channel = 1.
    4) support to use channel attention(CALayer) in ResBlock of SFE(shallow feature extraction) in the MainNet.
"""

"""
Some feature or bug fixing which have already been planed/started but still not finished yet:
    1) TTSR的LR输入，HR Reference分别过小波变换得到各自的三个高频分量进LTE，同时在TTSR最外侧加一个long skip connection从而保证LR的信息中的低频部分也都被使用
"""

"""
Reference:
[1] 2020.Learning Texture Transformer Network for Image Super-Resolution
(https://arxiv.org/abs/2006.04139)
"""

"-------------------------------------------------------------------------------------------------"


from option import args
from utils import mkExpDir
from dataset import dataloader
from model import TTSR
from loss.loss import get_loss_dict
from trainer import Trainer

import os
import torch
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ### make save_dir where save all the output.
    _logger = mkExpDir(args)

    ### dataloader of training set, evaluation set and testing set.
    if (args.dataset == 'CUFED'): # For CUFED dataset, do not use dataloader to load test data if it is in test mode(args.test == True stands for test mode)
        _dataloader = dataloader.get_dataloader(args) if (not args.test) else None
    elif (args.dataset == 'MRI_SR'): # For NRU_SR dataset, we also use dataloader to load test data.
        _dataloader = dataloader.get_dataloader(args)

    ### get device and load model into device.
    device = torch.device('cpu' if args.cpu else 'cuda')
    _model = TTSR.TTSR(args).to(device)
    
    ### load the model into more than one gpu if have more than one gpu.
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = torch.nn.DataParallel(_model, list(range(args.num_gpu)))

    ### create loss function.
    _loss_all = get_loss_dict(args, _logger)

    ### create trainer.
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### training / eval / testing the model.
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.eval):
        t.load(model_path=args.model_path)
        t.evaluate()
    else:
        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)
