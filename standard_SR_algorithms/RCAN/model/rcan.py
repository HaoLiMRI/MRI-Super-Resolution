"""
This is the demo code for RCAN model proposed to generate super resolution(SR) image in the paper: 
2018.Image Super-Resolution Using Very Deep Residual Channel Attention Networks
https://arxiv.org/abs/1807.02758
"""

"""
In the original paper which proposed RCAN(2018. Image Super-Resolution Using Very Deep Residual Channel Attention Networks, mentioned as "original RCAN paper" in following),
The general, relationship between module and sub-module, sub-sub.module, etc, is something like:
RCAN(Deep Residual Channel Attention Network) include "RIR(Residual in Residual module) + upsampling module"; RIR consists of several RG(Residual Group);
each RG consists of several RCAB(Residual Channel Attention Block)s; each RCAB include CA(Channel Attention Layer).
"""


import torch
import torch.nn as nn

def make_model(args, parent=False):
    return RCAN(args)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    """
    Channel Attention (CA) Layer, is basical block in RCAN. One CA forms one RCAB(Residual Channel Attention Block).
    See figure 3 of original RCAN paper 
    """
    def __init__(self, channel, reduction=16):
        """
        reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        """
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # global average pooling, output size is 1 for each channel
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


## Residual Channel Attention Block (RCAB)
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
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction)) # CA
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) # conv --> ReLU --> conv --> CA
        #res = self.body(x).mul(self.res_scale)
        res += x # local skip link of RCAB
        return res


## Residual Group (RG)
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
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_rcablocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size)) # last conv after several RCABs as show in figure 2
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) # RCAB_1 --> RCAB_2 --> ... --> RCAB_(n_rcablocks) --> conv
        res += x # short skip connection in RG
        return res


## Upsampler Module, implemented by employeed of sub-pixel conv
class Upsampler(nn.Sequential):
    """
    Upsampling/Upscale module, used as last part of "SR reconstruction network model" if the network model employ the "post-upsampling mode".
    Beware the actual upsampling approach is "sub-pixel conv" (which is nn.PixelShuffle() in Pytorch) which was proposed in
    paper: "2016. Real-Time single image and video super-resolution using an efficient sub-pixel convolutional neural network"
    """
    def __init__(self, conv, scale, n_feats):
        super(Upsampler, self).__init__()
        if scale == 2 or scale == 3:
            self.upsampler = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats * scale * scale, kernel_size = 3, padding=1, stride=1),
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
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        upsampled_x = self.upsampler(x)
        return upsampled_x


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    """
    RCAN(Deep Residual Channel Attention Network) = RIR(Residual in Residual module) + Upsampler Module.
    See bottom figure in figure 2 of original RCAN paper
    """
    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args.n_resgroups # number of RGs in RIR/RCAN
        n_rcablocks = args.n_rcablocks # number of RCABs in one RG
        n_feats = args.n_feats # number of features/channels of input image
        kernel_size = 3 # conv filter size used for all conv in RCAN
        reduction = args.reduction # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        scale = args.scale # resize factor, e.g. 2, 3, 4, ... 
        act = nn.ReLU(True)
        
        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # ----------------------------------------------------------------------------------------------------------------------- #
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)] # the first conv layer in RCAN, show in figure 2 of RCAN paper

        # define body module. The RIR(Residual in Residual) which consists of n_resgroups RGs, show in figure 2 of RCAN paper
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_rcablocks=n_rcablocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module. The last stage is upsampling module and one more conv layer, show in figure 2 of RCAN paper
        modules_tail = [
            Upsampler(conv, scale, n_feats),
            conv(n_feats, args.n_colors, kernel_size)]

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        # ----------------------------------------------------------------------------------------------------------------------- #

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x) # do NOT understand why need this, may NOT be useful for us
        x = self.head(x) # input image goes through first conv layer

        res = self.body(x) # data goes through RIR(Residual in Residual)
        res += x # long skip connection of RIR

        x = self.tail(res) # data goes trhough upsampling module and one more conv layer
        x = self.add_mean(x) # do NOT understand why need this, may NOT be useful for us

        return x 

    # --------------------------------------we may NOT need this section------------------------------------------------------- #
        # don't know exactly what is doing here
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
    # ----------------------------------------------------------------------------------------------------------------------- #
