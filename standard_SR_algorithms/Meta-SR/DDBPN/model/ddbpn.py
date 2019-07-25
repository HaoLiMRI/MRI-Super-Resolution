"""
This is the demo code for DDBPN model proposed to generate super resolution(SR) image in the paper: 
2018. Deep Back-Projection Networks For Super-Resolution
https://arxiv.org/abs/1803.02735
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

"""
In the original paper which proposed DDBPN(2018. Deep Back-Projection Networks For Super-Resolution, mentioned as "original DBPN paper" in following), there are some important facts:
    1. DDBNP actually avoid using batch norm in the network. They point out: "Unlike the original DenseNets, we avoid dropout and batch norm, which are not suitable for SR, because they remove the range flexibility of the features [31]. Instead, we use 1 x 1 
       convolution layer as feature pooling and dimensional reduction [42, 12] before entering the projection unit"
    2. DDBNP take transpose conv for upsampling, NOT sub-pixel conv for upsampling as RDN use
    3. DDBPN follows the principle of "iterative up and downsampling", NOT follows the principle of "single upsampling" as RDN use and "predefined upsampling" as use
"""

def make_model(args, parent=False):
    return DDBPN(args)

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

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


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


class DDBPN(nn.Module):
    """
    Dense Deep Back-Projection Projection Network(DDBPN). It is as figure 5 in original DBPN paper
    """
    def __init__(self, args):
        """
        args: Set of specification of arguements, which have been defined outside ddbnp.py
              it includes several arguements, e.g.
              scale: upscale factor
              n_colors: the number of channels of input image, e.g. 3 for R,G,B
        """
        super(DDBPN, self).__init__()
        scale = args.scale[0]
        n_colors = args.n_colors

        n0 = 128
        nr = 32
        self.depth = 6 # how many "up projection units" expected to be in DDBPN(number of "down projection units" will be depth - 1)

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # ----------------------------------------------------------------------------------------------------------------------- #

        # build up "Initial Feature Extraction Module" in figure 5
        initial = [
            nn.Conv2d(n_colors, n0, 3, padding=1), # first 3x3 conv layer(light green block) in figure 5
            nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1), # second 1x1 conv layer(light green block) in figure 5
            nn.PReLU(nr)
        ]
        self.initial = nn.Sequential(*initial)

        # build up "Back-Projection stages Module" in figure 5
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

        # build up "Reconstruction Module" in figure 5
        reconstruction = [
            nn.Conv2d(self.depth * nr, n_colors, 3, padding=1) # 3x3 conv layer. the size of image has already been upscaled as expected, so just downsize the number of channels to n_colors
        ]
        self.reconstruction = nn.Sequential(*reconstruction)

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        # ----------------------------------------------------------------------------------------------------------------------- #

    def forward(self, x):
        """
        implement the flow of Dense Deep Back-Projection Projection Network(DDBPN) as figure 5 in original DBPN paper
        """
        # substraction mean first, later add mean. I don't understand the purpose
        x = self.sub_mean(x)
        # let the LR image to pass into initial feature extraction module
        x = self.initial(x)

        # ------------------------------------------------ implement the flow of "Back-Projection Stages" acoording to figure 5 ------------------------------------------------------- #
        h_list = []
        l_list = []
        for i in range(self.depth - 1): # data flow for the first "depth - 1, e.g. 5 when depth = 6" up projection units and "depth - 1, e.g. 5 when depth = 6" down projection units 
            if i == 0:
                l = x
            else:
                l = torch.cat(l_list, dim=1) # concatenate all the outputs from down projection units already passed through so far
            h_list.append(self.upmodules[i](l)) # leave the data which "concatenated all the outputs from down projection units already passed through so far" into current up projection unit in the list "upmodules"
            h = torch.cat(h_list, dim=1) # concatenate all the outputs from up projection units already passed through so far
            l_list.append(self.downmodules[i](h)) # leave the data which "concatenated all the outputs from up projection units already passed through so far" into current down projection unit in the list "downmodules"
        h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1))) # last up projection units, concatenate all the outputs from all down projection units in list "upmodules", leave into the final up projection unit
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

        out = self.reconstruction(torch.cat(h_list, dim=1)) # concatenate all the output from each up projection unit in list "upmodules" and leave them into reconstruction module 
        # substraction mean in the very early stage, now add mean. I don't understand the purpose
        out = self.add_mean(out)

        return out
