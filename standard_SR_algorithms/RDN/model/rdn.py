"""
This is the demo code for RDN model proposed to generate super resolution(SR) image in the paper: 
2018. Residual Dense Network for Image Super-Resolution
https://arxiv.org/abs/1802.08797
"""


import torch
import torch.nn as nn

"""
In the original paper which proposed RDN(2018. Residual Dense Network for Image Super-Resolution, mentioned as "original RDN paper" in following), there are terminologies:
the number of RDB(residual dense block) in RDN denotes as D for short,
the number of Conv layers per RDB denotes as C for short,
the growth rate denotes as G for short.
"""


def make_model(args, parent=False):
    return RDN(args)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class RDB_Conv(nn.Module):
    """
    RDB_Conv: basical block in RDB. there are several RDB_Convs in RDB
    """
    def __init__(self, inChannels, growRate, kSize=3):
        """
        growRate: It is actually how many channels of feature map output from each conv layer in dense block(now it is resdiual dense block)
        kSize: kernel conv filter size
        """
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1), # make sure the "H and W" of output feature map is same as "H and W" of input, by setting padding=(kSize-1)//2
            # Beware BN is removed in this network model
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        # concatenate input and output to/from one conv + relu, to generate final output for each RDB_conv in one RDB.
        # It actually will output the feature map with "number of channels equal to inChannels+growRate"
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    """
    RDB: Residual Dense Block. e.g. figure 3 of original RDN paper. basical block in RDN. there are several RDBs in RDN
    Beware the number of channels for feature maps inside one RDB, before passing into "local feature fusion", will become larger(G0 + C*G). After passing the whole RDB the number of channels of output will still be G0
    """
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        """
        nConvLayers, C: It denotes how many RDB_Convs will be used in RDB
        growRate0, G0: It denotes how many "number of channels" of input data of this RDB
        growRate, G: It denotes how many "number of channels" we expect to grow for output data of each RDB_Conv in this RDB. 
                     So after passing C RDB_Convs, the number of channels of "output which is before the local feature fusion(LFF)" will be G0 + G*C
        """
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G)) # pass into each RDB_Conv, every time get G more channels in "number of channels" of feature map outputed
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion(LFF). It is actual the "Concat + 1x1 conv" which is shown in figure 3 of original RDN paper 
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1) # Beware after this LFF the number of feature maps(channels) become G0 again

    def forward(self, x):
        return self.LFF(self.convs(x)) + x # output = output + residual link, for local residual learning


class RDN(nn.Module):
    """
    RDB: Residual Dense Network. e.g. figure 2 of original RDN paper
    """
    def __init__(self, args):
        """
        args: Batch of specification of arguements, which have been defined outside rdn_sr.py
              it includes several arguements, e.g.
              r: resize factor, e.g. 2, 3, 4, ... 
              G0: 
              kSize: default kernel conv filter size
        """
        super(RDN, self).__init__()
        r = args.scale[0] # resize factor, e.g. 2, 3, 4, ... 
        G0 = args.G0 # 
        kSize = args.RDNkSize #

        # D stands for the number of RDBs in RDN,
        # C stands for the number of Conv layers per RDB,
        # G stands for the growth rate.
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        rgb_mean = (0.4488, 0.4371, 0.4040) # what are those? mean value for what
        rgb_std = (1.0, 1.0, 1.0) # what are those? standard deviation value for what
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        # ----------------------------------------------------------------------------------------------------------------------- #

        # Shallow feature extraction(SFE) net, the two conv layers in the begin of RDN, shown in figure 2 of original RDN paper
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks in sequential
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion(GFF). It is actual the "Concat + 1x1 conv + conv" which is shown in figure 2 of original RDN paper 
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1), # concat + 1x1 conv
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1) # conv
        ])

        # Up-sampling net module
        # Upsampling/Upscale module, used as second part of "SR reconstruction network model".
        # Beware the actual upsampling approach is "sub-pixel conv" (which is nn.PixelShuffle() in Pytorch) which was proposed in
        # paper: "2016. Real-Time single image and video super-resolution using an efficient sub-pixel convolutional neural network"
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1 # output = output + residual link, for global residual learning

        x = self.UPNet(x)
        x = self.add_mean(x) # do NOT understand why need this, may NOT be useful for us
        return x
