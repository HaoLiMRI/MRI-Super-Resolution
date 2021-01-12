"""
Reference:
[1] 2020.Learning Texture Transformer Network for Image Super-Resolution
(https://arxiv.org/abs/2006.04139)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Class SearchTransfer implements the "Hard Attention" module and "Relevance Embedding" module in figure 2 of [1].
"""
class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        '''
        lrsr_lv3 is the Q in equation (1) of [1].
        refsr_lv3 is the K in equation (2) of [1].
        ref_lv3 is the V in equation (3) of [1].
        ref_lv2 is the feature map extracted ref as well, which has double size in H and W but half size in C, compared to ref_lv3.
        ref_lv1 is the feature map extracted ref as well, which has double size in H and W but half size in C, compared to ref_lv2.
        '''
        ### search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]
        
        ### Relevance Embedding
        # Calculate the R according to equation (4) in [1].
        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, Hr*Wr, H*W]

        ### Hard Attention
        # Equation (5) in [1]. R_lv3_star_arg is the h_i in equation (5).
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W]
        # Unfold patches of V.
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)
        # Using the hard-attention map, h_i(R_lv3_star_arg), as the index to apply selection operation on unfolded patches of 
        # V according to equation (6) in [1].
        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)
        # Get T in different size corresponding to different feature map size.
        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)

        # Generate soft-attention map, S. a soft-attention map S is computed from ri,j to represent the confidence of the 
        # transferred texture features for each position in T.
        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        return S, T_lv3, T_lv2, T_lv1 # Return soft-attention map, S, and e transferred HR texture features, T, shown in the figure 2 of [1].