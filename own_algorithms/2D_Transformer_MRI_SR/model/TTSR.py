"""
Reference:
[1] 2020.Learning Texture Transformer Network for Image Super-Resolution
(https://arxiv.org/abs/2006.04139)
"""

from model import MainNet, LTE, SearchTransfer

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Class TTSR implements the entire framework of TTSR network which is shown in figure 2 of [1].
"""
class TTSR(nn.Module):
    def __init__(self, args):
        super(TTSR, self).__init__()
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )
        self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, n_colors=args.n_colors,
            res_scale=args.res_scale, scale_factor=args.scale_factor)
        self.LTE = LTE.LTE(requires_grad=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.n_colors = args.n_colors

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            if self.args.dataset == 'CUFED':
                sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            elif self.args.dataset == 'MRI_SR':
                sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy(sr)
            return sr_lv1, sr_lv2, sr_lv3

        if self.n_colors == 1:
            lrsr = torch.cat((lrsr, lrsr, lrsr), 1)
            ref = torch.cat((ref, ref, ref), 1)
            refsr = torch.cat((refsr, refsr, refsr), 1)
        elif not(self.n_colors == 3):
            raise SystemExit('Error: n_colors must be 1 or 3!')

        if self.args.dataset == 'CUFED':
            _, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)   # lrsr_lv3 is the Q in equation (1) of [1].
            _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)  # refsr_lv3 is the K in equation (2) of [1].
            ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)  # ref_lv3 is the V in equation (3) of [1].
        elif self.args.dataset == 'MRI_SR':
            _, _, lrsr_lv3  = self.LTE(lrsr.detach())   # lrsr_lv3 is the Q in equation (1) of [1].
            _, _, refsr_lv3 = self.LTE(refsr.detach())  # refsr_lv3 is the K in equation (2) of [1].
            ref_lv1, ref_lv2, ref_lv3 = self.LTE(ref.detach())  # ref_lv3 is the V in equation (3) of [1].

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)

        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        return sr, S, T_lv3, T_lv2, T_lv1
