from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_ssim_l1 import SSIM 

"calculate gradient map for any input MRI image"
def calculate_gradient_map(n_colors, mri_img):
    # sobel operator
    vertical_edge_mask = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0)
    horizontal_edge_mask = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0)

    if n_colors == 3:   # For MRI image with number of channel = 3, we just stack 3 MRI image with number of channel = 1 together. 
        vertical_edge_mask = torch.cat((vertical_edge_mask, vertical_edge_mask, vertical_edge_mask),0)
        horizontal_edge_mask = torch.cat((horizontal_edge_mask, horizontal_edge_mask, horizontal_edge_mask),0)
    elif not(n_colors == 1):
        raise SystemExit('Error: n_colors must be 1 or 3!')

    vertical_edge_mask = vertical_edge_mask.float().unsqueeze(0).cuda()
    horizontal_edge_mask = horizontal_edge_mask.float().unsqueeze(0).cuda()

    gradient_vertical_map = F.conv2d(mri_img, vertical_edge_mask, padding = 1, stride = 1, groups = 1)
    gradient_horizontal_map = F.conv2d(mri_img, horizontal_edge_mask, padding = 1, stride = 1, groups = 1)

    gradient_map = abs(gradient_vertical_map) + abs(gradient_horizontal_map)

    return gradient_map


"Calculate the fft to fetch k space result and ifft to go back to image domain"
def FFT_K_SPACE(img):
    k_space_result = torch.rfft(img, signal_ndim = 2, onesided = False)
    return k_space_result


"""
L1 Charbonnier Loss. See more information regarding L1 Charboniier Loss from paper: 2018.Fast and Accurate Image Super-Resolution with 
Deep Laplacian Pyramid Networks. L1 Charboniier Loss in theory can be used to replace the (smooth) L1 loss, to provide reconstructed 
image with less over-smoothing issues and problem.
"""
class L1_Charbonnier_Loss(nn.Module):
    def __init__(self):
        super(L1_Charbonnier_Loss, self).__init__()
        self.eps = 1e-4

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
#        print(loss.size())
        return loss


class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        elif (type == 'Charbonnier'):
            self.loss = L1_Charbonnier_Loss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, sr, hr):
        return self.loss(sr, hr)


class Gradient_Map_Loss(nn.Module):
    def __init__(self, n_colors):
        super(Gradient_Map_Loss, self).__init__()
        self.n_colors = n_colors
        self.loss = nn.L1Loss()

    def forward(self, sr, hr):
        loss = self.loss(calculate_gradient_map(self.n_colors, sr), calculate_gradient_map(self.n_colors, hr))
        return loss


class K_Space_Loss(nn.Module):
    def __init__(self):
        super(K_Space_Loss, self).__init__()

        self.loss = nn.MSELoss()

    def forward(self, sr, hr):
        sr_k_space = FFT_K_SPACE(sr)
        hr_k_space = FFT_K_SPACE(hr)
        loss = self.loss(sr_k_space[:,:,:,:,0], hr_k_space[:,:,:,:,0]) + self.loss(sr_k_space[:,:,:,:,1], hr_k_space[:,:,:,:,1])
        return loss


class SSIM_Loss(nn.Module):
    def __init__(self, luminance_weight = 2, contrast_weight = 2, structure_weight = 4):
        super(SSIM_Loss, self).__init__()
        self.luminance_weight = luminance_weight
        self.contrast_weight = contrast_weight
        self.structure_weight = structure_weight
        self.ssim = SSIM(luminance_weight=self.luminance_weight, contrast_weight=self.contrast_weight, structure_weight=self.structure_weight)
        self.loss = nn.L1Loss()

    def forward(self, sr, hr):
        sr_ssim_weighted, _ = self.ssim(sr, hr)
        """ print('sr_ssim_weighted:', sr_ssim_weighted) """
        hr_ssim_weighted, _ = self.ssim(hr, hr)
        """ print('hr_ssim_weighted:', hr_ssim_weighted) """
        loss = self.loss(sr_ssim_weighted, hr_ssim_weighted)
        """ print('ssim_loss:', loss) """
        """
        sr_lu, sr_co, sr_st = self.ssim(sr, hr)
        print('sr_luminance: ', sr_lu.mean())
        print('sr_contrast: ', sr_co.mean())
        print('sr_structure: ', sr_st.mean())
        hr_lu, hr_co, hr_st = self.ssim(hr, hr)
        print('hr_luminance: ', hr_lu.mean())
        print('hr_contrast: ', hr_co.mean())
        print('hr_structure: ', hr_st.mean())
        loss = self.loss(sr_lu*sr_co*sr_st, hr_lu*hr_co*hr_st)
        """
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def forward(self, sr_relu5_1, hr_relu5_1):
        loss = F.mse_loss(sr_relu5_1, hr_relu5_1)
        return loss


class TPerceptualLoss(nn.Module):
    def __init__(self, use_S=True, type='l2'):
        super(TPerceptualLoss, self).__init__()
        self.use_S = use_S
        self.type = type

    def gram_matrix(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, h*w)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def forward(self, map_lv3, map_lv2, map_lv1, S, T_lv3, T_lv2, T_lv1):
        ### S.size(): [N, 1, h, w]
        if (self.use_S):
            S_lv3 = torch.sigmoid(S)
            S_lv2 = torch.sigmoid(F.interpolate(S, size=(S.size(-2)*2, S.size(-1)*2), mode='bicubic'))
            S_lv1 = torch.sigmoid(F.interpolate(S, size=(S.size(-2)*4, S.size(-1)*4), mode='bicubic'))
        else:
            S_lv3, S_lv2, S_lv1 = 1., 1., 1.

        if (self.type == 'l1'):
            loss_texture  = F.l1_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.l1_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            loss_texture += F.l1_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.
        elif (self.type == 'l2'):
            loss_texture  = F.mse_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.mse_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            loss_texture += F.mse_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.
        
        return loss_texture


class AdversarialLoss(nn.Module):
    def __init__(self, logger, use_cpu=False, num_gpu=1, gan_type='WGAN_GP', gan_k=1, 
        lr_dis=1e-4, train_crop_size=32, n_colors=1):

        super(AdversarialLoss, self).__init__()
        self.logger = logger
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = discriminator.Discriminator(train_crop_size*4, n_colors).to(self.device)
        if (num_gpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(num_gpu)))
        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)

        # if (D_path):
        #     self.logger.info('load_D_path: ' + D_path)
        #     D_state_dict = torch.load(D_path)
        #     self.discriminator.load_state_dict(D_state_dict['D'])
        #     self.optimizer.load_state_dict(D_state_dict['D_optim'])
            
    def forward(self, fake, real):
        fake_detach = fake.detach()

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if (self.gan_type.find('WGAN') >= 0):
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif (self.gan_type == 'GAN'):
                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.

            # Discriminator update
            loss_d.backward()
            self.optimizer.step()

        d_fake_for_g = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g.mean()
        elif (self.gan_type == 'GAN'):
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)

        # Generator loss
        return loss_g
  
    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict


def get_loss_dict(args, logger):
    loss = {}
    if (abs(args.rec_w - 0) <= 1e-8):
        raise SystemExit('NotImplementError: ReconstructionLoss must exist!')
    else:
        loss['rec_loss'] = ReconstructionLoss(type='Charbonnier')
    if (abs(args.grad_w - 0) > 1e-8):
        loss['grad_loss'] = Gradient_Map_Loss(n_colors = args.n_colors)
    if (abs(args.kspace_w - 0) > 1e-8):
        loss['k_space_loss'] = K_Space_Loss()
    if (abs(args.ssim_w - 0) > 1e-8):
        loss['ssim_loss'] = SSIM_Loss()
    if (abs(args.per_w - 0) > 1e-8):
        loss['per_loss'] = PerceptualLoss()
    if (abs(args.tpl_w - 0) > 1e-8):
        loss['tpl_loss'] = TPerceptualLoss(use_S=args.tpl_use_S, type=args.tpl_type)
    if (abs(args.adv_w - 0) > 1e-8):
        loss['adv_loss'] = AdversarialLoss(logger=logger, use_cpu=args.cpu, num_gpu=args.num_gpu, 
            gan_type=args.GAN_type, gan_k=args.GAN_k, lr_dis=args.lr_rate_dis,
            train_crop_size=args.train_crop_size, n_colors=args.n_colors)
    return loss
