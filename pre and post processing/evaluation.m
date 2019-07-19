% Super resolution image quality evaluation using SSIM and PSNR
% SSIM is expected to be as close to 1 as possible
% PSNR is expected to be as large as possible
% SSIM and PSNE or SR and LR are compared
% 
% Change slice_tr or slice_te in range of 1 to 16 for different images
% WT is the upper range of window width
%

clear
clc

load HR_training_image.mat
load LR_training_image.mat
load SR_training_image.mat

load HR_test_image.mat
load LR_test_image.mat
load SR_test_image.mat

hr_training=permute(squeeze(HR_training_image),[2,3,1]);
sr_training=permute(squeeze(SR_training_image),[2,3,1]);
lr_training=permute(squeeze(LR_training_image),[2,3,1]);

hr_test=single(permute(squeeze(HR_test_image),[2,3,1]));
sr_test=permute(squeeze(SR_test_image),[2,3,1]);
lr_test=single(permute(squeeze(LR_test_image),[2,3,1]));

%%
slice_tr = 16;
WT = 0.4;

[ssimval_srtr, ssimmap_srtr] = ssim(sr_training(:,:,slice_tr),hr_training(:,:,slice_tr), 'DynamicRange',1);
[ssimval_lrtr, ssimmap_lrtr] = ssim(lr_training(:,:,slice_tr),hr_training(:,:,slice_tr), 'DynamicRange',1);
peaksnr_srtr = psnr(sr_training(:,:,slice_tr),hr_training(:,:,slice_tr));
peaksnr_lrtr = psnr(lr_training(:,:,slice_tr),hr_training(:,:,slice_tr));

figure;set(gcf,'Position',get(0,'ScreenSize'));
subplot(1,3,1); imagesc(hr_training(:,:,slice_tr),[0 WT]);colormap(gray);axis image;title('HRGT training');
subplot(1,3,2); imagesc(sr_training(:,:,slice_tr),[0 WT]);colormap(gray);axis image;title('SR training');xlabel({strcat('SSIM:',32,num2str(ssimval_srtr)),strcat('PSNR:',32,num2str(peaksnr_srtr))});
subplot(1,3,3); imagesc(lr_training(:,:,slice_tr),[0 WT]);colormap(gray);axis image;title('LR training');xlabel({strcat('SSIM:',32,num2str(ssimval_lrtr)),strcat('PSNR:',32,num2str(peaksnr_lrtr))});
  
% fprintf('The SSIM value of training SR is %0.4f, peaksnr is %0.4f.\n',ssimval_srtr,peaksnr_srtr);
% fprintf('The SSIM value of training LR is %0.4f, peaksnr is %0.4f.\n',ssimval_lrtr,peaksnr_lrtr);

%%
slice_te = 16;
WT = 0.4;

[ssimval_srte, ssimmap_srte] = ssim(sr_test(:,:,slice_te),hr_test(:,:,slice_te), 'DynamicRange',1);
[ssimval_lrte, ssimmap_lrte] = ssim(lr_test(:,:,slice_te),hr_test(:,:,slice_te), 'DynamicRange',1);
peaksnr_srte = psnr(sr_test(:,:,slice_te),hr_test(:,:,slice_te));
peaksnr_lrte = psnr(lr_test(:,:,slice_te),hr_test(:,:,slice_te));


figure;set(gcf,'Position',get(0,'ScreenSize'));
subplot(1,3,1); imagesc(hr_test(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('HRGT test'); 
subplot(1,3,2); imagesc(sr_test(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('SR test');xlabel({strcat('SSIM:',32,num2str(ssimval_srte)),strcat('PSNR:',32,num2str(peaksnr_srte))});
subplot(1,3,3); imagesc(lr_test(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('LR test');xlabel({strcat('SSIM:',32,num2str(ssimval_lrte)),strcat('PSNR:',32,num2str(peaksnr_lrte))});

% fprintf('The SSIM value of test SR is %0.4f, peaksnr is %0.4f.\n',ssimval_srte,peaksnr_srte);
% fprintf('The SSIM value of test LR is %0.4f, peaksnr is %0.4f.\n',ssimval_lrte,peaksnr_lrte);