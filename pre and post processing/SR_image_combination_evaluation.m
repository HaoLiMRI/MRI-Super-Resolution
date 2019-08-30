%%

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

hr_test=double(permute(squeeze(HR_test_image),[2,3,1]));
sr_test=double(permute(squeeze(SR_test_image),[2,3,1]));
lr_test=double(permute(squeeze(LR_test_image),[2,3,1]));


dim1_hr = 256;
dim2_hr = 256;
dim3 = size(sr_test,3)/16;

dim1_lr = 64;
dim2_lr = 64;

scale_factor = dim1_hr/dim1_lr;

SR_images = zeros(dim1_hr,dim2_hr,dim3);
HR_images = zeros(dim1_hr,dim2_hr,dim3);
LR_images = zeros(dim1_lr,dim2_lr,dim3);
if scale_factor~=1
    LR_bicubic = zeros(dim1_hr,dim2_hr,dim3);
    LR_nearest = zeros(dim1_hr,dim2_hr,dim3);
end

for k=1:dim3
    for i=1:4
        for j=1:4
            SR_images((i-1)*64+1:(i-1)*64+64,(j-1)*64+1:(j-1)*64+64,k)=sr_test(:,:,(i-1)*4+j+(k-1)*16);
            HR_images((i-1)*64+1:(i-1)*64+64,(j-1)*64+1:(j-1)*64+64,k)=hr_test(:,:,(i-1)*4+j+(k-1)*16);
            LR_images((i-1)*64/scale_factor+1:(i-1)*64/scale_factor+64/scale_factor,(j-1)*64/scale_factor+1:(j-1)*64/scale_factor+64/scale_factor,k)=lr_test(:,:,(i-1)*4+j+(k-1)*16);
        end
    end
    if scale_factor~=1
        LR_bicubic(:,:,k)=imresize(LR_images(:,:,k),scale_factor,'bicubic');
        LR_nearest(:,:,k)=imresize(LR_images(:,:,k),scale_factor,'nearest');
    end
end

%%
if scale_factor~=1
    lrbi_test = zeros(size(hr_test));
    lrnn_test = lrbi_test;
    for i=1:size(hr_test,3)
        lrbi_test(:,:,i)=imresize(lr_test(:,:,i),scale_factor,'bicubic');
        lrnn_test(:,:,i)=imresize(lr_test(:,:,i),scale_factor,'nearest'); 
    end
end



%%
slice_te = 220;
WT = 0.6;
% WT_ssim = 0.5;

[ssimval_SRte, ssimmap_SRte] = ssim(SR_images(:,:,slice_te),HR_images(:,:,slice_te), 'DynamicRange',1);
peaksnr_SRte = psnr(SR_images(:,:,slice_te),HR_images(:,:,slice_te));
if scale_factor ==1
    [ssimval_LRte, ssimmap_LRte] = ssim(LR_images(:,:,slice_te),HR_images(:,:,slice_te), 'DynamicRange',1);
    peaksnr_LRte = psnr(LR_images(:,:,slice_te),HR_images(:,:,slice_te));
    fig_num = 3;
    row_num = 1;
else
    [ssimval_LRbite, ssimmap_LRbite] = ssim(LR_bicubic(:,:,slice_te),HR_images(:,:,slice_te), 'DynamicRange',1);
    peaksnr_LRbite = psnr(LR_bicubic(:,:,slice_te),HR_images(:,:,slice_te));
    [ssimval_LRnnte, ssimmap_LRnnte] = ssim(LR_nearest(:,:,slice_te),HR_images(:,:,slice_te), 'DynamicRange',1);
    peaksnr_LRnnte = psnr(LR_nearest(:,:,slice_te),HR_images(:,:,slice_te));
    fig_num = 3;
    row_num = 2;
end


figure;set(gcf,'Position',get(0,'ScreenSize'));
subplot(row_num,fig_num,1); imagesc(HR_images(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('HRGT test'); xlabel('SSIM / PSNR');
subplot(row_num,fig_num,2); imagesc(SR_images(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('SR test');xlabel({strcat(num2str(ssimval_SRte),32,'/',32,num2str(peaksnr_SRte))});
subplot(row_num,fig_num,3); imagesc(LR_images(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('LR test');
if scale_factor ==1
    xlabel({strcat(num2str(ssimval_LRte),32,'/',32,num2str(peaksnr_LRte))});
else
    subplot(row_num,fig_num,4); imagesc(LR_bicubic(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('LR bicubic');xlabel({strcat(num2str(ssimval_LRbite),32,'/',32,num2str(peaksnr_LRbite))});
    subplot(row_num,fig_num,5); imagesc(LR_nearest(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('LR nearest');xlabel({strcat(num2str(ssimval_LRnnte),32,'/',32,num2str(peaksnr_LRnnte))});
end

%%
% figure;set(gcf,'Position',get(0,'ScreenSize'));
% subplot(1,2,1); imagesc(1-ssimmap_srte(:,:),[0 WT_ssim]);colormap(gray);axis image;title('SR test'); xlabel({strcat('SSIM:',32,num2str(ssimval_SRte)),strcat('PSNR:',32,num2str(peaksnr_SRte))});
% subplot(1,2,2); imagesc(1-ssimmap_LRte(:,:),[0 WT_ssim]);colormap(gray);axis image;title('LR test'); xlabel({strcat('SSIM:',32,num2str(ssimval_LRte)),strcat('PSNR:',32,num2str(peaksnr_LRte))});



%%
slice_te = 3367;
WT = 0.6;

[ssimval_srte, ssimmap_srte] = ssim(sr_test(:,:,slice_te),hr_test(:,:,slice_te), 'DynamicRange',1);
peaksnr_srte = psnr(sr_test(:,:,slice_te),hr_test(:,:,slice_te));
if scale_factor ==1
    [ssimval_lrte, ssimmap_lrte] = ssim(lr_test(:,:,slice_te),hr_test(:,:,slice_te), 'DynamicRange',1);
    peaksnr_lrte = psnr(lr_test(:,:,slice_te),hr_test(:,:,slice_te));
    fig_num = 3;
    row_num = 1;
else
    [ssimval_lrbite, ssimmap_lrbite] = ssim(lrbi_test(:,:,slice_te),hr_test(:,:,slice_te), 'DynamicRange',1);
    peaksnr_lrbite = psnr(lrbi_test(:,:,slice_te),hr_test(:,:,slice_te));
    [ssimval_lrnnte, ssimmap_lrnnte] = ssim(lrnn_test(:,:,slice_te),hr_test(:,:,slice_te), 'DynamicRange',1);
    peaksnr_lrnnte = psnr(lrnn_test(:,:,slice_te),hr_test(:,:,slice_te));
    fig_num = 3;
    row_num = 2;
end


figure;set(gcf,'Position',get(0,'ScreenSize'));
subplot(row_num,fig_num,1); imagesc(hr_test(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('HRGT test'); xlabel('SSIM / PSNR');
subplot(row_num,fig_num,2); imagesc(sr_test(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('SR test');xlabel({strcat(num2str(ssimval_srte),32,'/',32,num2str(peaksnr_srte))});
subplot(row_num,fig_num,3); imagesc(lr_test(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('LR test');
if scale_factor ==1
    xlabel({strcat(num2str(ssimval_lrte),32,'/',32,num2str(peaksnr_lrte))});
else
    subplot(row_num,fig_num,4); imagesc(lrbi_test(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('LR bicubic');xlabel({strcat(num2str(ssimval_lrbite),32,'/',32,num2str(peaksnr_lrbite))});
    subplot(row_num,fig_num,5); imagesc(lrnn_test(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('LR nearest');xlabel({strcat(num2str(ssimval_lrnnte),32,'/',32,num2str(peaksnr_lrnnte))});
end