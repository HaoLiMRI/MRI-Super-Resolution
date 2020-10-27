%%

clear
% clc

load HR_test_image.mat
load LR_test_image.mat
load SR_test_image.mat

%%

hr_test=double(permute(squeeze(HR_test_image),[2,3,1]));
sr_test=double(permute(squeeze(SR_test_image),[2,3,1]));
lr_test=double(permute(squeeze(LR_test_image),[2,3,1]));


%%
dim_hr_combined_1 = 256;
dim_hr_combined_2 = 256;
dim_lr_combined_1 = 128;
dim_lr_combined_2 = 128;

dim_hr = size(hr_test,1);
dim_lr = size(lr_test,1);

vertical_segments = dim_hr_combined_1/dim_hr*2-1;
horizontal_segments = dim_hr_combined_2/dim_hr*2-1;
total_segments = vertical_segments * horizontal_segments;

scale_factor = dim_hr/dim_lr;
dim3 = size(hr_test,3)/total_segments;

SR_images = zeros(dim_hr_combined_1,dim_hr_combined_2,dim3);
HR_images = zeros(dim_hr_combined_1,dim_hr_combined_2,dim3);
LR_images = zeros(dim_lr_combined_1,dim_lr_combined_2,dim3);
if scale_factor~=1
    LR_bicubic = zeros(dim_hr_combined_1,dim_hr_combined_2,dim3);
    LR_nearest = zeros(dim_hr_combined_1,dim_hr_combined_2,dim3);
end

for k=1:dim3
    for i=1:horizontal_segments
        for j=1:vertical_segments
            if i==1
                left_edge_hr = 1;
                left_edge_lr = 1;
            else
                left_edge_hr = dim_hr/4+1;
                left_edge_lr = dim_lr/4+1;
            end
            if i==horizontal_segments
                right_edge_hr = dim_hr;
                right_edge_lr = dim_lr;
            else
                right_edge_hr = dim_hr*3/4;
                right_edge_lr = dim_lr*3/4;
            end
            if j==1
                top_edge_hr = 1;
                top_edge_lr = 1;
            else
                top_edge_hr = dim_hr/4+1;
                top_edge_lr = dim_lr/4+1;
            end
            if j==vertical_segments
                bottom_edge_hr = dim_hr;
                bottom_edge_lr = dim_lr;
            else
                bottom_edge_hr = dim_hr*3/4;
                bottom_edge_lr = dim_lr*3/4;
            end
            SR_images((i-1)*dim_hr/2+left_edge_hr:(i-1)*dim_hr/2+right_edge_hr,(j-1)*dim_hr/2+top_edge_hr:(j-1)*dim_hr/2+bottom_edge_hr,k)=sr_test(left_edge_hr:right_edge_hr,top_edge_hr:bottom_edge_hr,(i-1)*vertical_segments+j+(k-1)*(vertical_segments * horizontal_segments));
            HR_images((i-1)*dim_hr/2+left_edge_hr:(i-1)*dim_hr/2+right_edge_hr,(j-1)*dim_hr/2+top_edge_hr:(j-1)*dim_hr/2+bottom_edge_hr,k)=hr_test(left_edge_hr:right_edge_hr,top_edge_hr:bottom_edge_hr,(i-1)*vertical_segments+j+(k-1)*(vertical_segments * horizontal_segments));
            LR_images((i-1)*dim_lr/2+left_edge_lr:(i-1)*dim_lr/2+right_edge_lr,(j-1)*dim_lr/2+top_edge_lr:(j-1)*dim_lr/2+bottom_edge_lr,k)=lr_test(left_edge_lr:right_edge_lr,top_edge_lr:bottom_edge_lr,(i-1)*vertical_segments+j+(k-1)*(vertical_segments * horizontal_segments));
        end
    end
    if scale_factor~=1
        LR_bicubic(:,:,k)=imresize(LR_images(:,:,k),scale_factor,'bicubic');
        LR_nearest(:,:,k)=imresize(LR_images(:,:,k),scale_factor,'nearest');
    end
end

%%
slice_te = 280;
WT = 0.6;

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
    subplot(row_num,fig_num,5); imagesc(LR_nearest(:,:,slice_te),[0 WT/2]);colormap(gray);axis image;title('LR nearest');xlabel({strcat(num2str(ssimval_LRnnte),32,'/',32,num2str(peaksnr_LRnnte))});
    subplot(row_num,fig_num,6); imagesc(abs(HR_images(:,:,slice_te)-SR_images(:,:,slice_te)),[0 WT/2]);colormap(gray);axis image;title('SR HR Difference');
end

figure;set(gcf,'Position',get(0,'ScreenSize'));
subplot(row_num,fig_num,1); imagesc(abs(fftshift(fftn(HR_images(:,:,slice_te)))),[0 8]);colormap(gray);axis image;title('HRGT test'); xlabel('SSIM / PSNR');
subplot(row_num,fig_num,2); imagesc(abs(fftshift(fftn(SR_images(:,:,slice_te)))),[0 8]);colormap(gray);axis image;title('SR test');xlabel({strcat(num2str(ssimval_SRte),32,'/',32,num2str(peaksnr_SRte))});
subplot(row_num,fig_num,3); imagesc(abs(fftshift(fftn(LR_images(:,:,slice_te)))),[0 8]);colormap(gray);axis image;title('LR test');
if scale_factor ==1
    xlabel({strcat(num2str(ssimval_LRte),32,'/',32,num2str(peaksnr_LRte))});
else
    subplot(row_num,fig_num,4); imagesc(abs(fftshift(fftn(LR_bicubic(:,:,slice_te)))),[0 8]);colormap(gray);axis image;title('LR bicubic');xlabel({strcat(num2str(ssimval_LRbite),32,'/',32,num2str(peaksnr_LRbite))});
    subplot(row_num,fig_num,5); imagesc(abs(fftshift(fftn(LR_nearest(:,:,slice_te)))),[0 8]);colormap(gray);axis image;title('LR nearest');xlabel({strcat(num2str(ssimval_LRnnte),32,'/',32,num2str(peaksnr_LRnnte))});
    subplot(row_num,fig_num,6); imagesc(abs(abs(fftshift(fftn(HR_images(:,:,slice_te))))-abs(fftshift(fftn(SR_images(:,:,slice_te))))),[0 8]);colormap(gray);axis image;title('SR HR Difference');
end

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

%%
ssim_sr_all = zeros(1,size(HR_images,3));
psnr_sr_all = zeros(1,size(HR_images,3));
ssim_lr_bicubic_all = zeros(1,size(HR_images,3));
psnr_lr_bicubic_all = zeros(1,size(HR_images,3));
ssim_lr_nearest_all = zeros(1,size(HR_images,3));
psnr_lr_nearest_all = zeros(1,size(HR_images,3));
for i = 1:size(HR_images,3)
    [ssim_sr_all(i), ssimmap_sr] = ssim(SR_images(:,:,i),HR_images(:,:,i), 'DynamicRange',1);
    psnr_sr_all(i) = psnr(SR_images(:,:,i),HR_images(:,:,i));
    [ssim_lr_bicubic_all(i), ssimmap_lr] = ssim(LR_bicubic(:,:,i),HR_images(:,:,i), 'DynamicRange',1);
    psnr_lr_bicubic_all(i) = psnr(LR_bicubic(:,:,i),HR_images(:,:,i));
    [ssim_lr_nearest_all(i), ssimmap_lr] = ssim(LR_nearest(:,:,i),HR_images(:,:,i), 'DynamicRange',1);
    psnr_lr_nearest_all(i) = psnr(LR_nearest(:,:,i),HR_images(:,:,i));
end
[mean(ssim_sr_all) std(ssim_sr_all) mean(psnr_sr_all) std(psnr_sr_all)]
[mean(ssim_lr_bicubic_all) std(ssim_lr_bicubic_all) mean(psnr_lr_bicubic_all) std(psnr_lr_bicubic_all)]
[mean(ssim_lr_nearest_all) std(ssim_lr_nearest_all) mean(psnr_lr_nearest_all) std(psnr_lr_nearest_all)]


%%
% load log of loss
% clear;
fileID = fopen("20201021_loss.txt");
Loss_org = textscan(fileID,'%s %s %s %s %s %s %s %f32');
fclose(fileID);
whos Loss_org
loss = Loss_org{8};
loss = permute(loss, [2 1]);

for i=1:12:size(loss,2)-11
    feature_training(ceil(i/12)) = loss(i);
    pixel_training(ceil(i/12)) = loss(i+1);
    kspace_training(ceil(i/12)) = loss(i+2);
    ssim_training(ceil(i/12)) = loss(i+3);
    gradient_training(ceil(i/12)) = loss(i+4);
    loss_training(ceil(i/12)) = loss(i+5);
    feature_validation(ceil(i/12)) = loss(i+6);
    pixel_validation(ceil(i/12)) = loss(i+7);
    kspace_validation(ceil(i/12)) = loss(i+8);
    ssim_validation(ceil(i/12)) = loss(i+9);
    gradient_validation(ceil(i/12)) = loss(i+10);
    loss_validation(ceil(i/12)) = loss(i+11);
end

save loss.mat feature_training gradient_training kspace_training loss_training pixel_training ssim_training feature_validation gradient_validation kspace_validation loss_validation pixel_validation ssim_validation

figure;set(gcf,'Position',get(0,'ScreenSize'));
subplot(2,3,1);plot(feature_training,'b');hold;plot(feature_validation,'r');title('feature\_map\_loss');xlabel('EPOCH');ylabel('Loss');
subplot(2,3,2);plot(pixel_training,'b');hold;plot(pixel_validation,'r');title('pixel\_wise\_loss');xlabel('EPOCH');ylabel('Loss');
subplot(2,3,3);plot(kspace_training,'b');hold;plot(kspace_validation,'r');title('k\_space\_loss');xlabel('EPOCH');ylabel('Loss');
subplot(2,3,4);plot(ssim_training,'b');hold;plot(ssim_validation,'r');title('ssim\_loss');xlabel('EPOCH');ylabel('Loss');
subplot(2,3,5);plot(gradient_training,'b');hold;plot(gradient_validation,'r');title('gradient\_loss');xlabel('EPOCH');ylabel('Loss');
subplot(2,3,6);plot(loss_training,'b');hold;plot(loss_validation,'r');title('total loss');xlabel('EPOCH');ylabel('Loss');


%%
load loss.mat

figure;set(gcf,'Position',get(0,'ScreenSize'));
subplot(2,3,1);plot(feature_training(21:1000),'b');hold;plot(feature_validation(21:1000),'r');title('feature\_map\_loss');xlabel('EPOCH');ylabel('Loss');
subplot(2,3,2);plot(pixel_training(21:1000),'b');hold;plot(pixel_validation(21:1000),'r');title('pixel\_wise\_loss');xlabel('EPOCH');ylabel('Loss');
subplot(2,3,3);plot(kspace_training(21:1000),'b');hold;plot(kspace_validation(21:1000),'r');title('k\_space\_loss');xlabel('EPOCH');ylabel('Loss');
subplot(2,3,4);plot(ssim_training(21:1000),'b');hold;plot(ssim_validation(21:1000),'r');title('ssim\_loss');xlabel('EPOCH');ylabel('Loss');
subplot(2,3,5);plot(gradient_training(21:1000),'b');hold;plot(gradient_validation(21:1000),'r');title('gradient\_loss');xlabel('EPOCH');ylabel('Loss');
subplot(2,3,6);plot(loss_training(21:1000),'b');hold;plot(loss_validation(21:1000),'r');title('total loss');xlabel('EPOCH');ylabel('Loss');