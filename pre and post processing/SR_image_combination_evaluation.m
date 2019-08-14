%%
dim1 = 256;
dim2 = 256;
dim3 = size(sr_test,3)/16;
SR_images = zeros(dim1,dim2,dim3);
HR_images = zeros(dim1,dim2,dim3);
LR_images = zeros(dim1,dim2,dim3);
for k=1:dim3
    for i=1:4
        for j=1:4
            SR_images((i-1)*64+1:(i-1)*64+64,(j-1)*64+1:(j-1)*64+64,k)=sr_test(:,:,(i-1)*4+j+(k-1)*16);
            HR_images((i-1)*64+1:(i-1)*64+64,(j-1)*64+1:(j-1)*64+64,k)=hr_test(:,:,(i-1)*4+j+(k-1)*16);
            LR_images((i-1)*64+1:(i-1)*64+64,(j-1)*64+1:(j-1)*64+64,k)=lr_test(:,:,(i-1)*4+j+(k-1)*16);
        end
    end
end


%%
slice_te = 256;
WT = 0.4;

[ssimval_srte, ssimmap_srte] = ssim(SR_images(:,:,slice_te),HR_images(:,:,slice_te), 'DynamicRange',1);
[ssimval_lrte, ssimmap_lrte] = ssim(LR_images(:,:,slice_te),HR_images(:,:,slice_te), 'DynamicRange',1);
peaksnr_srte = psnr(SR_images(:,:,slice_te),HR_images(:,:,slice_te));
peaksnr_lrte = psnr(LR_images(:,:,slice_te),HR_images(:,:,slice_te));


figure;set(gcf,'Position',get(0,'ScreenSize'));
subplot(1,3,1); imagesc(HR_images(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('HRGT test'); 
subplot(1,3,2); imagesc(SR_images(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('SR test');xlabel({strcat('SSIM:',32,num2str(ssimval_srte)),strcat('PSNR:',32,num2str(peaksnr_srte))});
subplot(1,3,3); imagesc(LR_images(:,:,slice_te),[0 WT]);colormap(gray);axis image;title('LR test');xlabel({strcat('SSIM:',32,num2str(ssimval_lrte)),strcat('PSNR:',32,num2str(peaksnr_lrte))});


%%
WT_fft = 5;
figure;set(gcf,'Position',get(0,'ScreenSize'));
subplot(1,3,1); imagesc(fftshift(abs(fftn(HR_images(:,:,slice_te)))),[0 WT_fft]);colormap(gray);axis image;title('HRGT test'); 
subplot(1,3,2); imagesc(fftshift(abs(fftn(SR_images(:,:,slice_te)))),[0 WT_fft]);colormap(gray);axis image;title('SR test');xlabel({strcat('SSIM:',32,num2str(ssimval_srte)),strcat('PSNR:',32,num2str(peaksnr_srte))});
subplot(1,3,3); imagesc(fftshift(abs(fftn(LR_images(:,:,slice_te)))),[0 WT_fft]);colormap(gray);axis image;title('LR test');xlabel({strcat('SSIM:',32,num2str(ssimval_lrte)),strcat('PSNR:',32,num2str(peaksnr_lrte))});
