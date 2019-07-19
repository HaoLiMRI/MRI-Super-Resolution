function [HRGT,LR] = LR_generate(IMG1,num,scale_factor,test_data,zerofilling_cut_switch,fillingmode,calibration_lines)
%LR_GENERATE 此处显示有关此函数的摘要
%   此处显示详细说明
    IMG1=IMG1/max(max(max(IMG1)));
    [dim1,dim2,dim3]=size(IMG1);
    kspace=fftn(IMG1);
    
    if zerofilling_cut_switch
        if fillingmode
            kspace(:,round(dim2*0.5/scale_factor)+1:round(dim2*(scale_factor-0.5)/scale_factor),:)=0;
%           round(dim2*0.5/scale_factor)+1
%           round(dim2*(scale_factor-0.5)/scale_factor)
            kspace(:,:,round(dim3*0.5/scale_factor)+1:round(dim3*(scale_factor-0.5)/scale_factor))=0;
%           round(dim3*0.5/scale_factor)+1
%           round(dim3*(scale_factor-0.5)/scale_factor)
        else
            for i=1:dim2
                if mod(i,scale_factor)~=1
                    if ((i>(calibration_lines/2)) && (i<(dim2+1-calibration_lines/2)))
                        kspace(:,i,:) = 0;
                    end
                    if ((i>(calibration_lines/2)) && (i<(dim3+1-calibration_lines/2)))
                        kspace(:,:,i) = 0;
                    end
                end
            end
        end
%         figure;imagesc(fftshift(real(permute(kspace(200,:,:),[2,3,1]))),[0 100]),colormap(gray);axis image;
        IMG2=ifftn(kspace);
        IMG2(:,:,:)=sqrt(real(IMG2(:,:,:)).^2+imag(IMG2(:,:,:)).^2);
        IMG2=IMG2/max(max(max(IMG2)));
    else
        kspace_0=fftshift(kspace);
        kspace_cut=kspace_0(dim1*0.5*(scale_factor-1)/scale_factor+1:dim1*0.5*(scale_factor+1)/scale_factor,dim2*0.5*(scale_factor-1)/scale_factor+1:dim2*0.5*(scale_factor+1)/scale_factor,dim3*0.5*(scale_factor-1)/scale_factor+1:dim3*0.5*(scale_factor+1)/scale_factor);
        figure;imagesc(abs(permute(kspace_cut(round(200/scale_factor),:,:),[2,3,1])),[0 100]),colormap(gray);axis image;
        IMG2=ifftn(fftshift(kspace_cut));
        IMG2(:,:,:)=sqrt(real(IMG2(:,:,:)).^2+imag(IMG2(:,:,:)).^2);
        IMG2=IMG2/max(max(max(IMG2)));
    end

%%
%     slc = 65;
%     WW = 0.4;
%     [a,b]=ssim(IMG2(:,:,slc),IMG1(:,:,slc));
%     p = psnr(IMG2(:,:,slc),IMG1(:,:,slc));
%     figure;set(gcf,'Position',get(0,'ScreenSiz'));% title({strcat(num2str(scale_factor),32,'folds,',32,'slice',32,num2str(slc))})
%     subplot(1,2,1); imagesc(IMG1(:,:,slc),[0 WW]);colormap(gray);axis image;title('HRGT');
%     subplot(1,2,2); imagesc(IMG2(:,:,slc),[0 WW]);colormap(gray);axis image;title('LR');xlabel({strcat('SSIM:',32,num2str(a)),strcat('PSNR:',32,num2str(p))});

%%
%     slc = 15;
    %    #1  #2  #3  #4  #5  #6  #7  #8  #9 #10 #11 #12 #13 #14 #15 #16 #17 #18 #19 #20 #21 #22 #23 #24 #25 #26 #27 #28
    le=[ 30, 25, 25, 30, 20, 20, 27, 25, 25, 20, 35, 22, 28, 20, 15, 20, 25, 25, 25, 23, 25, 30, 15, 25, 27, 20, 20, 30];
    ri=[285,280,280,285,275,275,282,280,280,275,290,277,283,275,270,275,280,280,280,278,280,285,270,280,282,275,275,285];
    to=[ 35, 35, 35, 22, 22, 10, 40, 40, 20, 45, 10, 45,  1, 15, 15, 30, 20, 45, 50, 45, 25, 33, 30, 10, 25, 30, 30, 38];
    bo=[290,290,290,277,277,265,295,295,275,300,265,300,256,270,270,285,275,300,305,300,280,288,285,265,280,285,285,293];

    IMG3=IMG1(to(num):bo(num),le(num):ri(num),:);
    IMG4=IMG2(to(num):bo(num),le(num):ri(num),:);
%     [a,b]=ssim(IMG4(:,:,slc),IMG3(:,:,slc));
%     p = psnr(IMG4(:,:,slc),IMG3(:,:,slc));
%     figure;set(gcf,'Position',get(0,'ScreenSiz'));% title({strcat(num2str(scale_factor),32,'folds,',32,'slice',32,num2str(slc))})
%     subplot(1,2,1); imagesc(IMG3(:,:,slc),[0 WW]);colormap(gray);axis image;title('HRGT');
%     subplot(1,2,2); imagesc(IMG4(:,:,slc),[0 WW]);colormap(gray);axis image;title('LR');xlabel({strcat('SSIM:',32,num2str(a)),strcat('PSNR:',32,num2str(p))});


%%
% 2D generator
    counter=1;
%     noise_threshold = 0.03;
%     noise_ratio = 0.9;
%     ave_signal = 0.1;
%     total_threshold = floor(64 * 64 * (noise_threshold * noise_ratio + ave_signal * (1 - noise_ratio)));

    if test_data==0
        stride = 64;
    else
        stride = 32;
    end
    
    for k=1:size(IMG3,3)
        for i=1:stride:size(IMG3,1)-63
            for j=1:stride:size(IMG3,2)-63
                counter;
                HRGT(:,:,counter)=IMG3(i:i+63,j:j+63,k);
                LR(:,:,counter)=IMG4(i:i+63,j:j+63,k);
%                 if(sum(sum(sum(HRGT(:,:,counter))))>=total_threshold)
                if test_data == 0
                    counter = counter + 1;
                else
                    if std2(HRGT(:,:,counter))>=0.005
                        counter=counter+1;
                    end
                end
            end
        end
    end
end

