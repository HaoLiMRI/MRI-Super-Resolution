clear
clc

scale_factor = 8;
switch_2d_3d = 1;                    % 2d/3d kspace downsampling. 1: 2d, 0: 3d. 
if switch_2d_3d
    dim = 2;
else
    dim = 3;
end

filename = strcat('D:\HaoLi\SR\data\',num2str(scale_factor),'_folds_',num2str(dim),'d\data*.mat');
len = length(dir(filename));

for i = 1:len
    filename = strcat('D:\HaoLi\SR\data\',num2str(scale_factor),'_folds_',num2str(dim),'d\data',num2str(i),'.mat')

    load(filename);
    len_1 = size(LR,3);
    if i==1
        len_2 = 0;
    else
        len_2 = size(LR_all,3);
    end
    LR_all(:,:,len_2+1:len_2+len_1) = LR;
    clear LR
    HRGT_all(:,:,len_2+1:len_2+len_1) = HRGT;
    clear HRGT
end

fn_LR=strcat('D:\HaoLi\SR\data\LR_',num2str(scale_factor),'_folds_',num2str(dim),'d.mat');
LR = LR_all;
clear LR_all
save(fn_LR,'LR','-v7.3');

fn_HRGT=strcat('D:\HaoLi\SR\data\HRGT_',num2str(scale_factor),'_folds_',num2str(dim),'d.mat');
HRGT = HRGT_all;
clear HRGT_all
save(fn_HRGT,'HRGT','-v7.3');

clear