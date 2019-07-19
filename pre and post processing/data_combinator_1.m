clear
clc

scale_factor = 6;
filename = strcat('D:\HaoLi\SR\data\',num2str(scale_factor),'_folds\data*.mat');
len = length(dir(filename));

for i = 1:len
    filename = strcat('D:\HaoLi\SR\data\',num2str(scale_factor),'_folds\data',num2str(i),'.mat')

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

fn_LR=strcat('D:\HaoLi\SR\data\LR_',num2str(scale_factor),'_folds.mat');
LR = LR_all;
clear LR_all
save(fn_LR,'LR','-v7.3');

fn_HRGT=strcat('D:\HaoLi\SR\data\HRGT_',num2str(scale_factor),'_folds.mat');
HRGT = HRGT_all;
clear HRGT_all
save(fn_HRGT,'HRGT','-v7.3');

clear