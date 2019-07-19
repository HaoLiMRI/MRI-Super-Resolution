clear
clc

scale_factor = 6;
len = length(dir('D:\HaoLi\SR\data\HRGT_org\HRGT_org_*.mat'));
test_data = 1;                      % Training data and test data switch. 1: training data, 0: test data. Training data will dispose pure background segments.
zerofilling_cut_switch = 1;         % Kspace zero-filling or cutting. Cutting decrease the size of LR image.1: zero-filling, 0: cutting.
fillingmode = 1;                    % Kspace continuous or interleaved zero-filling. 1: continuous, 0: interleaved.
calibration_lines = 16;             % Nember of remained lines in kspace center.


for num=1:len
    filename1=strcat('D:\HaoLi\SR\data\HRGT_org\HRGT_org_',num2str(num),'.mat'); 
    load(filename1);
    [HRGT, LR]=LR_generate(IMG1,num,scale_factor,test_data,zerofilling_cut_switch,fillingmode,calibration_lines); 
    
    filename2=strcat('D:\HaoLi\SR\data\',num2str(scale_factor),'_folds\data',num2str(num),'.mat');
    save(filename2, 'HRGT', 'LR', '-v7.3');
end