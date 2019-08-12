clear
clc

scale_factor = 8;
len = length(dir('D:\HaoLi\SR\data\HRGT_org\HRGT_org_*.mat'));
test_data = 1;                      % Training data and test data switch. 1: training data, 0: test data. Training data will dispose pure background segments.
zerofilling_cut_switch = 1;         % Kspace zero-filling or cutting. Cutting decrease the size of LR image.1: zero-filling, 0: cutting.
fillingmode = 1;                    % Kspace continuous or interleaved zero-filling. 1: continuous, 0: interleaved.
calibration_lines = 16;             % Nember of remained lines in kspace center.
x_downsampling = 0;                 % Downsampling in x-direction. 1: yes, 0: no.
switch_2d_3d = 1;                   % 2d/3d kspace downsampling. 1: 2d, 0: 3d. 
if switch_2d_3d
    dim = 2;
else
    dim = 3;
end
rand_separate=[6,3,5,7,7,4,4,10,1,9,10,8,1,3,4,7,2,8,2,7,5,8,8,10,9,4,7,2];
counter_training=0;
counter_testing=0;

for num=1:len
    filename1=strcat('D:\HaoLi\SR\data\HRGT_org\HRGT_org_',num2str(num),'.mat')
    load(filename1);
    
    if rand_separate(num)<10
        test_data = 1;
        [HRGT, LR]=LR_generate(IMG1,num,scale_factor,test_data,zerofilling_cut_switch,fillingmode,calibration_lines,x_downsampling,switch_2d_3d); 
        counter_training=counter_training+1;
        filename2=strcat('D:\HaoLi\SR\data\',num2str(scale_factor),'_folds_',num2str(dim),'d\training\data',num2str(counter_training),'.mat');
        save(filename2, 'HRGT', 'LR', '-v7.3');
    else
        test_data = 0;
        [HRGT, LR]=LR_generate(IMG1,num,scale_factor,test_data,zerofilling_cut_switch,fillingmode,calibration_lines,x_downsampling,switch_2d_3d); 
        counter_testing=counter_testing+1;
        filename2=strcat('D:\HaoLi\SR\data\',num2str(scale_factor),'_folds_',num2str(dim),'d\evaluation\data',num2str(counter_testing),'.mat');
        save(filename2, 'HRGT', 'LR', '-v7.3');
    end
end