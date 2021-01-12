import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


import os
import numpy as np
import h5py
import torch as tc
import math


class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2,0,1))
        LR_sr = LR_sr.transpose((2,0,1))
        HR = HR.transpose((2,0,1))
        Ref = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}


class TrainSet(Dataset):
    def __init__(self, args):
        """""""""""""""""""""""""""""""""""""""""""""
        1.1 MRI HR and LR Data pair preprocessing training part
        """""""""""""""""""""""""""""""""""""""""""""
        "The folder where to load the training LR, HR data pair"
        folder_log_path = args.dataset_dir
        file_names = os.listdir(folder_log_path)

        num_low_resolution_mat_file = 0
        num_zero_padded_low_resolution_mat_file = 0
        num_high_resolution_groundtruth_mat_file = 0
        num_reference_mat_file = 0
        num_reference_down_up_mat_file = 0

        for idx_file in file_names:
            print(idx_file)
            if 'LR_training_2' in os.path.join(folder_log_path, idx_file):
                print('One more low resolution image set exist')
                num_low_resolution_mat_file = num_low_resolution_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                data_low_resolution = file_data_low_resolution['LR'][:] #----- numpy array
                print('Training data: Shape of LR data is: ', np.shape(data_low_resolution))
                print(data_low_resolution.dtype)
                torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_low_resolution = torch_data_low_resolution.permute(0, 1, 3, 2)
                print('Training data: Shape of LR data in Torch is: ', np.shape(torch_data_low_resolution))
                if num_low_resolution_mat_file == 1:
                    torch_data_low_resolution_sequence = torch_data_low_resolution
                elif num_low_resolution_mat_file > 1:
                    print(num_low_resolution_mat_file)
                    torch_data_low_resolution_sequence = tc.cat((torch_data_low_resolution_sequence, torch_data_low_resolution), 0)
                print('Training data: Shape of LR data sequence in Torch is: ', np.shape(torch_data_low_resolution_sequence))
            elif 'LR_UP_training_2' in os.path.join(folder_log_path, idx_file):
                print('One more zero padded low resolution image set exist')
                num_zero_padded_low_resolution_mat_file = num_zero_padded_low_resolution_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_zero_padded_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                """ data_zero_padded_low_resolution = file_data_zero_padded_low_resolution['LRUP'][:] #----- numpy array """
                data_zero_padded_low_resolution = file_data_zero_padded_low_resolution['HRGT'][:] # Hardcode it as HRGT just for using the wrong data for now!
                print('Training data: Shape of LR_UP data is: ', np.shape(data_zero_padded_low_resolution))
                torch_data_zero_padded_low_resolution = tc.from_numpy(data_zero_padded_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_zero_padded_low_resolution = torch_data_zero_padded_low_resolution.permute(0, 1, 3, 2)
                print('Training data: Shape of LR_UP data in Torch is: ', np.shape(torch_data_zero_padded_low_resolution))
                if num_zero_padded_low_resolution_mat_file == 1:
                    torch_data_zero_padded_low_resolution_sequence = torch_data_zero_padded_low_resolution
                if num_zero_padded_low_resolution_mat_file > 1:
                    print(num_zero_padded_low_resolution_mat_file)
                    torch_data_zero_padded_low_resolution_sequence = tc.cat((torch_data_zero_padded_low_resolution_sequence, torch_data_zero_padded_low_resolution), 0)
                print('Training data: Shape of LR_UP data sequence in Torch is: ', np.shape(torch_data_zero_padded_low_resolution_sequence))
            elif 'HRGT_training_2' in os.path.join(folder_log_path, idx_file):
                print('One more high resolution groundtruth image set exist')
                num_high_resolution_groundtruth_mat_file = num_high_resolution_groundtruth_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_high_resolution_groundtruth = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                data_high_resolution_groundtruth = file_data_high_resolution_groundtruth['HRGT'][:] #----- numpy array
                print('Training data: Shape of HR data is: ', np.shape(data_high_resolution_groundtruth))
                torch_data_high_resolution_groundtruth = tc.from_numpy(data_high_resolution_groundtruth) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.permute(0, 1, 3, 2)
                print('Training data: Shape of HR data in Torch is: ', np.shape(torch_data_high_resolution_groundtruth))
                if num_high_resolution_groundtruth_mat_file == 1:
                    torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth
                if num_high_resolution_groundtruth_mat_file > 1:
                    print(num_high_resolution_groundtruth_mat_file)
                    torch_data_high_resolution_groundtruth_sequence = tc.cat((torch_data_high_resolution_groundtruth_sequence, torch_data_high_resolution_groundtruth), 0)
                print('Training data: Shape of HR data sequence in Torch is: ', np.shape(torch_data_high_resolution_groundtruth_sequence))
            elif 'REF_training_2' in os.path.join(folder_log_path, idx_file):
                print('One more reference high resolution image set exist')
                num_reference_mat_file = num_reference_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_reference = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                """ data_reference = file_data_reference['REF'][:] #----- numpy array """
                data_reference = file_data_reference['HRGT'][:] # Hardcode it as HRGT just for using the wrong data for now!
                print('Training data: Shape of REF data is: ', np.shape(data_reference))
                torch_data_reference = tc.from_numpy(data_reference) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_reference = torch_data_reference.permute(0, 1, 3, 2)
                print('Training data: Shape of REF data in Torch is: ', np.shape(torch_data_reference))
                if num_reference_mat_file == 1:
                    torch_data_reference_sequence = torch_data_reference
                if num_reference_mat_file > 1:
                    print(num_reference_mat_file)
                    torch_data_reference_sequence = tc.cat((torch_data_reference_sequence, torch_data_reference), 0)
                print('Training data: Shape of REF data sequence in Torch is: ', np.shape(torch_data_reference_sequence))
            elif 'REF_DOWN_UP_training_2' in os.path.join(folder_log_path, idx_file):
                print('One more down and up sampled(zero padded) reference high resolution image set exist')
                num_reference_down_up_mat_file = num_reference_down_up_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_reference_down_up = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                """ data_reference_down_up = file_data_reference_down_up['REF_DOWN_UP'][:] #----- numpy array """
                data_reference_down_up = file_data_reference_down_up['HRGT'][:] # Hardcode it as HRGT just for using the wrong data for now!
                print('Training data: Shape of REF DOWN UP data is: ', np.shape(data_reference_down_up))
                torch_data_reference_down_up = tc.from_numpy(data_reference_down_up) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_reference_down_up = torch_data_reference_down_up.permute(0, 1, 3, 2)
                print('Training data: Shape of REF DOWN UP data in Torch is: ', np.shape(torch_data_reference_down_up))
                if num_reference_down_up_mat_file == 1:
                    torch_data_reference_down_up_sequence = torch_data_reference_down_up
                if num_reference_down_up_mat_file > 1:
                    print(num_reference_down_up_mat_file)
                    torch_data_reference_down_up_sequence = tc.cat((torch_data_reference_down_up_sequence, torch_data_reference_down_up), 0)
                print('Training data: Shape of REF DOWN UP data sequence in Torch is: ', np.shape(torch_data_reference_down_up_sequence))
            else:
                print('other type NOT support for now')

        """ num_training_samples = math.floor(torch_data_low_resolution_sequence.size(0))
        print('number of training samples is: ', num_training_samples) """
        print('All mat files have been concatenated into one tensor for each type, training data is ready to be loaded!')

        """""""""""""""""""""""""""""""""""""""""""""
        1.2 Load MRI HR and LR Data pair training part
        """""""""""""""""""""""""""""""""""""""""""""
        self.torch_data_low_resolution_training_sequence = torch_data_low_resolution_sequence.float()
        self.torch_data_zero_padded_low_resolution_sequence = torch_data_zero_padded_low_resolution_sequence.float()
        self.torch_data_high_resolution_groundtruth_training_sequence = torch_data_high_resolution_groundtruth_sequence.float()
        self.torch_data_reference_sequence = torch_data_reference_sequence.float()
        self.torch_data_reference_down_up_sequence = torch_data_reference_down_up_sequence.float()

    def __len__(self):
        return len(self.torch_data_low_resolution_training_sequence)

    def __getitem__(self, idx):
        ### LR
        LR = self.torch_data_low_resolution_training_sequence[idx]
        ### LR_sr
        LR_sr = self.torch_data_zero_padded_low_resolution_sequence[idx]
        ### HR
        HR = self.torch_data_high_resolution_groundtruth_training_sequence[idx]
        ### Ref
        Ref = self.torch_data_reference_sequence[idx]
        ### Ref_sr
        Ref_sr = self.torch_data_reference_down_up_sequence[idx]

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        return sample


class EvaluationSet(Dataset):
    def __init__(self, args):
        """""""""""""""""""""""""""""""""""""""""""""
        2.1 MRI HR and LR Validation Data pair preprocessing validation(evaluation) part
        """""""""""""""""""""""""""""""""""""""""""""
        "The folder where to load the validation(evaluation) LR, HR data pair"
        folder_log_path = args.dataset_dir
        file_names = os.listdir(folder_log_path)

        num_low_resolution_mat_file = 0
        num_zero_padded_low_resolution_mat_file = 0
        num_high_resolution_groundtruth_mat_file = 0
        num_reference_mat_file = 0
        num_reference_down_up_mat_file = 0

        for idx_file in file_names:
            print(idx_file)
            if 'LR_validation_2' in os.path.join(folder_log_path, idx_file):
                print('One more low resolution image set exist')
                num_low_resolution_mat_file = num_low_resolution_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                data_low_resolution = file_data_low_resolution['LR'][:] #----- numpy array
                print('Evaluation data: Shape of LR data is: ', np.shape(data_low_resolution))
                print(data_low_resolution.dtype)
                torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_low_resolution = torch_data_low_resolution.permute(0, 1, 3, 2)
                print('Evaluation data: Shape of LR data in Torch is: ', np.shape(torch_data_low_resolution))
                if num_low_resolution_mat_file == 1:
                    torch_data_low_resolution_sequence = torch_data_low_resolution
                elif num_low_resolution_mat_file > 1:
                    print(num_low_resolution_mat_file)
                    torch_data_low_resolution_sequence = tc.cat((torch_data_low_resolution_sequence, torch_data_low_resolution), 0)
                print('Evaluation data: Shape of LR data sequence in Torch is: ', np.shape(torch_data_low_resolution_sequence))
            elif 'LR_UP_validation_2' in os.path.join(folder_log_path, idx_file):
                print('One more zero padded low resolution image set exist')
                num_zero_padded_low_resolution_mat_file = num_zero_padded_low_resolution_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_zero_padded_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                """ data_zero_padded_low_resolution = file_data_zero_padded_low_resolution['LRUP'][:] #----- numpy array """
                data_zero_padded_low_resolution = file_data_zero_padded_low_resolution['HRGT'][:] # Hardcode it as HRGT just for using the wrong data for now!
                print('Evaluation data: Shape of LR_UP data is: ', np.shape(data_zero_padded_low_resolution))
                torch_data_zero_padded_low_resolution = tc.from_numpy(data_zero_padded_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_zero_padded_low_resolution = torch_data_zero_padded_low_resolution.permute(0, 1, 3, 2)
                print('Evaluation data: Shape of LR_UP data in Torch is: ', np.shape(torch_data_zero_padded_low_resolution))
                if num_zero_padded_low_resolution_mat_file == 1:
                    torch_data_zero_padded_low_resolution_sequence = torch_data_zero_padded_low_resolution
                if num_zero_padded_low_resolution_mat_file > 1:
                    print(num_zero_padded_low_resolution_mat_file)
                    torch_data_zero_padded_low_resolution_sequence = tc.cat((torch_data_zero_padded_low_resolution_sequence, torch_data_zero_padded_low_resolution), 0)
                print('Evaluation data: Shape of LR_UP data sequence in Torch is: ', np.shape(torch_data_zero_padded_low_resolution_sequence))
            elif 'HRGT_validation_2' in os.path.join(folder_log_path, idx_file):
                print('One more high resolution groundtruth image set exist')
                num_high_resolution_groundtruth_mat_file = num_high_resolution_groundtruth_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_high_resolution_groundtruth = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                data_high_resolution_groundtruth = file_data_high_resolution_groundtruth['HRGT'][:] #----- numpy array
                print('Evaluation data: Shape of HR data is: ', np.shape(data_high_resolution_groundtruth))
                torch_data_high_resolution_groundtruth = tc.from_numpy(data_high_resolution_groundtruth) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.permute(0, 1, 3, 2)
                print('Evaluation data: Shape of HR data in Torch is: ', np.shape(torch_data_high_resolution_groundtruth))
                if num_high_resolution_groundtruth_mat_file == 1:
                    torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth
                if num_high_resolution_groundtruth_mat_file > 1:
                    print(num_high_resolution_groundtruth_mat_file)
                    torch_data_high_resolution_groundtruth_sequence = tc.cat((torch_data_high_resolution_groundtruth_sequence, torch_data_high_resolution_groundtruth), 0)
                print('Evaluation data: Shape of HR data sequence in Torch is: ', np.shape(torch_data_high_resolution_groundtruth_sequence))
            elif 'REF_validation_2' in os.path.join(folder_log_path, idx_file):
                print('One more reference high resolution image set exist')
                num_reference_mat_file = num_reference_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_reference = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                """ data_reference = file_data_reference['REF'][:] #----- numpy array """
                data_reference = file_data_reference['HRGT'][:] # Hardcode it as HRGT just for using the wrong data for now!
                print('Evaluation data: Shape of REF data is: ', np.shape(data_reference))
                torch_data_reference = tc.from_numpy(data_reference) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_reference = torch_data_reference.permute(0, 1, 3, 2)
                print('Evaluation data: Shape of REF data in Torch is: ', np.shape(torch_data_reference))
                if num_reference_mat_file == 1:
                    torch_data_reference_sequence = torch_data_reference
                if num_reference_mat_file > 1:
                    print(num_reference_mat_file)
                    torch_data_reference_sequence = tc.cat((torch_data_reference_sequence, torch_data_reference), 0)
                print('Evaluation data: Shape of REF data sequence in Torch is: ', np.shape(torch_data_reference_sequence))
            elif 'REF_DOWN_UP_validation_2' in os.path.join(folder_log_path, idx_file):
                print('One more down and up sampled(zero padded) reference high resolution image set exist')
                num_reference_down_up_mat_file = num_reference_down_up_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_reference_down_up = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                """ data_reference_down_up = file_data_reference_down_up['REF_DOWN_UP'][:] #----- numpy array """
                data_reference_down_up = file_data_reference_down_up['HRGT'][:] # Hardcode it as HRGT just for using the wrong data for now!
                print('Evaluation data: Shape of REF DOWN UP data is: ', np.shape(data_reference_down_up))
                torch_data_reference_down_up = tc.from_numpy(data_reference_down_up) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_reference_down_up = torch_data_reference_down_up.permute(0, 1, 3, 2)
                print('Evaluation data: Shape of REF DOWN UP data in Torch is: ', np.shape(torch_data_reference_down_up))
                if num_reference_down_up_mat_file == 1:
                    torch_data_reference_down_up_sequence = torch_data_reference_down_up
                if num_reference_down_up_mat_file > 1:
                    print(num_reference_down_up_mat_file)
                    torch_data_reference_down_up_sequence = tc.cat((torch_data_reference_down_up_sequence, torch_data_reference_down_up), 0)
                print('Evaluation data: Shape of REF DOWN UP data sequence in Torch is: ', np.shape(torch_data_reference_down_up_sequence))
            else:
                print('other type NOT support for now')

        """ num_evaluation_samples = math.floor(torch_data_low_resolution_sequence.size(0))
        print('number of evaluation samples is: ', num_evaluation_samples) """
        print('All mat files have been concatenated into one tensor for each type, evaluation(validation) data is ready to be loaded!')

        """""""""""""""""""""""""""""""""""""""""""""
        2.2 Load MRI HR and LR Validation Data pair validation(evaluation) part
        """""""""""""""""""""""""""""""""""""""""""""
        self.torch_data_low_resolution_evaluation_sequence = torch_data_low_resolution_sequence.float()
        self.torch_data_zero_padded_low_resolution_sequence = torch_data_zero_padded_low_resolution_sequence.float()
        self.torch_data_high_resolution_groundtruth_evaluation_sequence = torch_data_high_resolution_groundtruth_sequence.float()
        self.torch_data_reference_sequence = torch_data_reference_sequence.float()
        self.torch_data_reference_down_up_sequence = torch_data_reference_down_up_sequence.float()

    def __len__(self):
        return len(self.torch_data_low_resolution_evaluation_sequence)

    def __getitem__(self, idx):
        ### LR
        LR = self.torch_data_low_resolution_evaluation_sequence[idx]
        ### LR_sr
        LR_sr = self.torch_data_zero_padded_low_resolution_sequence[idx]
        ### HR
        HR = self.torch_data_high_resolution_groundtruth_evaluation_sequence[idx]
        ### Ref
        Ref = self.torch_data_reference_sequence[idx]
        ### Ref_sr
        Ref_sr = self.torch_data_reference_down_up_sequence[idx]

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        return sample



class FinalTestSet(Dataset):
    def __init__(self, args):
        """""""""""""""""""""""""""""""""""""""""""""
        2.1 MRI HR and LR Test Data pair preprocessing test part
        """""""""""""""""""""""""""""""""""""""""""""
        "The folder where to load the test LR, HR data pair"
        folder_log_path = args.dataset_dir
        file_names = os.listdir(folder_log_path)

        num_low_resolution_mat_file = 0
        num_zero_padded_low_resolution_mat_file = 0
        num_high_resolution_groundtruth_mat_file = 0
        num_reference_mat_file = 0
        num_reference_down_up_mat_file = 0

        for idx_file in file_names:
            print(idx_file)
            if 'LR_test_2' in os.path.join(folder_log_path, idx_file):
                print('One more low resolution image set exist')
                num_low_resolution_mat_file = num_low_resolution_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                data_low_resolution = file_data_low_resolution['LR'][:] #----- numpy array
                print('Testing data: Shape of LR data is: ', np.shape(data_low_resolution))
                print(data_low_resolution.dtype)
                torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_low_resolution = torch_data_low_resolution.permute(0, 1, 3, 2)
                print('Testing data: Shape of LR data in Torch is: ', np.shape(torch_data_low_resolution))
                if num_low_resolution_mat_file == 1:
                    torch_data_low_resolution_sequence = torch_data_low_resolution
                elif num_low_resolution_mat_file > 1:
                    print(num_low_resolution_mat_file)
                    torch_data_low_resolution_sequence = tc.cat((torch_data_low_resolution_sequence, torch_data_low_resolution), 0)
                print('Testing data: Shape of LR data sequence in Torch is: ', np.shape(torch_data_low_resolution_sequence))
            elif 'LR_UP_test_2' in os.path.join(folder_log_path, idx_file):
                print('One more zero padded low resolution image set exist')
                num_zero_padded_low_resolution_mat_file = num_zero_padded_low_resolution_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_zero_padded_low_resolution = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                """ data_zero_padded_low_resolution = file_data_zero_padded_low_resolution['LRUP'][:] #----- numpy array """
                data_zero_padded_low_resolution = file_data_zero_padded_low_resolution['HRGT'][:] # Hardcode it as HRGT just for using the wrong data for now!
                print('Testing data: Shape of LR_UP data is: ', np.shape(data_zero_padded_low_resolution))
                torch_data_zero_padded_low_resolution = tc.from_numpy(data_zero_padded_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_zero_padded_low_resolution = torch_data_zero_padded_low_resolution.permute(0, 1, 3, 2)
                print('Testing data: Shape of LR_UP data in Torch is: ', np.shape(torch_data_zero_padded_low_resolution))
                if num_zero_padded_low_resolution_mat_file == 1:
                    torch_data_zero_padded_low_resolution_sequence = torch_data_zero_padded_low_resolution
                if num_zero_padded_low_resolution_mat_file > 1:
                    print(num_zero_padded_low_resolution_mat_file)
                    torch_data_zero_padded_low_resolution_sequence = tc.cat((torch_data_zero_padded_low_resolution_sequence, torch_data_zero_padded_low_resolution), 0)
                print('Testing data: Shape of LR_UP data sequence in Torch is: ', np.shape(torch_data_zero_padded_low_resolution_sequence))
            elif 'HRGT_test_2' in os.path.join(folder_log_path, idx_file):
                print('One more high resolution groundtruth image set exist')
                num_high_resolution_groundtruth_mat_file = num_high_resolution_groundtruth_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_high_resolution_groundtruth = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                data_high_resolution_groundtruth = file_data_high_resolution_groundtruth['HRGT'][:] #----- numpy array
                print('Testing data: Shape of HR data is: ', np.shape(data_high_resolution_groundtruth))
                torch_data_high_resolution_groundtruth = tc.from_numpy(data_high_resolution_groundtruth) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.permute(0, 1, 3, 2)
                print('Testing data: Shape of HR data in Torch is: ', np.shape(torch_data_high_resolution_groundtruth))
                if num_high_resolution_groundtruth_mat_file == 1:
                    torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth
                if num_high_resolution_groundtruth_mat_file > 1:
                    print(num_high_resolution_groundtruth_mat_file)
                    torch_data_high_resolution_groundtruth_sequence = tc.cat((torch_data_high_resolution_groundtruth_sequence, torch_data_high_resolution_groundtruth), 0)
                print('Testing data: Shape of HR data sequence in Torch is: ', np.shape(torch_data_high_resolution_groundtruth_sequence))
            elif 'REF_test_2' in os.path.join(folder_log_path, idx_file):
                print('One more reference high resolution image set exist')
                num_reference_mat_file = num_reference_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_reference = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                """ data_reference = file_data_reference['REF'][:] #----- numpy array """
                data_reference = file_data_reference['HRGT'][:] # Hardcode it as HRGT just for using the wrong data for now!
                print('Testing data: Shape of REF data is: ', np.shape(data_reference))
                torch_data_reference = tc.from_numpy(data_reference) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_reference = torch_data_reference.permute(0, 1, 3, 2)
                print('Testing data: Shape of REF data in Torch is: ', np.shape(torch_data_reference))
                if num_reference_mat_file == 1:
                    torch_data_reference_sequence = torch_data_reference
                if num_reference_mat_file > 1:
                    print(num_reference_mat_file)
                    torch_data_reference_sequence = tc.cat((torch_data_reference_sequence, torch_data_reference), 0)
                print('Testing data: Shape of REF data sequence in Torch is: ', np.shape(torch_data_reference_sequence))
            elif 'REF_DOWN_UP_test_2' in os.path.join(folder_log_path, idx_file):
                print('One more down and up sampled(zero padded) reference high resolution image set exist')
                num_reference_down_up_mat_file = num_reference_down_up_mat_file + 1
                print(os.path.join(folder_log_path, idx_file))
                file_data_reference_down_up = h5py.File(os.path.join(folder_log_path, idx_file), 'r')
                """ data_reference_down_up = file_data_reference_down_up['REF_DOWN_UP'][:] #----- numpy array """
                data_reference_down_up = file_data_reference_down_up['HRGT'][:] # Hardcode it as HRGT just for using the wrong data for now!
                print('Testing data: Shape of REF DOWN UP data is: ', np.shape(data_reference_down_up))
                torch_data_reference_down_up = tc.from_numpy(data_reference_down_up) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_reference_down_up = torch_data_reference_down_up.permute(0, 1, 3, 2)
                print('Testing data: Shape of REF DOWN UP data in Torch is: ', np.shape(torch_data_reference_down_up))
                if num_reference_down_up_mat_file == 1:
                    torch_data_reference_down_up_sequence = torch_data_reference_down_up
                if num_reference_down_up_mat_file > 1:
                    print(num_reference_down_up_mat_file)
                    torch_data_reference_down_up_sequence = tc.cat((torch_data_reference_down_up_sequence, torch_data_reference_down_up), 0)
                print('Testing data: Shape of REF DOWN UP data sequence in Torch is: ', np.shape(torch_data_reference_down_up_sequence))
            else:
                print('other type NOT support for now')

        """ num_evaluation_samples = math.floor(torch_data_low_resolution_sequence.size(0))
        print('number of evaluation samples is: ', num_evaluation_samples) """
        print('All mat files have been concatenated into one tensor for each type, evaluation(validation) data is ready to be loaded!')

        """""""""""""""""""""""""""""""""""""""""""""
        2.2 Load MRI HR and LR Testing Data pair test part
        """""""""""""""""""""""""""""""""""""""""""""
        self.torch_data_low_resolution_test_sequence = torch_data_low_resolution_sequence.float()
        self.torch_data_zero_padded_low_resolution_sequence = torch_data_zero_padded_low_resolution_sequence.float()
        self.torch_data_high_resolution_groundtruth_test_sequence = torch_data_high_resolution_groundtruth_sequence.float()
        self.torch_data_reference_sequence = torch_data_reference_sequence.float()
        self.torch_data_reference_down_up_sequence = torch_data_reference_down_up_sequence.float()

    def __len__(self):
        return len(self.torch_data_low_resolution_test_sequence)

    def __getitem__(self, idx):
        ### LR
        LR = self.torch_data_low_resolution_test_sequence[idx]
        ### LR_sr
        LR_sr = self.torch_data_zero_padded_low_resolution_sequence[idx]
        ### HR
        HR = self.torch_data_high_resolution_groundtruth_test_sequence[idx]
        ### Ref
        Ref = self.torch_data_reference_sequence[idx]
        ### Ref_sr
        Ref_sr = self.torch_data_reference_down_up_sequence[idx]

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        return sample