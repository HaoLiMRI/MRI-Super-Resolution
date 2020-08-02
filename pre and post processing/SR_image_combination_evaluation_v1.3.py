"""
This is the file which read all cropped MRI SR image, combine them into whole MRI SR image and evaluate using SSIM and PSNR

% Super resolution image quality evaluation using SSIM and PSNR
% SSIM is expected to be as close to 1 as possible
% PSNR is expected to be as large as possible
% SSIM and PSNR between SR and HR are calculated
"""
"""
Author: chisyliu@hotmail.com *
        hao.li@med.uni-heidelberg.de *
        
        * Both authors contribute equally
Version: 1.3.0
"""
import sys
import os
""" import h5py """
import scipy.io
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
import torch as tc


"load mat files of cropped data and combine several cropped images as whole one"
def loadMatFileDataAndCombineCroppedImage(path):
    file_names = os.listdir(path)
    image_data = {} # Empty dictionary which will save all the cropped image
    collection_of_entire_image = {} # Empty dictionary which will save all the entire image
    list_of_file_name_without_suffix = [] # Empty list which will save all key of dictionary of all the entire image
    for file_name in file_names:
        print(file_name)
        if 'test_image' in os.path.join(path, file_name): # only select test_image
            """ _image_file = h5py.File(os.path.join(path, file_name), 'r') # load the _image file from specific path """
            
            # Step 1: Read the .mat file and load into dictionary
            # Read the .mat file using scipy.io.loadmat
            _image_file = scipy.io.loadmat(os.path.join(path, file_name))
            print(type(_image_file))
            # Get rid of suffix
            file_name_without_suffix = file_name.split(".")[0]
            print(file_name_without_suffix)
            # Load the corrensponding data from the element of dictionary
            image_data[file_name_without_suffix] = _image_file[file_name_without_suffix]
            print("The shape of cropped %s is " %file_name_without_suffix, image_data[file_name_without_suffix].shape)

            # Step 2: Combine the cropped images into whole image
            number_of_entire_images = image_data[file_name_without_suffix].shape[0]//16
            h_cropped_image = image_data[file_name_without_suffix].shape[1]
            w_cropped_image = image_data[file_name_without_suffix].shape[2]
            entire_image = np.zeros((number_of_entire_images, 
                w_cropped_image*4, 
                h_cropped_image*4))
            for i in range(number_of_entire_images):
                for j in range(4):
                    for k in range(4):
                        entire_image[i, j*h_cropped_image:(j*h_cropped_image + h_cropped_image), k*w_cropped_image:(k*w_cropped_image + w_cropped_image)] = image_data[file_name_without_suffix][i*16 + j*4 + k, :, :]
            collection_of_entire_image[file_name_without_suffix] = entire_image
            list_of_file_name_without_suffix.append(file_name_without_suffix)
        else:
            print('other type NOT support for now')
    return collection_of_entire_image, list_of_file_name_without_suffix


"calculate SSIM between two images"
def calculateSSIM(img1, img2, max_I):
    C1 = (0.01 * max_I)**2
    C2 = (0.03 * max_I)**2
    C3 = C2/2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))

    luminance_factor = (2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)
    contrast_factor = (2*np.sqrt(sigma1_sq)*np.sqrt(sigma2_sq)+C2)/(sigma1_sq + sigma2_sq + C2)
    structure_factor = (sigma12 + C3)/(np.sqrt(sigma1_sq)*np.sqrt(sigma2_sq)+C3)

    return ssim_map.mean(), luminance_factor.mean(), contrast_factor.mean(), structure_factor.mean()


"calculate PSNR between two images"
def calculatePSNR(img1, img2, max_I):
    # max_I, 
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_I / math.sqrt(mse))


"calculate gradient map of input image using sobel operator"
def calculateGradientMap(img):
    # sobel operator
    vertical_edge_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_edge_mask = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_vertical_map = cv2.filter2D(img, cv2.CV_64F, vertical_edge_mask)
    gradient_horizontal_map = cv2.filter2D(img, cv2.CV_64F, horizontal_edge_mask)
    gradient_map = abs(gradient_vertical_map) + abs(gradient_horizontal_map)
    if Amplify_Small_Value_In_Gradient_Map == True:
        gradient_map = 1 - np.exp(-gradient_map) # 1 - exp(-x)
    return gradient_map

"calculate gradient map of input image using sobel operator, with Pytorch"
def calculateGradientMapUsingPytorch(img, Amplify_Small_Value_In_Gradient_Map):
    img = tc.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    # sobel operator
    vertical_edge_mask = tc.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_edge_mask = tc.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    vertical_edge_mask = vertical_edge_mask.float().unsqueeze(0).unsqueeze(0)
    horizontal_edge_mask = horizontal_edge_mask.float().unsqueeze(0).unsqueeze(0)

    gradient_vertical_map = tc.nn.functional.conv2d(img, vertical_edge_mask, padding = 1, stride = 1, groups = 1)
    gradient_horizontal_map = tc.nn.functional.conv2d(img, horizontal_edge_mask, padding = 1, stride = 1, groups = 1)

    gradient_map = abs(gradient_vertical_map) + abs(gradient_horizontal_map)

    if Amplify_Small_Value_In_Gradient_Map == True:
        gradient_map = 1 - tc.exp(-gradient_map) # 1 - exp(-x)

    return gradient_map[0, 0, :, :]


"calculate k space data of input image, with Pytorch"
def calculateKSpaceUsingPytorch(img):
    img = tc.from_numpy(img).float()
    k_space_data = tc.rfft(img, signal_ndim = 2, onesided = False)

    return k_space_data


"wrapper to run all functionality"
def runSrImageCombinationAndEvaluation(Amplify_Small_Value_In_Gradient_Map):
    # Set up the folder where the .mat files exist
    """ file_dir = "D:/Tech_Resource/Paper_Resource/MRI SR以及相关论文/our_project_code/result/20200723_result_data_RCAN_l1_gradssim_laf_100_32_2folds_2d_downsize" """
    file_dir = "D:/Tech_Resource/Paper_Resource/MRI SR以及相关论文/our_project_code/result/20200728_result_data_RCAN_l1_laf_100_32_2folds_2d_downsize(without_gradssim_l1_loss)"
    # Load the .mat files and combine cropped images into entire image and return
    collection_of_entire_image, list_of_file_name_without_suffix = loadMatFileDataAndCombineCroppedImage(file_dir)
    print(type(collection_of_entire_image))

    for i in range(len(list_of_file_name_without_suffix)):
        print("The size of entire %s is " %list_of_file_name_without_suffix[i], collection_of_entire_image[list_of_file_name_without_suffix[i]].shape)
        if list_of_file_name_without_suffix[i] == "HR_test_image":
            entire_hr_image_data = collection_of_entire_image[list_of_file_name_without_suffix[i]]
        elif list_of_file_name_without_suffix[i] == "SR_test_image":
            entire_sr_image_data = collection_of_entire_image[list_of_file_name_without_suffix[i]]
        else:
            pass
    
    for i in range(10): # Only plot and calculate 10 SR HR image pairs among all entire_hr_image_data.shape[0] pairs SR HR
        print("max_I of entire_sr_image_data is", np.amax(entire_sr_image_data))
        print("max_I of entire_hr_image_data is", np.amax(entire_hr_image_data))
        # Here calculate SSIM of SR HR image pair
        ssim_between_sr_hr, luminance_factor_between_sr_hr, contrast_factor_between_sr_hr, structure_factor_between_sr_hr = calculateSSIM(entire_sr_image_data[i*10], entire_hr_image_data[i*10], max_I = 1.0)   # max_I是表示图像点颜色的最大数值
        print("SSIM between %d th SR HR MRI image: " %(i*10 + 1), ssim_between_sr_hr)
        print("luminance_factor between %d th SR HR MRI image: " %(i*10 + 1), luminance_factor_between_sr_hr)
        print("contrast_factor between %d th SR HR MRI image: " %(i*10 + 1), contrast_factor_between_sr_hr)
        print("structure_factor between %d th SR HR MRI image: " %(i*10 + 1), structure_factor_between_sr_hr)
        
        # Here calculate PSNR of SR HR image pair
        psnr_between_sr_hr = calculatePSNR(entire_sr_image_data[i*10], entire_hr_image_data[i*10], max_I = 1.0)   # max_I的是表示图像点颜色的最大数值
        print("PSNR between %d th SR HR MRI image: " %(i*10 + 1), psnr_between_sr_hr)

        # Here calculate gradient map of SR HR image pair using Pytorch, same as how it is computed in network model
        gradient_map_entire_sr_image_data = calculateGradientMapUsingPytorch(entire_sr_image_data[i*10], Amplify_Small_Value_In_Gradient_Map)
        gradient_map_entire_hr_image_data = calculateGradientMapUsingPytorch(entire_hr_image_data[i*10], Amplify_Small_Value_In_Gradient_Map)

        # Here calculate k space data of SR HR image pair using Pytorch, same as how it is computed in network model
        k_space_entire_sr_image_data = calculateKSpaceUsingPytorch(entire_sr_image_data[i*10])
        k_space_entire_hr_image_data = calculateKSpaceUsingPytorch(entire_hr_image_data[i*10])

        # Plot SR HR image pair
        plt.subplot(1, 2, 1)
        plt.imshow(entire_hr_image_data[i*10], cmap='gray')
        plt.title("HR_test_image")
        plt.subplot(1, 2, 2)
        plt.title("SR_test_image")
        plt.imshow(entire_sr_image_data[i*10], cmap='gray')
        plt.suptitle("The %d th pair of HR vs SR, SSIM is %f, structure_factor is %f, contrast_factor is %f, luminance_factor is %f, PSNR is %f" %((i*10 + 1), ssim_between_sr_hr, structure_factor_between_sr_hr, contrast_factor_between_sr_hr, luminance_factor_between_sr_hr, psnr_between_sr_hr))
        plt.subplots_adjust()
        plt.show()

        # Plot gradient map of SR HR image pair
        plt.subplot(1, 2, 1)
        plt.imshow(gradient_map_entire_hr_image_data, cmap='gray')
        plt.title("Gradient_Map_HR_test_image")
        plt.subplot(1, 2, 2)
        plt.title("Gradient_Map_SR_test_image")
        plt.imshow(gradient_map_entire_sr_image_data, cmap='gray')
        plt.suptitle("The %d th pair of Gradient Map HR vs Gradient Map SR, SSIM is %f, structure_factor is %f, contrast_factor is %f, luminance_factor is %f, PSNR is %f" %((i*10 + 1), ssim_between_sr_hr, structure_factor_between_sr_hr, contrast_factor_between_sr_hr, luminance_factor_between_sr_hr, psnr_between_sr_hr))
        plt.subplots_adjust()
        plt.show()

        # Plot k space data of SR HR image pair
        plt.subplot(2, 2, 1)
        plt.imshow(k_space_entire_hr_image_data[:, :, 0], cmap='gray', vmin=0, vmax=32) # vmin/vmax stand for windowing size
        plt.title("K_Space_Data_HR_test_image_real_part")
        plt.subplot(2, 2, 2)
        plt.imshow(k_space_entire_hr_image_data[:, :, 1], cmap='gray', vmin=0, vmax=32)
        plt.title("K_Space_Data_HR_test_image_image_part")
        plt.subplot(2, 2, 3)
        plt.title("K_Space_Data_SR_test_image_real_part")
        plt.imshow(k_space_entire_sr_image_data[:, :, 0], cmap='gray', vmin=0, vmax=32)
        plt.subplot(2, 2, 4)
        plt.title("K_Space_Data_SR_test_image_image_part")
        plt.imshow(k_space_entire_sr_image_data[:, :, 1], cmap='gray', vmin=0, vmax=32)
        plt.suptitle("The %d th pair of K Space Data HR vs K Space Data SR, SSIM is %f, structure_factor is %f, contrast_factor is %f, luminance_factor is %f, PSNR is %f" %((i*10 + 1), ssim_between_sr_hr, structure_factor_between_sr_hr, contrast_factor_between_sr_hr, luminance_factor_between_sr_hr, psnr_between_sr_hr))
        plt.subplots_adjust()
        plt.show()


if __name__ == '__main__':
    """
    Setting
    """
    Amplify_Small_Value_In_Gradient_Map = True

    runSrImageCombinationAndEvaluation(Amplify_Small_Value_In_Gradient_Map)