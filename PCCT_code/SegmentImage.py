# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:56:15 2023

@author: u0139075
"""

import SimpleITK as sitk
import numpy as np
import os
import argparse
import datetime
import time
import sys
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider, Button

def WriteSegmentation(segm, output):
    output_folder = os.path.dirname(output)
    if output_folder == '':
        raise Warning('ERROR: you want to save the segmented image, but did not specify a specific directory. Hence, I do not know where the data should be stored!')
    else:
        sitk.WriteImage(segm, output)

def update(val):
    slice_nb = int(slice_slider.val)
    
    for nb in range(nb_plots):
        if not isinstance(arrays[nb], tuple):
            ax[nb].imshow(arrays[nb][slice_nb,:,:], cmap='gray')
                    
        else: #overlay
            ax[nb].imshow(arrays[nb][0][slice_nb,:,:], cmap='gray')
            ax[nb].imshow(arrays[nb][1][slice_nb,:,:], cmap=cmap2)


def PlotSegmentation(array_list, title_list):
    global nb_plots
    global ax
    global arrays
    global slice_slider
    global cmap2
    
    arrays = array_list
    cmap2 = matplotlib.colors.ListedColormap((['none','red'])) 
    
    nb_plots = len(arrays)
    fig, ax = plt.subplots(1, nb_plots)
    
    for nb in range(nb_plots):
        
        if not isinstance(arrays[nb], tuple):
            ax[nb].imshow(arrays[nb][0,:,:], cmap='gray')
                    
        else: #overlay
            ax[nb].imshow(arrays[nb][0][0,:,:], cmap='gray')
            ax[nb].imshow(arrays[nb][1][0,:,:], cmap=cmap2)
        
        ax[nb].set_title(title_list[nb])
    
    plt.subplots_adjust(bottom=0.25)   
    
    # Make a horizontal slider to control the slice nb.
    axslice = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=axslice,
        label='Slice Number',
        valmin= 0,
        valmax=np.shape(array_list[0])[0]-1,
        valinit=0,
    )

    # register the update function with each slider
    slice_slider.on_changed(update)
    plt.show()     
    
def DSC(img1, img2):
    arr1 = sitk.GetArrayViewFromImage(img1)
    arr2 = sitk.GetArrayViewFromImage(img2)
    vol1 = arr1 != 0
    vol2 = arr2 != 0
    overlap = 2*np.sum(vol1 * vol2)
    total = np.sum(vol1) + np.sum(vol2)
    return overlap/total

def CreateSphere(dx, dy, dz, radius):
    dd = int(np.round(radius/dz))
    dr = int(np.round(radius/dy))
    dc = int(np.round(radius/dx))
        
    sphere = np.zeros((2*dd+1, 2*dr+1, 2*dc+1), dtype=bool)
    for depth in range(2*dd+1):
        for row in range(2*dr+1):
            for col in range(2*dc+1):
                d = depth - dd
                r = row - dr
                c = col - dc
                
                if (d*dz)**2 + (r*dy)**2 + (c*dx)**2 <= radius**2:
                    sphere[depth, row, col] = 1
    return sphere

def CreateGaussianKernel(sphere):
    sd, sr, sc = np.shape(sphere)
    kernel_size = max(sd, sr, sc)
    sigma = 0.3 * ((kernel_size - 1)*0.5 - 1) + 0.8
    [d,r,c] = np.where(sphere == True)
    offd, offr, offc = np.floor(sd/2), np.floor(sr/2), np.floor(sc/2)
    dd = d-offd
    dr = r-offr
    dc = c-offc
    unscaled_kernel = np.zeros(np.shape(sphere))
    for vox in range(len(d)):
        di, ri, ci = dd[vox], dr[vox], dc[vox]
        x_vec = di**2 + ri**2 + ci**2
        gaus_val = np.exp(-x_vec/(2*sigma**2))
        unscaled_kernel[d[vox], r[vox], c[vox]] = gaus_val
    
    sum_val = np.sum(unscaled_kernel)
    kernel = unscaled_kernel / sum_val
    return kernel
        
    
    
def ComputeSubArrays(inputs, nb_depth_splits, nb_row_splits, nb_col_splits, operation):
    if operation == 'split':
        sub_arrs = np.array_split(inputs, nb_depth_splits, axis = 0)
        sub_arrs = [np.array_split(sub, nb_row_splits, axis = 1) for sub in sub_arrs]
        sub_arrs = [np.array_split(subsub, nb_col_splits, axis = 2) for sub in sub_arrs for subsub in sub]
        sub_arrs = [subsub for sub in sub_arrs for subsub in sub]
        return sub_arrs
    
    elif operation == 'restore':
        row_start_pos = np.arange(0, len(inputs), nb_col_splits)
        concat_row = [np.concatenate([chunk for chunk in inputs[i:i+nb_col_splits]], axis=2) for i in row_start_pos]
        
        col_start_pos = np.arange(0, len(concat_row), nb_row_splits)
        concat_col = [np.concatenate([chunk for chunk in concat_row[i:i+nb_row_splits]], axis=1) for i in col_start_pos]
        
        rest_arr = np.concatenate([matrix for matrix in concat_col], axis=0)
        
        return rest_arr
    
def AdaptiveProcessingPixel(bone, segm, sphere, adaptive_method, CHUNKED_DATA):
    (Dsph, Rsph, Csph) = np.shape(sphere)
    dd = (Dsph-1)/2
    dr = (Rsph-1)/2
    dc = (Csph-1)/2
    
    [voxD, voxR, voxC] = np.where(segm[:,:,:,1] == 0) # start from the labeled parts
    nb_vox = len(voxD)
    
    if CHUNKED_DATA:
        (Dbone, Rbone, Cbone) = np.shape(bone)       
        for i in range(nb_vox):
            di, ri, ci = voxD[i], voxR[i], voxC[i]
            
            SPHERE_FITS = (di - dd >= 0 and di + dd <= Dbone-1 and \
                           ri - dr >= 0 and ri + dr <= Rbone-1 and \
                           ci - dc >= 0 and ci + dc <= Cbone-1)
            
            if SPHERE_FITS:
                try:
                    tmp_part = bone[int(di-dd) : int(di+dd+1) , int(ri-dr) : int(ri+dr+1) , int(ci-dc) : int(ci+dc+1)]
                    tmp_part = tmp_part*sphere
                
                    if adaptive_method == 'mean':
                        local_thresh = np.mean(tmp_part[sphere!=0])
                    elif adaptive_method == 'median':
                        local_thresh = np.median(tmp_part[sphere!=0])
                    elif adaptive_method == 'mean_min_max':
                        local_thresh = 0.5*np.max(tmp_part[sphere!=0]) + 0.5*np.min(tmp_part[sphere!=0])
                    elif adaptive_method == 'gaussian':
                        local_thresh = np.sum(tmp_part) # the sphere is not boolean, but a discretized kernel
                    else:
                        raise ValueError(f'ERROR: invalid adaptive method selected! (input given is {adaptive_method})')
                    
                    if bone[di, ri, ci] <= local_thresh:
                        segm[di, ri, ci, 0] = 0
                    else:
                        segm[di, ri, ci, 0] = 255
                    
                    # set label segmentation to True, indicating that the voxel is computed
                    segm[di, ri, ci, 1] = 1    
                except Exception as e:
                    print(f'Error processing part at indices ({di-dd}:{di+dd+1}, {ri-dr}:{ri+dr+1}, {ci-dc}:{ci+dc+1}): {e}')
            
        
    if not CHUNKED_DATA:
        for i in tqdm(range(nb_vox)):
            di, ri, ci = voxD[i], voxR[i], voxC[i]
            try:
                tmp_part = bone[int(di-dd) : int(di+dd+1) , int(ri-dr) : int(ri+dr+1) , int(ci-dc) : int(ci+dc+1)]
                tmp_part = tmp_part*sphere
            
                if adaptive_method == 'mean':
                    local_thresh = np.mean(tmp_part[tmp_part!=0])
                elif adaptive_method == 'median':
                    local_thresh = np.median(tmp_part[tmp_part!=0])
                elif adaptive_method == 'mean_min_max':
                    local_thresh = 0.5*np.max(tmp_part[sphere!=0]) + 0.5*np.min(tmp_part[sphere!=0])
                elif adaptive_method == 'gaussian':
                    local_thresh = np.sum(tmp_part)
                else:
                    raise ValueError(f'ERROR: invalid adaptive method selected! (input given is {adaptive_method})')
                
                if bone[di, ri, ci] < local_thresh:
                    segm[di, ri, ci, 0] = 0
                else:
                    segm[di, ri, ci, 0] = 255
                
                # set label segmentation to True, indicating that the voxel is computed
                segm[di, ri, ci, 1] = 1  
                
            except Exception as e:
                print(f'Error processing part at indices ({di-dd}:{di+dd+1}, {ri-dr}:{ri+dr+1}, {ci-dc}:{ci+dc+1}): {e}')
        
    return segm

def Gaussian(y, mu, sd):
    return 1/(sd*np.sqrt(2*np.pi)) * np.exp(-(y-mu)**2 / (2*sd**2))

def CalculateLogLikelihood(intensities, mu1, sd1, prior1, mu2, sd2, prior2):

    P = prior1 * Gaussian(intensities, mu1, sd1) + prior2 * Gaussian(intensities, mu2, sd2)
    L = np.log(P)
    return np.sum(L) / len(intensities)

def CalculatePosteriorProbabilities(intensities, mu1, sd1, prior1, mu2, sd2, prior2):
    g1 = Gaussian(intensities, mu1, sd1)
    g2 = Gaussian(intensities, mu2, sd2)
    p_int = prior1 * g1 + prior2 * g2
    post1 = prior1 * g1 / p_int
    post2 = prior2 * g2 / p_int
    
    return post1, post2
    


def SegmentOtsu(parameter_file = 'default', bone = None, mask = None, reference = None, nb_bins=128):
    """
    Perform Otsu thresholding. Parameters can be given in a parameter file. If not given, defaults will be used.
    If bone AND mask are also passed as an argument, they will be ignored in the parameter file. 
    Should you explicitly enter them, they should be SimpleITK images (and thus no paths)
    
    The function returns the segmented image and the Otsu threshold
    """
    
    # default parameters
    bone_file = ''
    mask_file = ''
    output_folder = ''
    WRITE_LOG = 0
    WRITE_SEGMENTATION = 0
    PLOT_SEGMENTATION = 0
    reference_file = ''
    
    # reading the parameter file
    if parameter_file != 'default':
        try:
            with open(parameter_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith('#') and '=' in line:
                        param, value = line.split('=')
                        param = param.strip()
                        value = value.strip()
                        if param == 'bone_file':
                            bone_file = value
                        elif param == 'mask_file':
                            mask_file = value
                        elif param == 'output_folder':
                            output_folder = value
                        elif param == 'nb_bins':
                            nb_bins = int(value)
                        elif param == 'WRITE_LOG':
                            WRITE_LOG = bool(int(value))
                        elif param == 'WRITE_SEGMENTATION':
                            WRITE_SEGMENTATION = bool(int(value))
                        elif param == 'PLOT_SEGMENTATION':
                            PLOT_SEGMENTATION = bool(int(value))
                        elif param == 'reference_file':
                            reference_file = value
                        else:
                            print(f'parameter input {param} is not a valid input!')
        except:
            raise NameError(f'{parameter_file} is not a valid file.')
    
    if bone is None or mask is None:
        filename = os.path.basename(bone_file).split('.')[0]
        print('Reading in image...')
        bone = sitk.ReadImage(bone_file)
        print('Reading in mask...')
        mask = sitk.ReadImage(mask_file)
    
    if mask.GetPixelID() != 1:
        mask = sitk.Cast(mask, sitk.sitkUInt8)

    if os.path.isfile(reference_file):
        print('Reading in reference image...')
        reference = sitk.ReadImage(reference_file)
    
    # the actual filter
    print('Performing segmentation')
    filt = sitk.OtsuThresholdImageFilter()
    filt.SetNumberOfHistogramBins(nb_bins)
    filt.SetMaskValue(255)
    filt.SetInsideValue(0)
    filt.SetOutsideValue(255)
    segm = filt.Execute(bone, mask)
    thresh = filt.GetThreshold()

    if WRITE_SEGMENTATION:
        output_file = os.path.join(output_folder, filename+'_segmOtsu.mha' )
        WriteSegmentation(segm, output_file)
    
    if WRITE_LOG:
        if reference is None:
            print('DSC cannot be calculated as you did not give a registered reference image in any way.')
            dsc = 'unknown'
        else:
            dsc = DSC(segm, reference)
        
        if output_folder == '':
            print('ERROR: you want to write a log-file, but did not specify a output directory. Hence, I do not know where the data should be stored!')
        else:
            log_file = os.path.join(output_folder, filename + '_Otsu_log.txt')
            log = open(log_file, 'a')
            
            log.write('OTSU SEGMENTATION LOG FILE \n')
            now = datetime.datetime.now()
            log.write(f'{now}\n\n')
            log.write('-- PARAMETER FILE --\n\n')
            with open(parameter_file, 'r') as f:
                for line in f:
                    log.write(line)
            log.write('\n\n')
            log.write('--- END OF PARAMETER FILE ---\n\n')
            log.write(f'Otsu threshold: {thresh}\n')
            log.write(f'DSC: {dsc}')
            log.close()
    
    if PLOT_SEGMENTATION:
        bone_arr = sitk.GetArrayFromImage(bone)
        segm_arr = sitk.GetArrayFromImage(segm)
        PlotSegmentation([bone_arr, segm_arr], ['image', f'Otsu segmentation - threshold {thresh}'])
        
    return segm, thresh


def SegmentGMM(parameter_file = 'default', ADAPTIVE = True, bone = None, mask = None, reference = None):
    """
    Script to segment the image with a Gaussian Mixture Model.
    The script labels voxels inside the mask as bone or soft tissue. 
    
    An uncertain region is set for likelihood bone lower than 4*likelihood marrow.
    If the boolean ADAPTIVE is set True, in the uncertain region, adaptive thresholding is used.
    """
    
    # default parameters
    bone_file =''
    mask_file =''
    output_folder=''
    reference_file=''
    CALIBRATE_MODEL = True  
    REDUCED_SAMPLING = False 
    adaptive_method = 'mean_min_max'  
    radius = 0.1 #mm  
    certain_bone_fact = 2
    certain_soft_fact = 2      
    nb_bins = 128           
    max_iterations = 20            
    eps = 1e-4
    nb_samples = 700000
    
    mean_bone = None
    mean_soft = None
    sd_bone = None
    sd_soft = None
    prior_bone = None
    prior_soft = None
    
    WRITE_SEGMENTATION = False
    WRITE_LOG = False
    PLOT_SEGMENTATIONS = False

    # reading the parameter file
    if parameter_file != 'default':
        try:
            with open(parameter_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith('#') and '=' in line:
                        param, value = line.split('=')
                        param = param.strip()
                        value = value.strip()
                        if param == 'bone_file':
                            bone_file = value
                        elif param == 'mask_file':
                            mask_file = value
                        elif param == 'output_folder':
                            output_folder = value
                        elif param == 'reference_file':
                            reference_file = value
                        elif param == 'CALIBRATE_MODEL':
                            CALIBRATE_MODEL = bool(int(value))
                        elif param == 'REDUCED_SAMPLING':
                            REDUCED_SAMPLING = bool(int(value))
                        elif param == 'ADAPTIVE':
                            ADAPTIVE = bool(int(value))
                        elif param == 'ADAPTIVE_METHOD':
                            adaptive_method = value
                        elif param == 'radius':
                            radius = float(value)
                        elif param == 'certain_bone_fact':
                            certain_bone_fact = float(value)
                        elif param == 'certain_soft_fact':
                            certain_soft_fact = float(value)
                        elif param == 'nb_bins':
                            nb_bins = int(value)
                        elif param == 'max_iterations':
                            max_iterations = int(value)
                        elif param == 'epsilon':
                            eps = float(value)
                        elif param == 'nb_samples':
                            nb_samples = int(value)
                        elif param == 'mean_bone':
                            mean_bone = float(value)
                        elif param == 'mean_soft':
                            mean_soft = float(value)
                        elif param == 'sd_bone':
                            sd_bone = float(value)
                        elif param == 'sd_soft':
                            sd_soft = float(value)
                        elif param == 'prior_bone':
                            prior_bone = float(value)
                        elif param == 'prior_soft':
                            prior_soft = float(value)
                        elif param == 'WRITE_SEGMENTATION':
                            WRITE_SEGMENTATION = bool(int(value))
                        elif param == 'WRITE_LOG':
                            WRITE_LOG = bool(int(value))
                        elif param == 'PLOT_SEGMENTATION':
                            PLOT_SEGMENTATION = bool(int(value))
                        else:
                            print(f'parameter input {param} is not a valid input!')
        except:
            raise NameError(f'{parameter_file} is not a valid file.')
     
    if bone is None or mask is None:
        filename = os.path.basename(bone_file).split('.')[0]
        print('Reading in image...')
        bone = sitk.ReadImage(bone_file)
        print('Reading in mask...')
        mask = sitk.ReadImage(mask_file)
        if output_folder == '':
            output_folder = os.path.dirname(bone_file)
            print(f"WARNING: no output directory for given. Using {output_folder}")
    
    if os.path.isfile(reference_file):
        print('Reading in reference image...')
        reference = sitk.ReadImage(reference_file)
    
    bone_arr = sitk.GetArrayViewFromImage(bone)
    mask_arr = sitk.GetArrayViewFromImage(mask)
    
    if WRITE_LOG:
        log_string = os.path.join(output_folder, filename + '_GMM_log.txt')
        log = open(log_string, 'a')
        log.write('GAUSSIAN MIXTURE MODEL SEGMENTATION LOG FILE \n')
        now = datetime.datetime.now()
        log.write(f'{now}\n\n')
        log.write('-- PARAMETER FILE --\n\n')
        with open(parameter_file, 'r') as f:
            for line in f:
                log.write(line)
        log.write('\n\n')
        log.write('--- END OF PARAMETER FILE ---\n\n')
        
    # calibrate model
    if CALIBRATE_MODEL:
        if WRITE_LOG: log.write('Calibrating model...\n')
        param_string = os.path.join(output_folder, filename + '_GMM_parameters.txt')
        GMMparams = open(param_string, 'w')
        
        (fov_z, fov_r, fov_c) = np.where(mask_arr != 0)
        nb_vox = len(fov_z)
        
        if REDUCED_SAMPLING: string = f'Estimating Gaussian Mixture Model based on {nb_samples} / {nb_vox} random voxels per iteration.'
        else: string = f'Calculating Gaussian Mixture Model based on all {nb_vox} voxels per iteration.'
        print(string)
        if WRITE_LOG: log.write(string+'\n')
        
        # -- 1) IMAGE HISTOGRAM
        fov_values = bone_arr[fov_z, fov_r, fov_c]
        
        (hist, edges) = np.histogram(fov_values, bins = nb_bins, density = True)
        bin_centers = np.asarray([0.5*(edges[i]+edges[i+1]) for i in range(len(edges)-1)])
        bin_width = edges[1] - edges[0]
        
        # -- 2) INITIAL SEGMENTATION
        otsu, t_otsu = SegmentOtsu(bone = bone, mask = mask, nb_bins = nb_bins)
        otsu_arr = sitk.GetArrayViewFromImage(otsu)
        # median_val = np.median(bone_arr[mask_arr != 0])
        # median_val /= 2
        # print(f'Half the median value as model intialization: {median_val}')
        # filt = sitk.BinaryThresholdImageFilter()
        # filt.SetLowerThreshold(median_val)
        # filt.SetUpperThreshold(32767) #assuming the 16int datatype
        # filt.SetInsideValue(255)
        # filt.SetOutsideValue(0)
        # otsu = filt.Execute(bone)
        # otsu_arr = sitk.GetArrayViewFromImage(otsu)

        
        # -- 3) INITIAL PARAMETRIZATION
        string = 'Model initialization...'
        print(string)
        if WRITE_LOG: log.write(string + '\n')
        start = time.time()
        
        (bone_z, bone_r, bone_c) = np.where(otsu_arr != 0)
        softMatrix = (otsu_arr == 0) * (mask_arr != 0) # Otsu is zero AND in FOV
        (soft_z, soft_r, soft_c) = np.where(softMatrix == True)
        
        bone_vals = bone_arr[bone_z, bone_r, bone_c]
        soft_vals = bone_arr[soft_z, soft_r, soft_c]
        
        mean_bone = np.mean(bone_vals)
        sd_bone = np.std(bone_vals)
        prior_bone = len(bone_vals) / nb_vox
        
        mean_soft = np.mean(soft_vals)
        sd_soft = np.std(soft_vals)
        prior_soft = len(soft_vals) / nb_vox
        
        gauss_bone = prior_bone * Gaussian(bin_centers, mean_bone, sd_bone)
        gauss_soft = prior_soft * Gaussian(bin_centers, mean_soft, sd_soft)
        
        likelihood = CalculateLogLikelihood(fov_values, mean_bone, sd_bone, prior_bone, \
                                                                mean_soft, sd_soft, prior_soft) 
        
        end = time.time()
        string = f'Model initialization done ({round(end-start,2)} seconds).\nInitial log-likelihood: {round(likelihood,5)}'
        print(string)
        print('GMM initialization: parameters')
        print('---------------')
        print("{:<10} | {d}".format("mean_bone",d=round(mean_bone,2)))
        print("{:<10} | {d}".format("sd_bone",d=round(sd_bone,2)))
        print("{:<10} | {d}".format("prior_bone",d=round(prior_bone,2)))
        print("{:<10} | {d}".format("mean_soft",d=round(mean_soft,2)))
        print("{:<10} | {d}".format("sd_soft",d=round(sd_soft,2)))
        print("{:<10} | {d}".format("prior_soft",d=round(prior_soft,2)))
        # print("{:<10} | {d}".format("tOtsu",d=round(t_otsu,2)))
        if WRITE_LOG: 
            log.write(string + '\n\n')
            log.write('GMM initialization: parameters\n')
            log.write('------------------------------\n')
            log.write("{:<10} | {d}\n".format("mean_bone",d=round(mean_bone,2)))
            log.write("{:<10} | {d}\n".format("sd_bone",d=round(sd_bone,2)))
            log.write("{:<10} | {d}\n".format("prior_bone",d=round(prior_bone,2)))
            log.write("{:<10} | {d}\n".format("mean_soft",d=round(mean_soft,2)))
            log.write("{:<10} | {d}\n".format("sd_soft",d=round(sd_soft,2)))
            log.write("{:<10} | {d}\n".format("prior_soft",d=round(prior_soft,2)))
            # log.write("{:<10} | {d}\n".format("tOtsu",d=round(t_otsu,2)))
        
        # -- 4) ITERATION
        iteration = 0
        CONVERGED = False
        
        string = f'--- Start iterative optimization with epsilon = {eps} and max {max_iterations} iterations'
        print(string)
        if WRITE_LOG: log.write(string + '\n')
        
        start = time.time()
        while not CONVERGED and iteration < max_iterations:
            # step 1: posteriori probabilities for all voxels
            if REDUCED_SAMPLING:
                voxel_list = np.random.randint(0, nb_vox, nb_samples)
                fov_set = fov_values[voxel_list]

            else:
                fov_set = fov_values
            
            post_bone, post_soft = CalculatePosteriorProbabilities(fov_set,\
                                                                   mean_bone, sd_bone, prior_bone,\
                                                                   mean_soft, sd_soft, prior_soft)
            # step 2: update model parameters
            mean_bone = np.sum(post_bone * fov_set) / np.sum(post_bone)
            sd_bone = np.sqrt(np.sum(post_bone * (fov_set - mean_bone)**2) / np.sum(post_bone))
            prior_bone = np.sum(post_bone) / len(fov_set)
            
            mean_soft = np.sum(post_soft * fov_set) / np.sum(post_soft)
            sd_soft = np.sqrt(np.sum(post_soft * (fov_set - mean_soft)**2) / np.sum(post_soft))
            prior_soft = np.sum(post_soft) / len(fov_set)
            
            # step 3: calculate log likelihood
            likelihood_old = likelihood
            likelihood = CalculateLogLikelihood(fov_values, mean_bone, sd_bone, prior_bone, \
                                                                    mean_soft, sd_soft, prior_soft) 
            iteration += 1
            CONVERGED = abs(likelihood - likelihood_old) < eps
            string = f'Iteration {iteration}: Diff = {round(likelihood-likelihood_old, 5)} (L = {round(likelihood, 5)})'
            print(string)
            if WRITE_LOG: log.write(string + '\n')
        
        end = time.time()
        if CONVERGED:
            string = f'Optimal Gaussian mixture model found. (Iteration took {round(end-start,2)}seconds)'
            print(string)
            if WRITE_LOG: log.write(string + '\n')
        else:
            string = f'Maximal numbers of iterations reached. (Iteration took {round(end-start,2)}seconds)'
            print(string)
            if WRITE_LOG: log.write(string + '\n\n')
        
        # theoretical threshold: gauss_bone = gauss_soft
        a = 1/sd_bone**2 - 1/sd_soft**2
        b = 2*mean_soft/sd_soft**2 - 2*mean_bone/sd_bone**2
        c = mean_bone**2/sd_bone**2 - mean_soft**2/sd_soft**2 - 2*np.log(prior_bone*sd_soft/(prior_soft*sd_bone))
        D = b**2-4*a*c
        t_GMM = np.array([(-b+np.sqrt(D))/(2*a) , (-b-np.sqrt(D))/(2*a)])
        response = Gaussian(t_GMM, mean_bone, sd_bone) # two thresholds (quadratic equation), typically one in the far ends of the tail, so look for highest response
        [t_GMM] = t_GMM[np.where(response == max(response))[0]]
        
        print('GMM parameters')
        print('---------------')
        print("{:<10} | {d}".format("mean_bone",d=round(mean_bone,2)))
        print("{:<10} | {d}".format("sd_bone",d=round(sd_bone,2)))
        print("{:<10} | {d}".format("prior_bone",d=round(prior_bone,2)))
        print("{:<10} | {d}".format("mean_soft",d=round(mean_soft,2)))
        print("{:<10} | {d}".format("sd_soft",d=round(sd_soft,2)))
        print("{:<10} | {d}".format("prior_soft",d=round(prior_soft,2)))
        print("{:<10} | {d}".format("threshold",d=round(t_GMM,2)))
        
        if WRITE_LOG:
            log.write('GMM parameters\n')
            log.write('---------------\n')
            log.write("{:<10} | {d}\n".format("mean_bone",d=round(mean_bone,2)))
            log.write("{:<10} | {d}\n".format("sd_bone",d=round(sd_bone,2)))
            log.write("{:<10} | {d}\n".format("prior_bone",d=round(prior_bone,2)))
            log.write("{:<10} | {d}\n".format("mean_soft",d=round(mean_soft,2)))
            log.write("{:<10} | {d}\n".format("sd_soft",d=round(sd_soft,2)))
            log.write("{:<10} | {d}\n".format("prior_soft",d=round(prior_soft,2)))
            log.write("{:<10} | {d}\n".format("threshold",d=round(t_GMM,2)))
        
        GMMparams.write(f'mean_bone={mean_bone}\n')
        GMMparams.write(f'mean_soft={mean_soft}\n')
        GMMparams.write(f'sd_bone={sd_bone}\n')
        GMMparams.write(f'sd_soft={sd_soft}\n')
        GMMparams.write(f'prior_bone={prior_bone}\n')
        GMMparams.write(f'prior_soft={prior_soft}\n')
        GMMparams.write(f't_GMM={t_GMM}')
        GMMparams.close()
    
    
        gauss_bone_conv = prior_bone * Gaussian(bin_centers, mean_bone, sd_bone)
        gauss_soft_conv = prior_soft * Gaussian(bin_centers, mean_soft, sd_soft)
        
        if PLOT_SEGMENTATION:
            f, ax = plt.subplots(2)
            ax[0].plot(bin_centers, hist, 'k', label = 'normalized histogram')
            ax[0].plot(bin_centers, gauss_bone, 'r--', label = 'initial bone model')
            ax[0].plot(bin_centers, gauss_soft, 'g--', label = 'initial soft model')
            ax[0].plot(bin_centers, gauss_bone + gauss_soft, 'b', label = 'intitial model histogram')
            ax[0].legend()
            
            ax[1].plot(bin_centers, hist, 'k', label = 'normalized histogram')
            ax[1].plot(bin_centers, gauss_bone_conv, 'r--', label = 'final bone model')
            ax[1].plot(bin_centers, gauss_soft_conv, 'g--', label = 'final soft model')
            ax[1].plot(bin_centers, gauss_bone_conv + gauss_soft_conv, 'b', label = 'final model histogram')
            ax[1].legend()
        
    else:
        if mean_bone is None or mean_soft is None or sd_bone is None or sd_soft is None \
            or prior_bone is None or prior_soft is None:
                try:
                    GMMstring = os.path.join(output_folder, filename+'_GMM_parameters.txt')
                    with open(GMMstring, 'r') as f:
                        print('Reading in model parameters...')
                        for line in f:
                            line = line.strip()
                            if not line.startswith('#') and '=' in line:
                                param, value = line.split('=')
                                param = param.strip()
                                value = value.strip()
                                if param == 'mean_bone':
                                    mean_bone = float(value)
                                elif param == 'mean_soft':
                                    mean_soft = float(value)
                                elif param == 'sd_bone':
                                    sd_bone = float(value)
                                elif param == 'sd_soft':
                                    sd_soft = float(value)
                                elif param == 'prior_bone':
                                    prior_bone = float(value)
                                elif param == 'prior_soft':
                                    prior_soft = float(value)
                                elif param == 't_GMM':
                                    t_GMM = float(value)
                except Exception as e:
                    string = f'Not all model parameters are given as input and no file is found that contains them: {e}'
                    if WRITE_LOG:
                        log.write(string)
                        log.close()
                    raise NameError(string)
    
    # at this point, either model is calibrated or parameters are read in
    string = 'Segmenting image with GMM threshold...'
    print(string)
    if WRITE_LOG: log.write(string + '\n')
                       
    filt = sitk.BinaryThresholdImageFilter()
    filt.SetLowerThreshold(t_GMM)
    filt.SetUpperThreshold(32767) #assuming the 16int datatype
    filt.SetInsideValue(255)
    filt.SetOutsideValue(0)
    segm_GMM = filt.Execute(bone)
    
    segm_arr = sitk.GetArrayFromImage(segm_GMM)
    # masking the segmentation
    # MAKE SURE THE MASK IS COMPLETE!!! (TOO LARGE IS NOT A PROBLEM, TOO SMALL AND YOU'LL LOSE BONE VOXELS)
    segm_arr *= (mask_arr != 0)  
    orig_arr = np.copy(segm_arr)
    if ADAPTIVE:
        
        # certainly bone: Pbone >= certain_bone_factor * Psoft
        a = 1/sd_soft**2 - 1/sd_bone**2
        b = 2*mean_bone/sd_bone**2 - 2*mean_soft/sd_soft**2
        c = mean_soft**2/sd_soft**2 - mean_bone**2/sd_bone**2 + 2*np.log(prior_bone*sd_soft/(certain_bone_fact*prior_soft*sd_bone))
        D = b**2-4*a*c
        t = np.array([(-b+np.sqrt(D))/(2*a) , (-b-np.sqrt(D))/(2*a)])
        response = Gaussian(t, mean_bone, sd_bone)
        [t_high] = t[np.where(response == max(response))[0]]
        
        #soft prob higher than bone prob
        c = mean_soft**2/sd_soft**2 - mean_bone**2/sd_bone**2 + 2*np.log(certain_soft_fact*prior_bone*sd_soft/(prior_soft*sd_bone))
        D = b**2-4*a*c
        t = np.array([(-b+np.sqrt(D))/(2*a) , (-b-np.sqrt(D))/(2*a)])
        response = Gaussian(t, mean_bone, sd_bone)
        [t_low] = t[np.where(response == max(response))[0]]

        string = f'Uncertainty thresholds: t_low = {t_low} and t_high = {t_high}'
        print(string)
        if WRITE_LOG: log.write(string+'\n')
        
        certain_labels = (bone_arr <= t_low) + (bone_arr >= t_high)
        certain_labels += (mask_arr == 0)
        
        nb_uncertain = len(np.where(certain_labels == 0)[0])
        string = f'{nb_uncertain} voxels are classfied as uncertain. Starting adaptive thresholding with the {adaptive_method} method.'
        print(string)
        if WRITE_LOG: log.write(string+'\n')
        
        # extend segm_arr to a fourth dimension with labels indicating if value is treated
        # (True indicates that no further investigation is required)
        segm_arr = np.stack([segm_arr, certain_labels], axis=-1)
        
        # create sphere for adaptive local region
        (dx, dy, dz) = bone.GetSpacing()
        sphere = CreateSphere(dx, dy, dz, radius)
        if adaptive_method == 'gaussian':
            sphere = CreateGaussianKernel(sphere)
        
        print('Dividing the arrays into chunks for multiprocessing...')
        # divide arrays into smaller pieces to enable multiprocessing
        sub_bone_arr = ComputeSubArrays(bone_arr, 2, 8, 8, 'split')
        sub_segm_arr = ComputeSubArrays(segm_arr, 2, 8, 8, 'split')
        
        nb_chunks = len(sub_bone_arr)
        
        print(f'Initiating the multiprocessing. Calculating local threhsolds for {nb_chunks} chunks in parallel...')
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(AdaptiveProcessingPixel,\
                                             sub_bone_arr, sub_segm_arr, [sphere]*nb_chunks,\
                                             [adaptive_method]*nb_chunks, [True]*nb_chunks ), total=nb_chunks))
            
        # put chunks back to original array
        print('Restoring the chuncks into full array')
        segm_arr = ComputeSubArrays(results, 2, 8, 8, 'restore')
        
        # segment border elements
        nb_remaining = len(np.where(segm_arr[:,:,:,1] == 0)[0])  # count False labels, indicating not treated
        print(f'Calculating the local threshold for {nb_remaining} border elements...')
        segm_arr = AdaptiveProcessingPixel(bone_arr, segm_arr, sphere, adaptive_method, CHUNKED_DATA=False)
        segm_arr = segm_arr[:,:,:,0]
        end = time.time()
        string = f'Adaptive thresholding done. (Computed {round(end-start,2)} seconds)'
        print(string)
        if WRITE_LOG: log.write(string + '\n')
        
        if PLOT_SEGMENTATION:
            uncertain = (certain_labels == 0)
            PlotSegmentation([bone_arr, orig_arr, (orig_arr,uncertain), segm_arr], ['image', f'GMM - threhs {round(t_GMM)}', 'Uncertain voxels', 'GMM-adaptive'])
            
    
    segm_GMM = sitk.GetImageFromArray(segm_arr)
    segm_GMM.SetSpacing(bone.GetSpacing())
    segm_GMM.SetOrigin(bone.GetOrigin())
    
    if WRITE_SEGMENTATION:
        if ADAPTIVE: output = os.path.join(output_folder, filename +'_segmGMMadaptive.mha')
        else: output = os.path.join(output_folder, filename +'_segmGMM.mha')
        if WRITE_LOG: log.write('Writing image...\n')
        WriteSegmentation(segm_GMM, output)
    
    if WRITE_LOG: 
        if reference is None:
            dsc = 'unknown'
        else:
            dsc = DSC(segm_GMM, reference)
        log.write(f'DSC = {dsc}\n\n')        
        log.close()
    
    if PLOT_SEGMENTATION and not ADAPTIVE:
        PlotSegmentation([bone_arr, segm_arr], ['image', f'GMM segmentation - threshold {t_GMM}'])
        
    return segm_GMM, t_GMM


def SegmentAdaptive(parameter_file = 'default', bone = None, mask = None, reference = None, MULTIPROCESSING = True):
    """
    Perform Otsu thresholding. Parameters can be given in a parameter file. If not given, defaults will be used.
    If bone AND mask are also passed as an argument, they will be ignored in the parameter file. 
    
    Should you explicitly enter them, they should be SimpleITK images (and thus no paths).
    To increase computation time, a rough first binarization is done with a threshold that is 50% of the Otsu threshold.
    Adaptive segmentation is done in a spherical region, expressed in physical size (to account for non-isotropic voxels).
    """
    
    # default parameters
    bone_file = ''
    mask_file = ''
    reference_file = ''
    output_folder = ''
    radius = 0.1 # mm
    adaptive_method = 'mean'
    init_threshold = 'low'
    WRITE_LOG = 0
    WRITE_SEGMENTATION = 0
    PLOT_SEGMENTATION = 0
    
    # reading the parameter file
    if parameter_file != 'default':
        try:
            with open(parameter_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith('#') and '=' in line:
                        param, value = line.split('=')
                        param = param.strip()
                        value = value.strip()
                        if param == 'bone_file':
                            bone_file = value
                        elif param == 'mask_file':
                            mask_file = value
                        elif param == 'output_folder':
                            output_folder = value
                        elif param == 'reference_file':
                            reference_file = value
                        elif param == 'radius':
                            radius = float(value)
                        elif param == 'ADAPTIVE_METHOD':
                            adaptive_method = value
                        elif param == 'INIT_THRESHOLD':
                            init_threshold = value
                        elif param == 'MULTIPROCESSING':
                            MULTIPROCESSING = bool(int(value))
                        elif param == 'WRITE_SEGMENTATION':
                            WRITE_SEGMENTATION = bool(int(value))
                        elif param == 'WRITE_LOG':
                            WRITE_LOG = bool(int(value))
                        elif param == 'PLOT_SEGMENTATION':
                            PLOT_SEGMENTATION = bool(int(value))
                        else:
                            print(f'parameter input {param} is not a valid input!')
        except:
            raise NameError(f'{parameter_file} is not a valid file.')
                        
    if bone is None or mask is None:
        filename = os.path.basename(bone_file).split('.')[0]
        print('Reading in image...')
        bone = sitk.ReadImage(bone_file)
        print('Reading in mask...')
        mask = sitk.ReadImage(mask_file)
        if output_folder == '':
            output_folder = os.path.dirname(bone_file)
            print(f"WARNING: no output directory for given. Using {output_folder}")
    
    if os.path.isfile(reference_file):
        print('Reading in reference image...')
        reference = sitk.ReadImage(reference_file)
        
    if init_threshold not in ('low', 'high'):
        print(f"WARNING: invalid input for initial segmentation threshold (enter with low or high). Using < low >")
        init_threshold = 'low'
        
    # Rough segmentation
    (_, thresh) = SegmentOtsu(bone = bone, mask = mask)
    thresh = int(thresh*1.2) if init_threshold == 'high' else int(thresh/2)
    filt = sitk.BinaryThresholdImageFilter()
    filt.SetLowerThreshold(thresh)
    filt.SetUpperThreshold(32767) #assuming the 16int datatype
    filt.SetInsideValue(255)
    filt.SetOutsideValue(0)
    segm_init = filt.Execute(bone)

    #create a sphere with 1s 
    [dx, dy, dz] = bone.GetSpacing()
    sphere = CreateSphere(dx, dy, dz, radius)
    print(f'initial threshold: {thresh}')
    print(f'sphere shape: {sphere.shape}')
    if adaptive_method == 'gaussian':
        sphere = CreateGaussianKernel(sphere)
       
    # switch to array implementation
    bone_arr = sitk.GetArrayViewFromImage(bone)
    mask_arr = sitk.GetArrayViewFromImage(mask)
    segm_arr = sitk.GetArrayFromImage(segm_init)
    
    # MAKE SURE THE MASK IS COMPLETE!!! (TOO LARGE IS NOT A PROBLEM, TOO SMALL AND YOU'LL LOSE BONE VOXELS)

    # extend segm_arr to a fourth dimension with labels indicating if value is treated
    # (True indicates that no further investigation is required)
    if init_threshold == 'high':
        labels = ((segm_arr > 0) + (mask_arr == 0)).astype(bool)
    else: #init_threshold == low
        labels = ((segm_arr == 0) + (mask_arr == 0)).astype(bool)
    segm_arr = np.stack([segm_arr, labels], axis=-1)
    
    if MULTIPROCESSING:
        print('Dividing the arrays into chunks for multiprocessing...')
        # divide arrays into smaller pieces to enable multiprocessing
        sub_bone_arr = ComputeSubArrays(bone_arr, 2, 8, 8, 'split')
        sub_segm_arr = ComputeSubArrays(segm_arr, 2, 8, 8, 'split')
        
        nb_chunks = len(sub_bone_arr)
        
        print(f'Initiating the multiprocessing. Calculating local threhsolds for {nb_chunks} chunks in parallel...')
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(AdaptiveProcessingPixel,\
                                             sub_bone_arr, sub_segm_arr, [sphere]*nb_chunks,\
                                             [adaptive_method]*nb_chunks, [True]*nb_chunks ), total=nb_chunks))
            
        # put chunks back to original array
        print(f'Restoring the chuncks into full array')
        segm_arr = ComputeSubArrays(results, 2, 8, 8, 'restore')
        
        # segment border elements
        nb_remaining = len(np.where(segm_arr[:,:,:,1] == 0)[0])  # count False labels, indicating not treated
        print(f'Calculating the local threshold for {nb_remaining} border elements...')
        segm_arr = AdaptiveProcessingPixel(bone_arr, segm_arr, sphere, adaptive_method, CHUNKED_DATA=False)
        segm_arr = segm_arr[:,:,:,0]
    
    if not MULTIPROCESSING:
        nb_remaining = len(np.where(segm_arr[:,:,:,1] == 0)[0])
        start_time = time.time()
        print(f'Calculating the local threshold for {nb_remaining} elements...') # counting False labels, indicating not treated
        segm_arr = AdaptiveProcessingPixel(bone_arr, segm_arr, sphere, adaptive_method, CHUNKED_DATA=False)
        segm_arr = segm_arr[:,:,:,0]
        
    stop_time = time.time()
    duration = round(stop_time-start_time,2)
    print(f'Processing took {duration} seconds')
    
    segm_arr = segm_arr * (mask_arr != 0)
    segm = sitk.GetImageFromArray(segm_arr)
    segm.SetOrigin(segm_init.GetOrigin())
    segm.SetSpacing(segm_init.GetSpacing())
    
    
    if WRITE_SEGMENTATION:
        output = os.path.join(output_folder, filename + '_segmAdaptive.mha')
        WriteSegmentation(segm, output)
    
    if WRITE_LOG:
        if reference is None:
            print('DSC cannot be calculated as you did not give a registered reference image in any way.')
            dsc = 'unknown'
        else:
            dsc = DSC(segm, reference)
 
        log_file = os.path.join(output_folder, filename + '_Adaptive_log.txt')
        log = open(log_file, 'a')
        
        log.write('ADAPTIVE SEGMENTATION LOG FILE \n')
        now = datetime.datetime.now()
        log.write(f'{now}\n\n')
        log.write('-- PARAMETER FILE --\n\n')
        with open(parameter_file, 'r') as f:
            for line in f:
                log.write(line)
        log.write('\n\n')
        log.write('--- END OF PARAMETER FILE ---\n\n')
        log.write(f'Segmentation took {duration} seconds\n')
        log.write(f'DSC: {dsc}')
        log.close()
        
    if PLOT_SEGMENTATION:
        PlotSegmentation([bone_arr, segm_arr], ['image', f'Adaptive segmentation - {adaptive_method} method'])
    
    return segm
    

def main():
    parser = argparse.ArgumentParser(description = \
                                     'Script to segment SimpleITK images. Currently implemented methods:\n' \
                                     '-Otsu --> single threshold\n' \
                                     '-Adaptive --> local threshold\n' \
                                     '-Gaussian Mixture Model --> single threshold\n' \
                                     '-Gaussian Mixture Model with adaptive in uncertainty\n\n'\
                                     'Because of the large amount of parameters, varying with each segmentation scheme, you should input them in a seperate txt-file.\n'\
                                     'Run >>> SegmentImage.py -list for an overview of all parameters',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-list', help='Give an overview of parameters for each segmentation algorithm, and exit the program', action='store_true', default=False)
    parser.add_argument('-method', help='Segmentation algorithm. Specify by keyword otsu , adaptive, or GMM')
    parser.add_argument('-param', help='Path to parameter txt file')
    
    args = parser.parse_args()
    
    if args.list:
        print()
        print('--- %% OTSU %% ---')
        print()
        print("{:<25} | {d}".format("Parameter",d="Description"))
        print("--------------------------------------------------------------------")
        print("{:<25} | {d}".format("bone_file",d="Link to gray image volume to be segmented"))
        print("{:<25} | {d}".format("mask_file",d="Link to black/white mask file. Standard input is 8bit Uint"))
        print("{:<25} | {d}".format("output_folder",d="Folder to store segmentations and log files"))
        print("{:<25} | {d}".format("nb_bins",d="Number of bins in the histogram for the Otsu method")) 
        print("{:<25} | {d}".format("WRITE_LOG",d="1/0. Write a log file, including the used parameter file, otsu threshold and the Dice Similarity of the scan to a reference image"))
        print("{:<25} | {d}".format("WRITE_SEGMENTATION",d="1/0. Write a the segmented image as an mha file. The file is named as the input image, with `segmOtsu` appended"))
        print("{:<25} | {d}".format("PLOT_SEGMENTATION",d="1/0. Plot original and segmented image in a slicer window."))
        print("{:<25} | {d}".format("reference_file",d="Link to registered (i.e. transformed and resampled) reference segmentation file. (If DSC is to be calculated)."))
        print()
        print()
        print('--- %% ADAPTIVE %% ---')
        print()
        print("{:<25} | {d}".format("Parameter",d="Description"))
        print("--------------------------------------------------------------------")
        print("{:<25} | {d}".format("bone_file",d="Link to gray image volume to be segmented"))
        print("{:<25} | {d}".format("mask_file",d="Link to black/white mask file. Standard input is 8bit Uint"))
        print("{:<25} | {d}".format("output_folder",d="Folder to store segmentations and log files"))
        print("{:<25} | {d}".format("radius",d="Radius of sphere to calculate local threshold, expressed in milimeters")) 
        print("{:<25} | {d}".format("ADAPTIVE_METHOD",d="gaussian, mean, median or mean_min_max")) 
        print("{:<25} | {d}".format("INIT_THRESHOLD",d="Enter with \"high\" or \"low\".For high, certain bone will be segmented and all the background will be segmented adaptive. For low, certain background will be skipped and bone will be adaptively segmented")) 
        print("{:<25} | {d}".format("MULTIPROCESSING",d="use multiprocessing for increased computational speed. (uses overhead so only usefull for large matrix sizes)")) 
        print("{:<25} | {d}".format("WRITE_LOG",d="1/0. Write a log file, including the used parameter file and the Dice Similarity of the scan to a reference image"))
        print("{:<25} | {d}".format("WRITE_SEGMENTATION",d="1/0. Write a the segmented image as an mha file. The file is named as the input image, with `segmAdatpive` appended"))
        print("{:<25} | {d}".format("reference_file",d="Link to registered (i.e. transformed and resampled) reference segmentation file. (If DSC is to be calculated)."))
        print()
        print()
        print('--- %% GAUSSIAN MIXTURE MODEL %% ---')
        print()
        print("{:<25} | {d}".format("Parameter",d="Description"))
        print("--------------------------------------------------------------------")
        print("{:<25} | {d}".format("bone_file",d="Link to gray image volume to be segmented"))
        print("{:<25} | {d}".format("mask_file",d="Link to black/white mask file. Standard input is 8bit Uint"))
        print("{:<25} | {d}".format("output_folder",d="Folder to store segmentations and log files"))
        print("{:<25} | {d}".format("nb_bins",d="number of histogram bins the Gaussian functions will model"))
        print("{:<25} | {d}".format("max_iterations",d="maximum iterations of log likelihood maximization"))
        print("{:<25} | {d}".format("epsilon",d="Convergence criterion: iteration stopped if L_n - L_n-1 < epsilon"))
        print("{:<25} | {d}".format("epsilon_percentage",d="convergence criterion expressed as percentage of initial LogLikelihood. Has priority over epsilon"))
        print("{:<25} | {d}".format("CALIBRATE_MODEL",d="1/0. Whether or not program calculates GMM parameters, or directly segments the bone with the already calculated GMM parameters." ))
        print("{:<25} | {d}".format("REDUCED_SAMPLING",d="1/0. Whether or not to estimate GMM with reduced number of voxels in FOV (mask)"))
        print("{:<25} | {d}".format("nb_samples",d="number of samples if reduced sampling mode is on"))
        print("{:<25} | {d}".format("ADAPTIVE",d="1/0. Whether or not to use adaptive thresholding in the uncertain voxels"))
        print("{:<25} | {d}".format("ADAPTIVE_METHOD",d="gaussian, mean, median, mean_min_max"))
        print("{:<25} | {d}".format("radius",d="Radius of sphere to calculate local threshold, expressed in milimeters")) 
        print("{:<25} | {d}".format("certain_bone_fact",d="Bone voxel is uncertain if Pbone < factor * Psoft")) 
        print("{:<25} | {d}".format("certain_soft_fact",d="Non-bone voxel is uncertain if Psoft < factor * Pbone")) 
        print("{:<25} | {d}".format("WRITE_LOG",d="1/0. Write a log file, including the used parameter file, the single threshold and the Dice Similarity of the scan to a reference image"))
        print("{:<25} | {d}".format("WRITE_SEGMENTATION",d="1/0. Write a the segmented image as an mha file. The file is named as the input image, with `segmGMM` or `segmGMMadaptive` appended"))
        print("{:<25} | {d}".format("PLOT_SEGMENTATION",d="1/0. Plot original and segmented image in a slicer window."))
        print("{:<25} | {d}".format("reference_file",d="Link to registered (i.e. transformed and resampled) reference segmentation file. (If DSC is to be calculated)."))

    elif not os.path.isfile(args.param):
        raise NameError('ERROR: you did not specify a valid parameter file')
    
    else:
        if args.method == 'otsu' or args.method == 'Otsu':
            SegmentOtsu(parameter_file = args.param)
        elif args.method == 'adaptive':
            SegmentAdaptive(parameter_file = args.param)
        elif args.method == 'GMM':
            SegmentGMM(parameter_file = args.param)
        else:
            raise NameError('ERROR: you did not specify a valid segmentation method')
    


if __name__ == '__main__':
    main()


