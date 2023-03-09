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
import matplotlib.pyplot as plt

def DSC(img1, img2):
    arr1 = sitk.GetArrayViewFromImage(img1)
    arr2 = sitk.GetArrayViewFromImage(img2)
    vol1 = arr1 != 0
    vol2 = arr2 != 0
    overlap = 2*np.sum(vol1 * vol2)
    total = np.sum(vol1) + np.sum(vol2)
    return overlap/total

def SegmentOtsu(parameter_file = 'default', bone = None, mask = None, reference = None):
    """
    Perform Otsu thresholding. Parameters can be given in a parameter file. If not given, defaults will be used.
    If bone AND mask are also passed as an argument, they will be ignored in the parameter file. 
    Should you explicitly enter them, they should be SimpleITK images (and thus no paths)
    """
    
    # default parameters
    bone_file = ''
    mask_file = ''
    nb_bins = 128
    WRITE_LOG = 0
    WRITE_SEGMENTATION = 0
    reference_file = ''
    
    # reading the parameter file
    if parameter_file != 'default':
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
                    elif param == 'nb_bins':
                        nb_bins = int(value)
                    elif param == 'WRITE_LOG':
                        WRITE_LOG = int(value)
                    elif param == 'WRITE_SEGMENTATION':
                        WRITE_SEGMENTATION = int(value)
                    elif param == 'reference_file':
                        reference_file = value
                    else:
                        print(f'parameter input {param} is not a valid input!')
    
    if bone is None or mask is None:
        print('Reading in image...')
        bone = sitk.ReadImage(bone_file)
        print('Reading in mask...')
        mask = sitk.ReadImage(mask_file)
    
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
    
    if WRITE_SEGMENTATION:
        if bone_file == '':
            print('ERROR: you want to save the segmented image, but did not specify a bone_file path. Hence, I do not know where the data should be stored!')
        else:
            output = bone_file[:-4] + '_segmOtsu.mha'
            print(f'Writing segmentation to {output}...')
            sitk.WriteImage(segm, output)
    
    if WRITE_LOG:
        thresh = filt.GetThreshold()
        if reference is None:
            print('DSC cannot be calculated as you did not give a registered reference image in any way.')
            dsc = 'unknown'
        else:
            dsc = DSC(segm, reference)
        
        if bone_file == '':
            print('ERROR: you want to write a log-file, but did not specify a bone_file path. Hence, I do not know where the data should be stored!')
        else:
            log_file = bone_file[:-4] + '_Otsu_log.txt'
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
        
    return segm
                    
            

def SegmentGMM(parameter_file, ADAPTIVE):
    pass

def SegmentAdaptive(parameter_file):
    pass

def main():
    parser = argparse.ArgumentParser(description = \
                                     'Script to segment SimpleITK images. Currently implemented methods:\n' \
                                     '-Otsu --> single threshold\n' \
                                     '-Gaussian Mixture Model --> single threshold\n' \
                                     '-Adaptive --> local threshold\n' \
                                     '-Gaussian Mixture Model with adaptive in uncertainty\n\n'\
                                     'Because of the large amount of parameters, varying with each segmentation scheme, you should input them in a seperate txt-file.\n'\
                                     'Run >>> SegmentImage.py -list for an overview of all parameters',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-list', help='Give an overview of parameters for each segmentation algorithm, and exit the program', action='store_true', default=False)
    parser.add_argument('-method', help='Segmentation algorithm. Specify by keyword otsu , adaptive, GMM or GGM_adaptive')
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
        print("{:<25} | {d}".format("nb_bins",d="Number of bins in the histogram for the Otsu method")) 
        print("{:<25} | {d}".format("WRITE_LOG",d="1/0. Write a log file, including the used parameter file and the Dice Similarity of the scan to a reference image"))
        print("{:<25} | {d}".format("WRITE_SEGMENTATION",d="1/0. Write a the segmented image as an mha file."))
        print("{:<25} | {d}".format("reference_file",d="Link to registered (i.e. transformed and resampled) reference segmentation file. (If DSC is to be calculated)."))
        print()
        print()
        print('--- %% GAUSSIAN MIXTURE MODEL %% ---')
        print()
        print("{:<25} | {d}".format("Parameter",d="Description"))
        print("--------------------------------------------------------------------")
        print("{:<25} | {d}".format("bone_file",d="Link to gray image volume to be segmented"))
        print("{:<25} | {d}".format("mask_file",d="Link to black/white mask file. Standard input is 8bit Uint"))
        print("{:<25} | {d}".format("micro_file",d="Link to registered (i.e. transformed and resampled) micro CT segmentation file"))
        print("{:<25} | {d}".format("CALIBRATE_MODEL",d="1/0. Whether or not program calculates GMM parameters, or segments the bone with the already calculated GMM parameters."))
        print("{:<25} | {d}".format("PLOT_SEGMENTATIONS",d="1/0. Plot Otsu, GMM with uncertainty in slicer. Ignored if Calibrate mode is on."))
        print("{:<25} | {d}".format("REDUCED_SAMPLING",d="1/0. Whether or not to estimate GMM with reduced number of voxels in FOV (mask)"))
        print("{:<25} | {d}".format("REDUCED_CRITERION",d="1/0. Whether or not to calculate log likelihood (optimizer) on same reduced set of voxels."))
        print("{:<25} | {d}".format("ADAPTIVE_METHOD",d="gaussian, mean, median or mean_min_max"))
        print("{:<25} | {d}".format("histogram_bins",d="number of histogram bins to model Gaussian functions to"))
        print("{:<25} | {d}".format("max_iterations",d="maximum iterations of log likelihood maximization"))
        print("{:<25} | {d}".format("eps",d="Convergence criterion: iteration stopped if L_n - L_n-1 < eps"))
        print("{:<25} | {d}".format("nb_samples",d="number of samples if reduced sampling mode is on"))
        print("{:<25} | {d}".format("WRITE_GMM",d="1/0. Whether or not to write GMM segmentation volume to file"))
    

    elif not os.path.isfile(args.param):
        print('ERROR: you did not specify a valid parameter file')
    
    else:
        if args.method == 'otsu':
            SegmentOtsu(parameter_file = args.param)
        elif args.method == 'adaptive':
            SegmentAdaptive(args.param)
        elif args.method == 'GMM':
            SegmentGMM(args.param, ADAPTIVE = False)
        elif args.method == 'GMM_adaptive':
            SegmentGMM(args.param, ADAPTIVE = True)
        else:
            print('ERROR: you did not specify a valid segmentation method')
    


if __name__ == '__main__':
    main()