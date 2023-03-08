# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:56:15 2023

@author: u0139075
"""

import SimpleITK as sitk
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def SegmentOtsu(parameter_file):
    pass

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
        pass
    

    elif not os.path.isfile(args.param):
        print('ERROR: you did not specify a valid parameter file')
    
    else:
        if args.method == 'otsu':
            SegmentOtsu(args.param)
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