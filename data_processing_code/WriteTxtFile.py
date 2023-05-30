# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:41:15 2023

@author: u0139075
"""

import numpy as np
import SimpleITK as sitk
import argparse
import os
from tqdm import tqdm

def WriteTxtFile(im, mask, out):
    
    '''
    Parameters
    ----------
    im : sitk-Image
        Input image from which HU units have to be stored in a txt file
    mask : sitk-Image
        Mask image containing the ROI. The script searches for values not equal to 0 as ROI values
    out : Path_to_file.txt 
        Text file in which all the voxel HU values will be listed. All spatial information is lost

    '''
    
    arrIm = sitk.GetArrayViewFromImage(im)
    arrMask = sitk.GetArrayViewFromImage(mask)
    [d,r,c] = np.where(arrMask != 0)
    file = open(out, 'w')
    
    for i in tqdm(range(len(d)-1)):
        voxel_value = arrIm[d[i], r[i], c[i]]
        file.write(f'{voxel_value}\n')
    last_value = arrIm[d[-1], r[-1], c[-1]]
    file.write(f'{last_value}') #in order not to have one break line too many
    
    file.close()
    

def main():
    parser = argparse.ArgumentParser(description= 'Script to extract all HU values within a ROI from an image, and store them in a vector txt file. '\
                                                 'All spatial information is lost.')
    parser.add_argument('-image', help='path to input image', required = True)
    parser.add_argument('-mask', help='path to mask image. A mask has values 0 for voxel out ouf the ROI.', required =True)
    parser.add_argument('-out', help='path to output text file. If the file already exists, it will be overwritten', required=True)
    
    args = parser.parse_args()
    if not os.path.isfile(args.image) or not os.path.isfile(args.mask):
        raise ValueError(f'You gave an invalid path for the mask and / or image!')
    
    if args.out.find('.txt') == -1:
        raise ValueError(f'{args.out} is not a text file. Specify as path/to/file.txt')
    
    print('Reading image...')
    im = sitk.ReadImage(args.image)
    print('Reading mask...')
    mask = sitk.ReadImage(args.mask)
    
    WriteTxtFile(im, mask, args.out)
    
    
    
if __name__ == '__main__':
    main()
        
    