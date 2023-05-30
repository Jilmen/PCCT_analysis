# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:27:00 2023

@author: u0139075
"""

import numpy as np
import SimpleITK as sitk
import os
import argparse

def ClipImage(im, mask, mask_value = 255, dilate = 0, WRITE_IMAGE = False, outIm = ''):
    
    if dilate > 0:
        filt = sitk.BinaryDilateImageFilter()
        filt.SetForegroundValue(mask_value)
        filt.SetKernelRadius(dilate)
        
        mask = filt.Execute(mask)
        
    arr_im = sitk.GetArrayFromImage(im)
    arr_mask = sitk.GetArrayViewFromImage(mask)
    [d,r,c] = np.where(arr_mask != mask_value)
    arr_im[d,r,c] = -1024
    
    im_clip = sitk.GetImageFromArray(arr_im)
    im_clip.SetOrigin(im.GetOrigin())
    im_clip.SetSpacing(im.GetSpacing())
    
    if WRITE_IMAGE:
        print('Writing image...')
        sitk.WriteImage(im_clip, outIm)
        print('Done!')
    
    return im_clip

def main():
    parser = argparse.ArgumentParser(description='Clip a ROI out of a full image by providing a mask')
    parser.add_argument('-image', help = 'input image', required = True)
    parser.add_argument('-mask', help = 'image containing the mask', required = True)
    parser.add_argument('-maskValue', help = 'value in mask image that indicates the ROI. Default is 255', default = '255')
    parser.add_argument('-dilateValue', help = 'Dilate the mask with pixels before the clipping operation. Default is no dilation', default = '0')
    parser.add_argument('-out', help = 'output file to write clipped image', required = True)
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.image) or not os.path.isfile(args.mask):
        raise ValueError('Input image and/or mask is not a valid file.')
    
    out = args.out
    pos = out.rfind('/')
    if pos != -1:
        folder = out[:pos]
        if not os.path.isdir(folder):
            raise ValueError(f'{folder} is not an existing directory. You have to create it yourself!')

    print('Reading image...')
    im = sitk.ReadImage(args.image)
    print('Reading mask...')
    mask = sitk.ReadImage(args.mask)
    
    dilate = int(args.dilateValue)
    maskValue = int(args.maskValue)
    
    ClipImage(im, mask, mask_value = maskValue, dilate = dilate, WRITE_IMAGE = True, outIm = out)
    

if __name__ == '__main__':
    main()