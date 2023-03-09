# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:01:31 2023

@author: u0139075
"""

import SimpleITK as sitk
import os
import numpy as np
import argparse
import sys

def CropImage(image, roi, margin, WRITE_IMAGE = False, outIm = '', outROI = ''):
    """
    Crop a large image such that only the region of interest and a predefined margin remain.
    Image and ROI should be SimpleITK images.
    The margin is specified in milimeters.
    Use the flag WRITE_IMAGE if you want to write the cropped image and ROI file. If so, provide names for the output images.

    """
    
    roi_array = sitk.GetArrayViewFromImage(roi)
    roi_bool = roi_array != 0
    (depths, rows, cols) = np.where(roi_bool == 1)
    
    min_x, max_x = np.min(cols), np.max(cols)
    min_y, max_y = np.min(rows), np.max(rows)
    min_z, max_z = np.min(depths), np.max(depths)
    
    (dx, dy, dz) = image.GetSpacing()
    (sx, sy, sz) = image.GetSize()
    
    Dx = np.ceil(margin/dx)
    Dy = np.ceil(margin/dy)
    Dz = np.ceil(margin/dz)
    
    min_x = int(max(0, min_x - Dx))
    min_y = int(max(0, min_y - Dy))
    min_z = int(max(0, min_z - Dz))
    max_x = int(min(sx - 1, max_x + Dx))
    max_y = int(min(sy - 1, max_y + Dy))
    max_z = int(min(sz - 1, max_z + Dz))
    
    image_crop = image[min_x:max_x, min_y:max_y, min_z:max_z]
    roi_crop = roi[min_x:max_x, min_y:max_y, min_z:max_z]
    
    if WRITE_IMAGE:
        print(f'Writing cropped image to {outIm}')
        sitk.WriteImage(image_crop, outIm)
        print(f'Writing cropped mask to {outROI}')
        sitk.WriteImage(roi_crop, outROI)
        print('Done')
    return image_crop, roi_crop
    

def main():
    parser = argparse.ArgumentParser(description = 'Script to crop large images to only a specified region of interest in the mask.\n'\
                                                   'Maximum positions in the mask are determined, and a margin is included.', \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-image', help='input image to be cropped', required = True)
    parser.add_argument('-mask', help='region of interest. Dimensions should match the image. You can enter multiple ROIs, but give the same amount of output names', nargs='+')
    parser.add_argument('-margin', help='Margin beyond the extrema of the region of interest to crop. Should be specified in milimeters.', default=10)
    parser.add_argument('-outImg', help='Name of output image. You can choose to specify a full path or just a filename.'\
                                        'In case of the latter, the files will be written in the same folder as the inputs', nargs='+')
    parser.add_argument('-outMask', help='Name of output mask. You can choose to specify a full path or just a filename.'\
                                        'In case of the latter, the files will be written in the same folder as the inputs', nargs='+')
    args = parser.parse_args()
    
    # check if input image exists
    if not os.path.exists(args.image):
        print('ERROR: you did not specify a valid image')
        sys.exit()
        
    #base strings of image and mask
    pos = args.image.rfind('/')
    base_im = args.image[:pos+1]
    
    # if one ROI is given, put in tuple for uniform handling
    if not isinstance(args.mask, list) and not isinstance(args.outMask, list) and not isinstance(args.outImg, list):
        set_masks = [args.mask]
        set_outMasks = [args.outMask]
        set_outImgs = [args.outImg]
    
    else:
        if len(args.mask) != len(args.outMask) or len(args.mask) != len(args.outImg):
            print('ERROR: you specified a list with multiple ROIs, but not the same number of output names.')
            sys.exit()
        else:
            set_masks = args.mask
            set_outMasks = args.outMask
            set_outImgs = args.outImg
            print(f'{len(set_masks)} mask files are given as input')
            
    print('Reading image...')
    img = sitk.ReadImage(args.image)
    
    for mask_nb in range(len(set_masks)):
        mask = set_masks[mask_nb]
        outMask = set_outMasks[mask_nb]
        outImg = set_outImgs[mask_nb]
        print(f'mask {mask_nb+1}/{len(set_masks)}')
        
    
        if not os.path.isfile(mask):
            print(f'{mask} is not a valid file. Skipping this mask.')
            
        else:
            if outImg.find('/') == -1:
                # image output has to be stored in same folder as input image
                outImg = base_im + outImg
                
            if outMask.find('/') == -1:
                # mask output has to be stored in same folder as input image
                pos = mask.rfind('/')
                outMask = mask[:pos+1] + outMask
            
            # check if extension is explicitly mentioned
            if outImg.find('.') == -1:
                outImg += '.mha'
            if outMask.find('.') == -1:
                outMask += '.mha'

            print('Reading mask...')
            mask = sitk.ReadImage(mask)
            
            CropImage(img, mask, float(args.margin), WRITE_IMAGE = True, outIm = outImg, outROI = outMask)

if __name__ == '__main__':
    main()
        
        
            
            
            
        
    
    
    
    
    
    
    
    
    