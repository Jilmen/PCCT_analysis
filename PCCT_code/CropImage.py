# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:01:31 2023

@author: u0139075
"""

import SimpleITK as sitk
import os
import numpy as np
import argparse

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
    parser.add_argument('-margin', help='Margin beyond the extrema of the region of interest to crop. Should be specified in milimeters.', default=10)
    parser.add_argument('-mask', help='region of interest. Dimensions should match the image.', required = True)
    parser.add_argument('-outImg', help='Name of output image. You can choose to specify a full path or just a filename.'\
                                        'In case of the latter, the files will be written in the same folder as the inputs', required = True)
    parser.add_argument('-outMask', help='Name of output mask. You can choose to specify a full path or just a filename.'\
                                        'In case of the latter, the files will be written in the same folder as the inputs', required = True)
    args = parser.parse_args()
    
    if not os.path.isfile(args.image) or not os.path.isfile(args.mask):
        print('ERROR: you did not specify a valid image or mask file.')
    else:
        if args.outImg.find('/') == -1:
            # image output has to be stored in same folder as input image
            pos = args.image.rfind('/')
            outImg = args.image[:pos+1] + args.outImg
        else: 
            # image output is explicitly given
            outImg = args.outImg
        
        if args.outMask.find('/') == -1:
            # mask output has to be stored in same folder as input mask
            pos = args.mask.rfind('/')
            outImg = args.mask[:pos+1] + args.outMask
        else: 
            # mask output is explicitly given
            outMask = args.outMask
        
        # check if extension is explicitly mentioned
        if outImg.find('.') == -1:
            outImg += '.mha'
        if outMask.find('.') == -1:
            outMask += '.mha'
        
        print('Reading image...')
        img = sitk.ReadImage(args.image)
        print('Reading mask...')
        mask = sitk.ReadImage(args.mask)
        
        CropImage(img, mask, args.margin, WRITE_IMAGE = True, outIm = outImg, outROI = outMask)

if __name__ == '__main__':
    main()
        
        
            
            
            
        
    
    
    
    
    
    
    
    
    