# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:46:11 2023

@author: u0139075
"""

import SimpleITK as sitk
import numpy as np
import os
import argparse

def SetOrigin(image, mask, outImage, outMask):
    
    if mask.GetSize() != image.GetSize():
        print('ERROR: Mask and image are not of equal size!')
    
    else:
        mask_array = sitk.GetArrayFromImage(mask)
        
        print('Locating weights...')
        (depths, rows, cols) = np.where(mask_array != 0)
        
        x = np.sum(cols) / len(cols)
        y = np.sum(rows) / len(rows)
        z = np.sum(depths)/len(depths)
        cog = (x,y,z)
        print(f'Centre of geometry calculated: {cog} [index]')
        
        physical_cog = np.asarray(image.TransformContinuousIndexToPhysicalPoint(cog))
        origin = np.asarray(image.GetOrigin())
        physical_cog_local = physical_cog - origin
        new_origin = [-i for i in physical_cog_local]
        
        print('Adjusting image and mask origin')
        image.SetOrigin(new_origin)
        mask.SetOrigin(new_origin)
        
        print('Writing images...')
        sitk.WriteImage(image, outImage)
        sitk.WriteImage(mask, outMask)
        print('Done!')


def main():
    parser = argparse.ArgumentParser(description= \
                                     'Script to set the origin of the image, expressed in the world coordinate system,'\
                                     'at the center of geometry (COG) of a chosen region of interest.'\
                                     '\nIt outputs a new image and mask, identical to the originals, but with the origin changed.')
    parser.add_argument('-image', help='image where the origin has to be shifted', required=True)
    parser.add_argument('-mask', help='black/white image containing the region of interestx. (ROI is white)', required=True)
    parser.add_argument('-outImage', help='Name of output image. If not specified, default is [image]_COGCentered')
    parser.add_argument('-outMask', help='Name of output mask. If not specified, default is [mask]_COGCentered')
   
    args = parser.parse_args()
    outImg = args.outImage
    outMsk = args.outMask   
    if outImg is None:
        outImg = args.image[:-4] + '_COGCentered.mha'
    if outMsk is None:
        outMsk = args.mask[:-4] + '_COGCentered.mha'
        
    if not os.path.isfile(args.image) or not os.path.isfile(args.mask):
        print('ERROR: you seem to have given an incorrect image and/or mask.')
    else:
        print('Reading input...')
        img = sitk.ReadImage(args.image)
        msk = sitk.ReadImage(args.mask)
        print('Done reading input')
        
        SetOrigin(img, msk, outImg, outMsk)

if __name__ == '__main__':
    main()
    
    