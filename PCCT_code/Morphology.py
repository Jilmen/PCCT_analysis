# -*- coding: utf-8 -*-
"""
Created on Mon May 29 09:03:46 2023

@author: u0139075
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import argparse
import os
import gc
import sys

def Pad(arr, pad_amount, action):
    
    """
    Extends image array with zero / False on the edges. 
    
    Input:
        - arr: numpy array to extend in size
        - pad_amount: number of voxels to increase in each direction
        - action: string, 'add' or 'remove' increase or decrease the size 
    """
    (i, j, k) = np.shape(arr)
    if action == 'add':
        new_shape = (i+2*pad_amount, j+2*pad_amount, k+2*pad_amount)
        new_arr = np.zeros(shape=new_shape, dtype = arr.dtype)
        new_arr[pad_amount:i+pad_amount, pad_amount:j+pad_amount, pad_amount:k+pad_amount] = arr
        
    elif action == 'remove':
        i=-2*pad_amount
        j-=2*pad_amount
        k-=2*pad_amount
        new_arr = arr[pad_amount:i+pad_amount, pad_amount:j+pad_amount, pad_amount:k+pad_amount]
    return new_arr
    
    
def SingleThreshold(arr, thresh):
    return arr >= thresh

def RemoveSpeckles(binary_arr, max_size):
    
    """
    Removes white speckles in a binary image using standard morphology operations:
        1. Eroding the image
        2. Dilating the image
    
    Input:
        - binary_arr: numpy array with binarized image (boolean voxels)
        - max_size: maximal size of a speckle (i.e. cluster voxels) to remove
    
    Output:
        - Boolean mask with same dimensions as input array. 
    """
    
    if not binary_arr.dtype == np.dtype(bool):
        binary_arr = binary_arr.astype(bool)
        
    struct = ndimage.generate_binary_structure(3, 1).astype(binary_arr.dtype)
    binary_arr = ndimage.binary_erosion(binary_arr, struct, iterations=max_size)
    binary_arr = ndimage.binary_dilation(binary_arr, struct, iterations=max_size)
    return binary_arr

def Sweep(binary_arr):
    
    """
    Only retains the largest object (white voxels) in a binary image, using a label map.
    The label map is tried to be constructed in 16 bit. If more clusters are detected than can be described, 
    the label map is constructed in 32 bit.
    
    Input:
        - binary_arr: numpy array with binarized image (boolean voxels)
    Output:
        - Boolean array with same dimensions as input array. 
    """
    
    try:
        labeled_arr, nb_labels = ndimage.label(binary_arr, output = np.uint16)
    except:
        print('Too many clusters to label with 16-bits. Trying 32 bits.')
        # freeing up buffer space
        labeled_arr = None
        nb_labels = None
        gc.collect()
        labeled_arr, nb_labels = ndimage.label(binary_arr, output = np.uint32)
    labs, counts = np.unique(labeled_arr, return_counts=True)
    ind = np.argmax(counts[1:]) + 1
    max_label = labs[ind]
    return labeled_arr == max_label

def FullBoneMask(image_arr, closing_vox):
    
    """
    Calculates a full bone mask using standard morphology operations:
        1. Sweeping the image, and only maintaining the largest object
        2. Dilating the image
        3. Filling the holes
        4. Eroding the image
    
    Input:
        - image: numpy array with binarized image
        - closing_vox: integer value with number of voxels for closing operation
    
    Output:
        - Boolean mask with same dimensions as image. 
    """
    # step 1: thresholding, has to be done already in preprocessing. 
    # Make sure image is boolean
    if not image_arr.dtype == np.dtype(bool):
        image_arr = image_arr.astype(bool)
    
    # make sure there is enough space for the closing operation
    image_arr = Pad(image_arr, closing_vox, 'add')
        
    # step 2: sweeping the image for largest object
    print('sweeping...')
    # despeckled_arr = RemoveSpeckles(binary_arr, 1)
    sweeped_arr = Sweep(image_arr)
    
    # step 3: dilate image
    print('dilating...')
    struct = ndimage.generate_binary_structure(3, 1).astype(sweeped_arr.dtype)
    dilated_arr = ndimage.binary_dilation(sweeped_arr, struct, iterations = closing_vox).astype(sweeped_arr.dtype)
    
    # step 4: fill holes
    print('filling...')
    filled_arr = ndimage.binary_fill_holes(dilated_arr).astype(dilated_arr.dtype)
    
    # step 5: erode image
    print('eroding...')
    mask_arr = ndimage.binary_erosion(filled_arr, struct, iterations = closing_vox).astype(bool)
    
    # remove padding
    mask_arr = Pad(mask_arr, closing_vox, 'remove')
    
    return mask_arr

def TrabecularMask(image_arr, mask_arr):
    dtype = mask_arr.dtype
    struct = ndimage.generate_binary_structure(3,1).astype(dtype)
    
    image_arr *= (mask_arr != 0)

    print('Calculating VOI...')
    im_dilate = ndimage.binary_dilation(image_arr, struct, 3)
    mask_erode = ndimage.binary_erosion(mask_arr, struct, 10)
    
    term1 = np.bitwise_xor(mask_arr, im_dilate)
    term2 = np.bitwise_xor(mask_arr, mask_erode)
    voi_open = np.bitwise_and(term1, np.bitwise_not(term2))
    voi_open = np.bitwise_and(voi_open, mask_arr)
    
    print('Closing VOI...')
    voi = ndimage.binary_dilation(voi_open, struct, 15).astype(dtype)
    voi = ndimage.binary_fill_holes(voi).astype(dtype)
    voi = ndimage.binary_erosion(voi, struct, 15).astype(dtype)
    return voi
    


def main():
    parser = argparse.ArgumentParser(description='implementation of multiple morphology based binary operations, implemented in numpy and scipy.ndimage')
    parser.add_argument('-image', help='path to input (gray) image', default='')
    parser.add_argument('-init_thresh', help='initial threshold before binary operations start', default='0')
    parser.add_argument('-operation', help='which morphology algorithm to perform', default='')
    parser.add_argument('-closing_vox', help='number of voxels for the closing operation in the binary mask algorithm', default=None)
    parser.add_argument('-despeckle_vox', help='maximum voxel size for despeckle operation', default=None)
    parser.add_argument('-bone_mask', help='Full bone mask as input for the trabeculmar voi algorithm. If not passed along, you should specify the closing voxels', default=None)
    parser.add_argument('-out', help = 'output file name', default='C:/workdir/morphology.mha')
    parser.add_argument('-list', help='list all implemented operations', action='store_true')
    
    
    args = parser.parse_args()
    
    if args.list:
        print("{:<25} | {d}".format("Operation",d="Explanation"))
        print("--------------------------------------------------------------------")
        print("{:<25} | {d}".format("mask",d="Full bone mask with closing and filling"))
        print("{:<25} | {d}".format("despeckle",d="Remove white speckles with maximum size"))
        print("{:<25} | {d}".format("despeckle_mask",d="Consecutively perform despeckling and full bone masking"))
        print("{:<25} | {d}".format("trab_voi",d="Trabecular VOI. You can give a full bone mask, or the parameters for the `despeckle_mask`procedure"))

        sys.exit(0)
        
        
    if not os.path.isfile(args.image):
        raise RuntimeError(f'{args.image} is not a valid path to a file')
    
    print('Reading image...')
    im = sitk.ReadImage(args.image)
    
    if args.operation == 'mask':
        if args.closing_vox == None:
            raise RuntimeError(f'Masking algorithm: missing input. closing_vox = {args.closing_vox}')
        
        vox = np.uint8(args.closing_vox)
        thresh = np.uint8(args.init_thresh)
        if thresh == 0:
            print('WARNING: no threshold given as input. Using default 100')
            thresh = 100
        
        print('thresholding...')
        image_arr = sitk.GetArrayFromImage(im)
        image_arr = SingleThreshold(image_arr, thresh)
        mask_arr = FullBoneMask(image_arr, vox)
        mask_arr = 255* mask_arr.astype(np.uint8)
        
        mask = sitk.GetImageFromArray(mask_arr)
        mask.SetOrigin(im.GetOrigin())
        mask.SetSpacing(im.GetSpacing())
        
        print('Writing image...')
        sitk.WriteImage(mask, args.out)
    
    elif args.operation == 'despeckle_mask':    
        if args.closing_vox == None or args.despeckle_vox == None:
            raise RuntimeError(f'Despeckle and mask algorithm: missing input. closing_vox = {args.closing_vox} ; despeckle_vox = {args.despeckle_vox}')
        
        closing_vox = np.uint8(args.closing_vox)
        despeckle_vox = np.uint8(args.despeckle_vox)
        thresh = np.uint8(args.init_thresh)
        if thresh == 0:
            print('WARNING: no threshold given as input. Using default 100')
            thresh = 100
        
        print('thresholding...')
        image_arr = sitk.GetArrayFromImage(im)
        image_arr = SingleThreshold(image_arr, thresh)
        
        print('despeckling...')
        despeckle_arr = RemoveSpeckles(image_arr, despeckle_vox)
        
        mask_arr = FullBoneMask(despeckle_arr, closing_vox)
        mask_arr = 255* mask_arr.astype(np.uint8)
        mask = sitk.GetImageFromArray(mask_arr)
        mask.SetOrigin(im.GetOrigin())
        mask.SetSpacing(im.GetSpacing())
        
        print('Writing image...')
        sitk.WriteImage(mask, args.out)
    
    elif args.operation == 'despeckle':
        if  args.despeckle_vox == None:
            raise RuntimeError(f'Despeckle : missing input. despeckle_vox = {args.despeckle_vox}')
        
        despeckle_vox = np.uint8(args.despeckle_vox)
        thresh = np.uint8(args.init_thresh)
        if thresh == 0:
            print('WARNING: no threshold given as input. Using default 100')
            thresh = 100
        
        print('thresholding...')
        image_arr = sitk.GetArrayFromImage(im)
        image_arr = SingleThreshold(image_arr, thresh)
        
        print('despeckling...')
        mask_arr = RemoveSpeckles(image_arr, despeckle_vox)
        mask_arr = 255* mask_arr.astype(np.uint8)
        mask = sitk.GetImageFromArray(mask_arr)
        mask.SetOrigin(im.GetOrigin())
        mask.SetSpacing(im.GetSpacing())
        
        print('Writing image...')
        sitk.WriteImage(mask, args.out)
        
    elif args.operation == 'trab_voi':
        thresh = np.uint8(args.init_thresh)
        if thresh == 0:
            print('WARNING: no threshold given as input. Using default 100')
            thresh = 100
            
        print('thresholding...')
        image_arr = sitk.GetArrayFromImage(im)
        image_arr = SingleThreshold(image_arr, thresh)
            
        if args.bone_mask == None:
            if args.closing_vox == None:
                raise RuntimeError('Trabecular volume of interest cannot be calculated if no mask or no closing voxels is passed')
            vox = np.uint8(args.closing_vox)
            mask_arr = FullBoneMask(image_arr, vox)
        else:
            print('Reading full bone mask...')
            mask = sitk.ReadImage(args.bone_mask)
            mask_arr = sitk.GetArrayViewFromImage(mask)
            mask_arr = mask_arr != 0
        
        voi_arr = TrabecularMask(image_arr, mask_arr)
        voi_arr = 255*voi_arr.astype(np.uint8)
        voi = sitk.GetImageFromArray(voi_arr)
        voi.SetSpacing(im.GetSpacing())
        voi.SetOrigin(im.GetOrigin())
        
        print('Writing image...')
        sitk.WriteImage(voi, args.out)
        
        
        
        
            
            
    else:
        print(f'WARNING: {args.operation} is not a valid operation. Exitting script without further operations...')
    

if __name__ == '__main__':
    main()