# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:12:28 2023

@author: u0139075
"""

import numpy as np
import SimpleITK as sitk
import argparse
import os

def MergeMasks(mask_list, reference_image):
    """
    Function to merge multiple binary masks into a single image. 
    Masks are correctly placed with respect to each other based on their origin.
    
    Inputs:
        - mask_list: iterable with the masks. Each mask should be a SimpleITK image, 0 as background value.
        - reference_image: SimpleITK image for which each mask is a ROI of.
        
    Output:
        - SimpleITK image with same spacing as inputs masks.
    """
    
    spacing = reference_image.GetSpacing()
    origin = reference_image.GetOrigin()
    (cols, rows, depths) = reference_image.GetSize()
    merged_array = np.zeros((depths,rows,cols), dtype=bool)

    for mask in mask_list:
        if not np.all(np.round(mask.GetSpacing(),5) == np.round(spacing,5)):
            raise ValueError('ERROR: not all masks have the same spacing!')
        
        local_origin = mask.GetOrigin()
        (ox, oy, oz) = reference_image.TransformPhysicalPointToIndex(local_origin)
        (sx, sy, sz) = mask.GetSize()
        
        # convert to arrays to easily implement Boolean OR
        mask_array = sitk.GetArrayViewFromImage(mask)
        mask_array = mask_array > 0
        
        merged_array[oz:oz+sz , oy:oy+sy , ox:ox+sx] += mask_array
        
    merged_image = sitk.GetImageFromArray(merged_array.astype(np.uint8))
    merged_image.SetSpacing(spacing)
    merged_image.SetOrigin(origin)
    return merged_image
        
def main():
    parser = argparse.ArgumentParser(description = 'Script to merge multiple smaller masks images into a larger combined mask, with preservation of relative position.')
    parser.add_argument('-masks', help='SimpleITK mask images to combine. (0 as background value). Enter all of them separated with spaces', nargs='+', required=True)
    parser.add_argument('-reference', help='SimpleITK reference image for which each mask is a ROI', required = True)
    parser.add_argument('-out', help='path for writing output combined mask.', required=True)
    
    args = parser.parse_args()
    
    
    for m in args.masks:
        if not os.path.isfile(m):
            raise RuntimeError(f'ERROR: {m} is not a valid mask file')
    if not os.path.isfile(args.reference):
        raise RuntimeError(f'ERROR: {args.reference} is not a valid reference file')
    
    print('Reading reference image...')
    ref = sitk.ReadImage(args.reference)
    
    print('Reading masks...')
    masks = [sitk.ReadImage(i) for i in args.masks]
    
    print('Merging...')
    merged = MergeMasks(masks, ref)
    
    print('Writing image...')
    sitk.WriteImage(merged, args.out)
    
if __name__ == '__main__':
    main()
        
    

        
        