# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:41:07 2023

@author: u0139075
"""
import SimpleITK as sitk
import os
import numpy as np
import argparse

def WriteVolume(slices, outputName, reference_image = None, origin = None, spacing = None):
    
    """
    Function to write a folder with slices of bmp images into a mha volume. 
    You have to either specify a reference image, or the origin and spacing.
    The output file will be stored in the folder with all the bmp images.
    """
    
    if reference_image is None and (origin is None or spacing is None):
        raise RuntimeError('ERROR: you did not specify a reference image or a complete set of metadata (origin and spacing)')
    
    else:
        if reference_image is not None:
            origin = reference_image.GetOrigin()
            spacing = reference_image.GetSpacing()
        
        print('Reading in slices...')
        file_names = os.listdir(slices)
        files = [os.path.join(slices, i) for i in file_names if i[-3:] == 'bmp' and 'spr' not in i]
        print(f'{len(files)} files found in directory {slices}')
        
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        reader.SetOutputPixelType(sitk.sitkUInt8)
        img = reader.Execute()
        
        print('Writing image...')
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        
        # check if outputName has extension already specified
        if outputName.find('.') == -1:
            outputName += '.mha'
        sitk.WriteImage(img, os.path.join(slices, outputName))
        print('Done!')

def main():
    parser = argparse.ArgumentParser(description = \
                                     'Script to read in a folder with bitmap images and write them in a 3D volume (mha file).'\
                                     '\nYou have to specify image origin and spacing, either by giving a reference image, or by giving it explicitly.')
    parser.add_argument('-folder', help = 'Folder containing bitmap images. If other files are also in the folder, they will be ignored.', \
                        required = True)
    parser.add_argument('-out', help = 'name for output file. It will automatically be stored in the same folder as the bitmaps.', required=True)
    parser.add_argument('-reference', help = 'path to reference image file', default = '')
    parser.add_argument('-origin', help = 'coordinate of the origin. Specify as three numbers with spaces: x y z', nargs = 3, default = None)
    parser.add_argument('-spacing', help = 'voxel dimensions. Specify as three numbers with spaces: dx dy dz', nargs = 3, default = None)
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.reference) and (args.origin is None or args.spacing is None):
        raise RuntimeError('ERROR: you did not specify a valid reference image or a complete set of metadata (origin and spacing)')
    else:
        if os.path.isfile(args.reference):
            print('Reading in reference image...')
            reference = sitk.ReadImage(args.reference)
            WriteVolume(args.folder, args.out, reference_image = reference)
        else:
            print('Using manually entered origin and spacing...')
            origin = [float(i) for i in args.origin]
            spacing = [float(i) for i in args.spacing]
            WriteVolume(args.folder, args.out, origin = origin, spacing = spacing)
            

if __name__ == '__main__':
    main()
    
        
    