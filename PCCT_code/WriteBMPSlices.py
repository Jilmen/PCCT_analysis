# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:06:24 2023

@author: u0139075
"""

import SimpleITK as sitk
import sys
import os
import numpy as np
import argparse

def Scale(volume, dtype):
    sitk_string = 'sitkUI' + dtype[2:]  #will always be unsigned ints, so index 2 is correct
    dtype = np.dtype(dtype)    
    max_dtype = np.iinfo(dtype).max
    
    arr = sitk.GetArrayViewFromImage(volume)
    MAX = np.max(arr)
    MIN = np.min(arr)
    
    arr = max_dtype / (MIN-MAX) * (arr - MAX)
    sitkPixelID = sitk.GetPixelIDValueFromString(sitk_string)
    volume = sitk.Cast(volume, sitkPixelID)
    return volume

def WriteBMPSlices(volume, output_folder):
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    writer = sitk.ImageSeriesWriter()
    writer.SetImageIO('BMPImageIO')
    
    # zeropadding
    nb_slices = volume.GetSize()[2]
    max_pad = len(str(nb_slices))-1
    zero_pad = ''
    for i in range(max_pad):
        zero_pad += '0'
    
    # filenames
    files = [os.path.join(output_folder, 'slice' + zero_pad[:-len(str(nb))] + str(nb) + '.bmp') for nb in range(nb_slices)]
    writer.SetFileNames(files)

    writer.Execute(volume)


def main():
    parser = argparse.ArgumentParser(description='script to read in a 3D volume and write it as a series of 2D bitmap images.'\
                                     '\nWarning, this is only directly possible for unsigned integer images!'\
                                         '\n If you want to write integer images to BMP, use the -rescale argument.')
    parser.add_argument('-vol', help='3D volume to be written as a series', required=True)
    parser.add_argument('-out', help='Folder to store output. If the folder does not exist, it will be created', required=True)
    parser.add_argument('-rescale', help='Flag to rescale the image data: a constant bias will remove any negative values, and all values will be scaled linearly in the range of the desired datatype.', \
                        action='store_true', default=False)
    parser.add_argument('-dtype', help='datatype for the rescaled image. These should be unsigned integer types!')
    args = parser.parse_args()

    if not os.path.isfile(args.vol):
        print("ERROR: you did not specify a valid file as input volume")
    
    elif args.rescale and (args.dtype is None or args.dtype[0] != 'u'):
        print("ERROR: you selected to rescale the image, but did not specify a correct datatype with the -dtype flag" \
              "\nThe compatible datatypes are uint8, uint16, uint32 or uint64.")
    
    elif args.rescale:
        print("Rescaling...")
        print("Reading in image...")
        volume = sitk.ReadImage(args.vol)
        print("Rescaling image...")
        volume = Scale(volume, args.dtype) 
        print("Writing...")
        WriteBMPSlices(volume, args.out)
        
    
    else:
        print("Reading in image...")
        volume = sitk.ReadImage(args.vol)
        print("Writing...")
        WriteBMPSlices(volume, args.out)

    
if __name__ == '__main__':
    main()
        