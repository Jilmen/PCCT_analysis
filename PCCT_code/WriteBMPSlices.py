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

def Scale(volume, windowMin = -700, windowMax = 2550):
    filt = sitk.IntensityWindowingImageFilter()
    filt.SetOutputMinimum(0)
    filt.SetOutputMaximum(255)
    filt.SetWindowMinimum(windowMin)
    filt.SetWindowMaximum(windowMax)
    volume = filt.Execute(volume)
    volume = sitk.Cast(volume, sitk.sitkUInt8)
    return volume

def WriteBMPSlices(volume, output_folder):
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    writer = sitk.ImageSeriesWriter()
    writer.SetImageIO('BMPImageIO')
    
    # zeropadding
    nb_slices = volume.GetSize()[2]
    max_pad = len(str(nb_slices))
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
    parser.add_argument('-rescale', help='Flag to rescale the image data. Images will be rescaled to unsigned 8bit image data.' \
                                        '\nBy default, the windowing of the input data will be from -700HU to 2550HU (including all soft tissue and bone). If needed, these defaults can be overwritten', \
                        action='store_true', default=False)
    parser.add_argument('-windowMin', help='minimum value of the windowing function used when rescaling the data to uint8. Default -700HU.', default='-700')
    parser.add_argument('-windowMax', help='minimum value of the windowing function used when rescaling the data to uint8. Default 2550HU.', default='2550')
    args = parser.parse_args()

    if not os.path.isfile(args.vol):
        print("ERROR: you did not specify a valid file as input volume")
    
    elif args.rescale:
        print("Rescaling... (Warning, image wil be rescaled to uint8 datatypes and information can be lost!)")
        print("Reading in image...")
        volume = sitk.ReadImage(args.vol)
        print("Rescaling image...")
        volume = Scale(volume, int(args.windowMin), int(args.windowMax)) 
        print("Writing...")
        WriteBMPSlices(volume, args.out)
        
    
    else:
        print("Reading in image...")
        volume = sitk.ReadImage(args.vol)
        print("Writing...")
        WriteBMPSlices(volume, args.out)

    
if __name__ == '__main__':
    main()
        