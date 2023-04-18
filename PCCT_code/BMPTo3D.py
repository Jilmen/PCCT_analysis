# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:45:53 2023

@author: u0139075
"""

import SimpleITK as sitk
import sys
import os
import argparse

def BMPTo3D(input_folder, output_file, dx):
    files = [os.path.join(input_folder, i) for i in os.listdir(input_folder) if i[-3:]=='bmp' and '0' in i]
    if len(files) == 0:
        raise ValueError(f'{input_folder} has no valid BMP files!')
        
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(files)
    series_reader.SetImageIO('BMPImageIO')
    series_reader.SetOutputPixelType(sitk.sitkUInt8)
    
    print(f'Reading in files from folder {input_folder}')
    image = series_reader.Execute()
    print(f'Done reading. 3D image of size {image.GetSize()}')
    
    image.SetSpacing((dx,dx,dx))
    
    print('Writing image...')
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_file)
    writer.Execute(image)
    print('Done!')

def main():
    parser = argparse.ArgumentParser(description='script to read a series of bitmap files into a volume and write the volume as a ITK compatible .mha-file.')
    parser.add_argument('-path', help='Link to directory containing dicom files', required=True)
    parser.add_argument('-out', help='Output file to be created', required=True)
    parser.add_argument('-pixelsize', help='Isotropic pixel size expressed in milimeters', required=True)
    args = parser.parse_args()
    
    if not os.path.isdir(args.path):
        raise ValueError(f'{args.path} is not a valid directory')
    
    out = args.out
    if out.find('.') == -1:
        out += '.mha'
    
    dx = float(args.pixelsize)
    
    BMPTo3D(args.path, out, dx)


if __name__ == '__main__':
    main()