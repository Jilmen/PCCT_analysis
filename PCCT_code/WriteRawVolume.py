# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:31:20 2023

@author: u0139075
"""

import SimpleITK as sitk
import sys
import os
import numpy as np
import argparse

def WriteRawVolume(volume, IMPORT_FILE):
    print("Reading image...")
    im = sitk.ReadImage(volume)
    
    dot_index = volume.find('.',-5)
    
    array = sitk.GetArrayFromImage(im)
    pixelID = im.GetPixelID()
    if pixelID == 0:
        asType = 'int8'
    elif pixelID == 1:
        asType = 'uint8'
    elif pixelID == 2:
        asType = 'int16'
    elif pixelID == 3:
        asType = 'uint16'
    elif pixelID == 4:
        asType = 'int32'
    elif pixelID == 5:
        asType = 'uint32'
    elif pixelID == 6:
        asType = 'int64'
    elif pixelID == 7:
        asType = 'uint64'
    else:
        print('Data is not integer type, storing as float 64')
        asType = 'float64'
    
    print(f'Used datatype: {asType}')
    array.astype(asType).tofile(volume[:dot_index] + '.raw')
    
    if IMPORT_FILE:
        print("Writing parameter file...")
        f = open(volume[:dot_index] + '_import_parameters.txt','w')
        f.write('data type: '+ asType + '\n')
        f.write('byte order: little endian \n')
        f.write('scan order: normal \n')
        f.write('header size: 0 bytes \n')
        dx = im.GetSpacing()[0]*1000
        f.write(f'Pixel size: {dx}\n')
        minval = np.min(array)
        maxval = np.max(array)
        f.write(f'Minor value: {minval}\n')
        f.write(f'Maximum value: {maxval}\n')
        (sx,sy,sz) = im.GetSize()
        f.write(f'Width: {sx}\n')
        f.write(f'Height: {sy}\n')
        f.write(f'Z size: {sz}\n')
        f.write('Z spacing: 1 \n')
        f.write('Z origin: 0 \n')
        f.close()
    print('Done!')

def main():
    parser = argparse.ArgumentParser(description='script to read in a 3D volume and write it as a raw file.\n'\
                                         'The same datatype is used as the volume. Only (un-)signed integers are implemented.'\
                                         'If the datatype is a float, float64 is used by default.\n'\
                                         'If wanted, a text file can be written in the same folder as the volume containing the CTan input parameters.')
    parser.add_argument('-vol', help='3D volume to be written as raw file', required=True)
    parser.add_argument('-import_file', help='If given as input, the import parameter text file will be written.', action='store_true', default=False)
    args = parser.parse_args()
    
    if not os.path.isfile(args.vol):
        print("ERROR: you did not specify a valid file as input volume")
        
    else:
        WriteRawVolume(args.vol, args.import_file)

if __name__ == '__main__':
    main()
       
                