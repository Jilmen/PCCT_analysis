# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:51:37 2023

@author: u0139075
"""

import SimpleITK as sitk
import sys
import numpy as np
import os
import argparse


def CheckDatatypes(path, dtype, cast):
    im = sitk.ReadImage(path)
    arr = sitk.GetArrayViewFromImage(im)
    minval = np.min(arr)
    maxval = np.max(arr)

    reduced_dtype = np.dtype(dtype)
    info = np.iinfo(reduced_dtype)
    MAX = info.max
    MIN = info.min


    print(f"max value image: {maxval}")
    print(f"min value image: {minval}")
    print(f"maximal value using {dtype}: {MAX}")
    print(f"minimal value using {dtype}: {MIN}")
    if maxval <= MAX and minval >= MIN:
        print("You can reduce the bits per voxel without losing image information")
        if cast:
            print("Casting down the image...")
            if dtype.find('u') == -1:       # signed integer to be translated to sitk syntax
                sitk_string = 'sitkI'+ dtype[1:]
            else:                                   # unsigned integer to be translated to sitk syntax
                sitk_string = 'sitkUI'+ dtype[2:]
            
            sitk_dtype = sitk.GetPixelIDValueFromString(sitk_string)
            im_cast = sitk.Cast(im, sitk_dtype)
        
            im_newName = path[:-4] + '_' + dtype + path[-4:]
            sitk.WriteImage(im_cast, im_newName)   

    else:
        print("WARNING: if you reduce the bits per voxel, you will lose image information!!")


def main():
    parser = argparse.ArgumentParser(description='Script to check if you can cast down your image without losing information')
    parser.add_argument('-path', help='Link to image', nargs='?')
    parser.add_argument('-dtype', help='Reduced data type', nargs='?')
    parser.add_argument('-table', action='store_true', default=False, help='Prints a list of all implemented datatypes, and the range of numbers they can represent')
    parser.add_argument('-cast', action='store_true', default=False, help='Boolean value indicating whether to cast down the image if possible (default: False)')
    args = parser.parse_args()
    
    if args.table:
        print("\nDifferent options for reduced datatypes and their meaning:")
        print("----------------------------------------")
        print("{:<6} | {:<24} {d}".format("uint8","8 bit unsigned integer:", d="[0 ; 255]"))
        print("{:<6} | {:<24} {d}".format("int8","8 bit signed integer:", d="[-128 ; 127]"))
        print("{:<6} | {:<24} {d}".format("uint16","16 bit unsigned integer:", d="[0 ; 65535]"))
        print("{:<6} | {:<24} {d}".format("int16","16 bit signed integer:", d="[-32768 ; 32767]"))
        print("{:<6} | {:<24} {d}".format("uint32","32 bit unsigned integer:", d="[0 ; 4294967295]"))
        print("{:<6} | {:<24} {d}".format("int32","32 bit signed integer:", d="[-2147483648 ; 2147483647]"))
        print("{:<6} | {:<24} {d}".format("uint64","64 bit unsigned integer:", d="[0 ; 18446744073709551616]"))
        print("{:<6} | {:<24} {d}".format("int64","64 bit signed integer:", d="[-9223372036854775808 ; 9223372036854775807]"))
    
    else: 
        
        if args.path is None or args.dtype is None:
            print("ERROR: you should specify a path and reduced data type")
            sys.exit(1)
        CheckDatatypes(args.path, args.dtype, args.cast)

if __name__ == "__main__":
    main()

