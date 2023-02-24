# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:51:37 2023

@author: u0139075
"""

import SimpleITK as sitk
import sys
import numpy as np
import os

if sys.argv[1] == '--help':
    print("script to check if you can cast down your image without losing information")
    print("\nUsage: python CheckDatatypes.py <link_to_image> <reduced data type> <CAST_Boolean>=0")
    print("The CAST_Boolean is a 0/1 bit, where 1 means that if the casting does not lead to information loss, it will execute it.\
          If no argument is given, it will by default not be done.")
    
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
    
    sys.exit(0)

path_string = sys.argv[1]
dtype_string = sys.argv[2]
if len(sys.argv)==4:
    CAST = int(sys.argv[3])
else:
    CAST = 0
    
im = sitk.ReadImage(path_string)
arr = sitk.GetArrayViewFromImage(im)
minval = np.min(arr)
maxval = np.max(arr)

reduced_dtype = np.dtype(dtype_string)
info = np.iinfo(reduced_dtype)
MAX = info.max
MIN = info.min



print("max value image: {}".format(maxval))
print("min value image: {}".format(minval))
print("maximal value using {}: {}".format(sys.argv[2], MAX))
print("minimal value using {}: {}".format(sys.argv[2], MIN))
if maxval <= MAX and minval >= MIN:
    print("You can reduce the bits per voxel without losing image information")
    if CAST:
        print("Casting down the image...")
        if dtype_string.find('u') == -1:       # signed integer to be translated to sitk syntax
            sitk_string = 'sitkI'+ dtype_string[1:]
        else:                                   # unsigned integer to be translated to sitk syntax
            sitk_string = 'sitkUI'+ dtype_string[2:]
        
        sitk_dtype = sitk.GetPixelIDValueFromString(sitk_string)
        im_cast = sitk.Cast(im, sitk_dtype)
    
        im_newName = path_string[:-4] + '_' + dtype_string + path_string[-4:]
        sitk.WriteImage(im_cast, im_newName)
        

else:
    print("WARNING: if you reduce the bits per voxel, you will lose image information!!")