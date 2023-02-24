# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:26:15 2023

@author: u0139075
"""

from __future__ import print_function

import SimpleITK as sitk
import sys
import os

if len(sys.argv) == 1:
    print("Not enough input arguments! \nUsage: python DicomTo3D.py <input_directory> <output_file> ")
    sys.exit(1)
elif len(sys.argv) < 3:
    if sys.argv[1] == "--help":
        print("script to read a series of dicom files into a volume and write the volume as a ITK compatible .mha-file.")
        print("Usage: python DicomTo3D.py <input_directory> <output_file>")
        sys.exit(0)
    sys.exit(1)
        
data_dir = sys.argv[1]
if not os.path.exists(data_dir):
    print("ERROR: the path {} does not seem to exist".format(data_dir))
    sys.exit() 
    
series_reader = sitk.ImageSeriesReader()
series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_dir)
series_reader.SetFileNames(series_file_names)

#Conserve metadata
print('Reading in data...')
series_reader.MetaDataDictionaryArrayUpdateOn()
series_reader.LoadPrivateTagsOn()
series_reader.SetImageIO('GDCMImageIO')

print("\n__________________READER INFO________________\n")
print(series_reader)
image3D = series_reader.Execute()
print('Reading done. Image size: ',image3D.GetSize())
print('Image Pixel type: ',image3D.GetPixelIDTypeAsString())

#Isotropic pixel size
singleSlice = series_file_names[0]
sizeReader = sitk.ImageFileReader()
sizeReader.SetFileName(singleSlice)
sizeReader.ReadImageInformation()
dz = float(sizeReader.GetMetaData('0018|0050'))
dz = dz/2

dx, dy = image3D.GetSpacing()[0], image3D.GetSpacing()[1]
image3D.SetSpacing((dx,dy,dz))
direction = sizeReader.GetDirection()
image3D.SetDirection(direction)
print("\n__________________PIXEL SETTINGS______________\n")
print("Pixel sizes are retrieved from DICOM metadata:",(dx,dy,dz))


print("\n__________________IMAGE INFO________________\n")
print(image3D)
#write image
writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()
writer.SetFileName(sys.argv[2])
writer.SetImageIO('MetaImageIO')
writer.Execute(image3D)
print("\n__________________WRITER INFO________________\n")
print(writer)