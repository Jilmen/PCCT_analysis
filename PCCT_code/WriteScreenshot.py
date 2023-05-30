# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:41:59 2023

@author: u0139075
"""

import SimpleITK as sitk
import numpy as np
from PIL import Image

def Normalize(arr):
    dtype = arr.dtype
    if dtype == np.dtype('int8'):
        arr = (arr - np.min(arr)) * 255 / (np.max(arr) - np.min(arr))
        arr = arr.astype(np.uint8)
        bits = 8
    elif dtype == np.dtype('int16'):
        arr = (arr - np.min(arr)) * 65535 / (np.max(arr) - np.min(arr))
        arr = arr.astype(np.uint16)
        bits = 16
    elif dtype != np.dtype('uint8') and dtype != np.dtype('uint16'):
        raise ValueError(f'Function only implemented for 8 bit and 16 bit integer representations. Input image is of {dtype} pixel type.')
    
    return arr, bits

def WriteScreenshot(im, out, x=None, y=None, z=None):
    
    if out.find('.') != -1:
        out = out[:out.find('.')]
    
    if x is not None:
        screenshot_x = im[x, : ,:]
        arr = sitk.GetArrayFromImage(screenshot_x)
        arr, bits = Normalize(arr)
        imx = Image.fromarray(arr)
        imx.save(out+'_x.png', 'PNG', bitdepth=bits)
    
    if y is not None:
        screenshot_y = im[:, y ,:]
        arr = sitk.GetArrayFromImage(screenshot_y)
        arr, bits = Normalize(arr)
        imy = Image.fromarray(arr)
        imy.save(out+'_y.png', 'PNG', bitdepth=bits)
    
    if z is not None:
        screenshot_z = im[:, :, z]
        arr = sitk.GetArrayFromImage(screenshot_z)
        arr, bits = Normalize(arr)
        imz = Image.fromarray(arr)
        imz.save(out+'_z.png', 'PNG', bitdepth=bits)
        
    
    
    
   
            
        
    