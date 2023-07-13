# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:01:31 2023

@author: u0139075
"""

import SimpleITK as sitk
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def update_axial(val):
    slice_nb = int(slice_slider1.val)
    ax[0].imshow(arr[slice_nb,:,:], cmap='gray')
    
def update_sagital(val):
    slice_nb = int(slice_slider2.val)
    ax[1].imshow(arr[:,slice_nb,:], cmap='gray')

def button_press(event):
    if event.inaxes == axbut1:
        coords_axial.clear()
        coords_axial.append((0,0))
        coords_axial.append((0,np.shape(arr)[1]))
        coords_axial.append((np.shape(arr)[2], 0))
        coords_axial.append((np.shape(arr)[2], np.shape(arr)[1]))
        print(f'default values axial bounding box \n {coords_axial}')
        
    elif event.inaxes == axbut2:
        coords_sagital.clear()
        coords_sagital.append(0)
        coords_sagital.append(np.shape(arr)[0])
        print(f'default values sagital bounding box \n {coords_sagital}')
    
    # handling the difference Spyder and command line execution
    check_close()
    
def onclick(event):
    if event.inaxes == ax[0] and len(coords_axial) != 4:
        ix, iy = event.xdata, event.ydata
        coords_axial.append((ix,iy))
        print(f'{ix} , {iy}')
    
    elif event.inaxes == ax[1] and len(coords_sagital) != 2:
        iz = event.ydata
        coords_sagital.append(iz)
        print(f'{iz}')
    
    # handling the difference Spyder and command line execution
    check_close()

def check_close():
    if len(coords_axial) == 4 and len(coords_sagital) == 2:
        plt.close('all')

def getCoordinates():    
    global coords_axial
    global coords_sagital
    global fig
    global ax
    global axbut1
    global axbut2
    global slice_slider1
    global slice_slider2

    coords_axial = []
    coords_sagital = []
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(arr[0,:,:], cmap='gray')
    ax[0].set_title('click on 4 bounding coordinates')
    ax[1].imshow(arr[:,0,:], cmap='gray')
    ax[1].set_title('click on the remaining 2 bounding coordinates (height)')
    
    plt.subplots_adjust(bottom=0.25)   
     # Make a horizontal slider to control the slice nb.
    axslice1 = plt.axes([0.25, 0.1, 0.4, 0.03])
    slice_slider1 = Slider(
         ax=axslice1,
         label='Axial',
         valmin= 0,
         valmax=np.shape(arr)[0]-1,
         valinit=0,
     )
    axslice2 = plt.axes([0.25, 0.02, 0.4, 0.03])
    slice_slider2 = Slider(
         ax=axslice2,
         label='Sagital',
         valmin= 0,
         valmax=np.shape(arr)[1]-1,
         valinit=0,
     )
    axbut1 = plt.axes([0.7, 0.1, 0.15, 0.07])
    button1 = Button(axbut1, 'Defaults')
    axbut2 = plt.axes([0.7, 0.02, 0.15, 0.07])
    button2 = Button(axbut2, 'Defaults')
    
    button1.on_clicked(button_press)
    button2.on_clicked(button_press)
    
     # register the update function with each slider
    slice_slider1.on_changed(update_axial)
    slice_slider2.on_changed(update_sagital)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show() 
    
    # command line execution pauses here automatically
    # Spyder execution does not pause here automatically
   
    # handling the Spyder execution  
    while len(coords_axial) < 4 or len(coords_sagital) < 2:
         plt.pause(0.1)
    
    check_close()
    fig.canvas.mpl_disconnect(cid)
    print('Bounding box completed')
     
    

def CropImageManually(image, WRITE_IMAGE = False, PRESERVE_DIMENSIONS = False, outIm = ''):
    global arr
   
    arr = sitk.GetArrayViewFromImage(image)

    getCoordinates()


    rows = [int(i[1]) for i in coords_axial]
    cols = [int(i[0]) for i in coords_axial]
    depths = [int(i) for i in coords_sagital]
    
    min_x, max_x = np.min(cols), np.max(cols)
    min_y, max_y = np.min(rows), np.max(rows)
    min_z, max_z = np.min(depths), np.max(depths)

    if PRESERVE_DIMENSIONS:
        out_im = image
        out_im[:min_x, :, :] = 0
        out_im[max_x:, :, :] = 0
        out_im[:, :min_y, :] = 0
        out_im[:, max_y:, :] = 0
        out_im[:, :, :min_z] = 0
        out_im[:, :, max_z:] = 0
        
    else:
        out_im = image[min_x:max_x, min_y:max_y, min_z:max_z]

    if WRITE_IMAGE:
        print(f'Writing image to {outIm}...')
        sitk.WriteImage(out_im, outIm)
        print('Done')
    
    return out_im


def CropImage(image, roi, margin):
    """
    Crop a large image such that only the region of interest and a predefined margin remain.
    Image and ROI should be SimpleITK images.
    The margin is specified in milimeters.
    Use the flag WRITE_IMAGE if you want to write the cropped image and ROI file. If so, provide names for the output images.

    """
    
    roi_array = sitk.GetArrayViewFromImage(roi)
    roi_bool = roi_array != 0
    (depths, rows, cols) = np.where(roi_bool == 1)
    
    min_x, max_x = np.min(cols), np.max(cols)
    min_y, max_y = np.min(rows), np.max(rows)
    min_z, max_z = np.min(depths), np.max(depths)
    
    (dx, dy, dz) = image.GetSpacing()
    (sx, sy, sz) = image.GetSize()
    
    Dx = np.ceil(margin/dx)
    Dy = np.ceil(margin/dy)
    Dz = np.ceil(margin/dz)
    
    min_x = int(max(0, min_x - Dx))
    min_y = int(max(0, min_y - Dy))
    min_z = int(max(0, min_z - Dz))
    max_x = int(min(sx - 1, max_x + Dx))
    max_y = int(min(sy - 1, max_y + Dy))
    max_z = int(min(sz - 1, max_z + Dz))
    
    image_crop = image[min_x:max_x, min_y:max_y, min_z:max_z]
    roi_crop = roi[min_x:max_x, min_y:max_y, min_z:max_z]

    return image_crop, roi_crop
    

def main():
    parser = argparse.ArgumentParser(description = 'Script to crop large images to only a specified region of interest in the mask.\n'\
                                                   'Maximum positions in the mask are determined, and a margin is included.', \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-image', help='input image to be cropped', required = True)
    parser.add_argument('-manual', help='manually select a boundary box in the image. You only have to provide the input image and output image name', action='store_true', default=False)
    parser.add_argument('-manual_oversized', help='same as manual mode, but original image dimensions are preserved and padded with zeros', action='store_true', default=False)
    parser.add_argument('-mask', help='region of interest. Dimensions should match the image. You can enter multiple ROIs, but give the same amount of output names', nargs='+')
    parser.add_argument('-margin', help='Margin beyond the extrema of the region of interest to crop. Should be specified in milimeters.', default=10)
    parser.add_argument('-outImg', help='Name of output image. You can choose to specify a full path or just a filename.'\
                                        'In case of the latter, the files will be written in the same folder as the inputs', nargs='+')
    parser.add_argument('-outMask', help='Name of output mask. You can choose to specify a full path or just a filename.'\
                                        'In case of the latter, the files will be written in the same folder as the inputs', nargs='+')
    args = parser.parse_args()
    
    # check if input image exists
    if not os.path.exists(args.image):
        raise ValueError('ERROR: you did not specify a valid image')
        
        
    #base strings of image and mask
    pos = args.image.rfind('/')
    base_im = args.image[:pos+1]
    
    if args.manual:
        print('Manually cropping image')
        print('Reading image...')
        im = sitk.ReadImage(args.image)
        outImg = args.outImg[0]
        if outImg.find('/') == -1:
            # image output has to be stored in same folder as input image
            outImg = base_im + outImg
        if outImg.find('.') == -1:
            outImg += '.mha'
        
        CropImageManually(im, WRITE_IMAGE = True, PRESERVE_DIMENSIONS = False, outIm = outImg)
    
    elif args.manual_oversized:
        print('Manually cropping image')
        print('Reading image...')
        im = sitk.ReadImage(args.image)
        outImg = args.outImg[0]
        if outImg.find('/') == -1:
            # image output has to be stored in same folder as input image
            outImg = base_im + outImg
        if outImg.find('.') == -1:
            outImg += '.mha'
        
        CropImageManually(im, WRITE_IMAGE = True, PRESERVE_DIMENSIONS= True, outIm = outImg)

    else:

        if len(args.mask) != len(args.outMask) or len(args.mask) != len(args.outImg):
            raise ValueError('ERROR: you specified a list with multiple ROIs, but not the same number of output names.')
            
        set_masks = args.mask
        set_outMasks = args.outMask
        set_outImgs = args.outImg
        print(f'{len(set_masks)} mask files are given as input')
                
        print('Reading image...')
        img = sitk.ReadImage(args.image)
        
        for mask_nb in range(len(set_masks)):
            mask = set_masks[mask_nb]
            outMask = set_outMasks[mask_nb]
            outImg = set_outImgs[mask_nb]
            print(f'mask {mask_nb+1}/{len(set_masks)}')
            
        
            if not os.path.isfile(mask):
                print(f'{mask} is not a valid file. Skipping this mask.')
                
            else:
                if outImg.find('/') == -1:
                    # image output has to be stored in same folder as input image
                    outImg = base_im + outImg
                    
                if outMask.find('/') == -1:
                    # mask output has to be stored in same folder as input image
                    pos = mask.rfind('/')
                    outMask = mask[:pos+1] + outMask
                
                # check if extension is explicitly mentioned
                if outImg.find('.') == -1:
                    outImg += '.mha'
                if outMask.find('.') == -1:
                    outMask += '.mha'
    
                print('Reading mask...')
                mask = sitk.ReadImage(mask)
                
                img_crop, roi_crop = CropImage(img, mask, float(args.margin))
                print(f'Writing cropped image to {outImg}')
                sitk.WriteImage(img_crop, outImg)
                print(f'Writing cropped mask to {outMask}')
                sitk.WriteImage(roi_crop, outMask)
                print('Done')
                
if __name__ == '__main__':
    main()
        
     