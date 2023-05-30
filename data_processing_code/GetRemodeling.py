# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:21:37 2023

@author: u0139075
"""

import SimpleITK as sitk
import os
import numpy as np
import sys
import concurrent.futures
from tqdm import tqdm
import argparse

def ClusterSizeMap(grid):
    
    [ii, jj, kk] = np.where(grid==True) 
    N = len(ii)
    print(f'{N}/{np.size(grid)} voxels')

    shape = np.shape(grid)
    size_map = np.zeros(shape, dtype=np.uint8)
    grid = np.stack((grid, np.ones(shape, dtype=bool)), axis = 3)
    grid[ii,jj,kk, 1] = False
    
    def check_and_add_to_cluster(i,j,k):
        CAN_ADD = True
        if grid[i,j,k,1] == True: #voxel already processed  
            CAN_ADD = False
            
        elif i < 0 or i >= shape[0] or j < 0 or j >= shape[1] or k < 0 or k >= shape[2]: #out of bounds
            CAN_ADD = False
            
        elif grid[i,j,k,0] == False: # not in 'True' cluster
            CAN_ADD = False
        
        if CAN_ADD:
            grid[i,j,k,1] = True #set voxel to processed
            cluster_growth.append((i,j,k))
            
    for vox in tqdm(range(N)):
        i = ii[vox]
        j = jj[vox]
        k = kk[vox]

        if grid[i,j,k,1] == False: # voxel not yet processed
            cluster = [(i,j,k)] #initialize cluster
            grid[i,j,k,1] = True
            
            cluster_growth = []
            while cluster: # runs until cluster coordinates list is empty
                for (posi, posj, posk) in cluster:
                    for di, dj, dk in [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]:
                        check_and_add_to_cluster(posi+di, posj+dj, posk+dk)
                
                if len(cluster_growth) > 0:
                    cluster.extend(cluster_growth)
                    cluster_growth.clear()
                else:
                    size = len(cluster)
                    for si,sj,sk in cluster:
                        size_map[si,sj,sk] = size
                    cluster.clear()
    return size_map
                            

def GetReproducability(im1, im2, mask, calibration_intercept, calibration_slope, log = False, PRINT = True, identifier = ''):
    arr1 = sitk.GetArrayViewFromImage(im1)
    arr2 = sitk.GetArrayViewFromImage(im2)
    mask_arr = sitk.GetArrayViewFromImage(mask)
    
    [d_mask , r_mask, c_mask] = np.where(mask_arr != 0)
    N = len(d_mask)
    
    diff = arr1 - arr2
    diffHU = diff[d_mask, r_mask, c_mask]
    # HU reproducibility
    mean_diff = np.mean(diffHU)
    sd_diff = np.std(diffHU)
    
    HU500 = (500-calibration_intercept)/calibration_slope
    mask500_arr = (mask_arr != 0) * (arr1 > HU500)
    [d500, r500, c500] = np.where(mask500_arr == True)
    mean_500 = np.mean(diff[d500, r500, c500])
    sd_500 = np.std(diff[d500, r500, c500])
    
    if PRINT:
        print(f'{len(d500)/N*100}% above 500mgHA')
        print(f'mean difference: {mean_diff}')
        print(f'SD difference: {sd_diff}')
        print(f'mean difference >500: {mean_500}')
        print(f'SD difference >500: {sd_500}')
    
    if log:
        file = open(log, 'a')
        file.write(f'{mean_diff}, {sd_diff}, {mean_500}, {sd_500}, {N}, {identifier} \n')

    

def GetRemodeling(im1, im2, mask, thresh, min_size_cluster, calibration_intercept, calibration_slope, log = False, PRINT = True, identifier = ''):
    '''
    Parameters
    ----------
    im1 : SimpleITK baseline image
    im2 : SimpleITK follow-up image / reproducability image
    mask : ROI
    thresh : Difference in bone mineral density [BMD mgHA/cm³] before formation/resorption is detected
    min_size_cluster : minimum number of adjacent voxels required to detect remodeling site
    calibration_intercept : intercept in formula BMD = intercept + slope * HU
    calibration_slope : slope in formula BMD = intercept + slope * HU
    log: text file to write output. Results are appended in the text file
    PRINT: bool to write results to output
    identifier: optional identifier (e.g. bone name) for in the logfile
    
    returns the detected bone formation and resorption, based on a threshold [mgHA/cm³] in the difference image.
    With bool
    '''
    
    arr1 = sitk.GetArrayViewFromImage(im1)
    arr2 = sitk.GetArrayViewFromImage(im2)
    mask_arr = sitk.GetArrayViewFromImage(mask)
    
    N = len(np.where(mask_arr != 0)[0])
    
    diff = arr1 - arr2
    diff_BMD = np.sign(diff) * (calibration_intercept + calibration_slope * np.abs(diff))
    
    formation_raw = (mask_arr != 0) * (diff_BMD >= thresh)
    resorption_raw = (mask_arr != 0) * (diff_BMD <= -thresh)

    print("Cluster map...")
    formation_cluster_size = ClusterSizeMap(formation_raw)
    resorption_cluster_size = ClusterSizeMap(resorption_raw)
    print("Cluster map calculated")
    
    formation_filtered = (mask_arr != 0) * (diff_BMD >= thresh) * (formation_cluster_size >= min_size_cluster)
    resorption_filtered = (mask_arr != 0) * (diff_BMD <= -thresh) * (resorption_cluster_size >= min_size_cluster)
    
    [d_fr, r_fr, c_fr] = np.where(formation_raw == True)
    [d_ff, r_ff, c_ff] = np.where(formation_filtered == True)
    [d_rr, r_rr, c_rr] = np.where(resorption_raw == True)
    [d_rf, r_rf, c_rf] = np.where(resorption_filtered == True)
    
    p_formation_raw = len(d_fr)/N*100
    p_formation_filt = len(d_ff)/N*100
    p_resorption_raw = len(d_rr)/N*100
    p_resorption_filt = len(d_rf)/N*100
    
    if PRINT:
        print("")
        print("{:<15} | {:<10} | {:<10}".format(f"t={thresh} mgHA/cm³" ,"formation","resorption"))
        print(41*"_")
        print("{:<15} | {:<10} | {:<10}".format("raw",f"{round(p_formation_raw,2)} %", f"{round(p_resorption_raw,2)} %"))
        print("{:<15} | {:<10} | {:<10}".format("filtered",f"{round(p_formation_filt,2)} %", f"{round(p_resorption_filt,2)} %"))
        print("")
    
    if log:
        file = open(log, 'a')
        file.write(f'{thresh}, {p_formation_raw}, {p_resorption_raw}, {p_formation_filt}, {p_resorption_filt}, {identifier} \n')

def main():
    parser  = argparse.ArgumentParser(description='Calculate formation and resorption based on HU difference image and calibration function.')
    parser.add_argument('-im1', help='SimpleITK image', required=True)
    parser.add_argument('-im2', help='SImpleITK image', required=True)
    parser.add_argument('-mask', help='Simple ITK image with ROI (255 valued)', required=True)
    parser.add_argument('-what', help='<remodeling> or <reproducability>', required =True)
    parser.add_argument('-thresh', help='Threshold (positive difference for formation, negative difference for resorption')
    parser.add_argument('-cluster', help='minimum cluster size for the filtered remodeling')
    parser.add_argument('-offset', help='intercept value for linear calibration function', required=True)
    parser.add_argument('-slope', help='slope value for linear calibration function', required=True)
    parser.add_argument('-logfile', help='output file to write to', default=False)
    parser.add_argument('-id', help='identifier printed in logfile', default='')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.im1) or not os.path.isfile(args.im2) or not os.path.isfile(args.mask):
        raise ValueError('Input is not a valid path to an image.')
    
    b = float(args.offset)
    a = float(args.slope)
   
    print('Reading images...')
    im1 = sitk.ReadImage(args.im1)
    im2 = sitk.ReadImage(args.im2)
    print('Reading mask...')
    mask = sitk.ReadImage(args.mask)
    
    if args.what == 'remodeling':
        try:
            t = int(args.thresh)
            K = int(args.cluster)
            GetRemodeling(im1, im2, mask, t, K, b, a, log=args.logfile, identifier=args.id)
        except Exception as e:
            raise ValueError(f'Incorrect input. Did you specify threshold and cluster?\n {e}')
    
    elif args.what == 'reproducability':
        GetReproducability(im1, im2, mask, b, a, log=args.logfile, PRINT=True, identifier=args.id)
        
        
    
if __name__ == '__main__':
    main()
        





