# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:45:26 2023

@author: u0139075
"""

import numpy as np
import SimpleITK as sitk
import os
import argparse
from SetOrigin import CalculateCOG

def CalculateInertiaTensor(sitkImage, cog):
    
    """
    Calculate a numpy Tensor array with the moments of inertia according to image coordinate system.
    
    Inputs:
        - sitkImage: mask SimpleITK image. Background is taken as 0
        - cog: iterable with coordinates of centre of geometry (x,y,z) expressed in world coordinate system
        
    Outputs:
        - numpy array with moments of inertia
        - numpy matrix 3xN with the coordinates of all the mask voxels
    """
    
    (cog_x, cog_y, cog_z) = cog
    
    array = sitk.GetArrayFromImage(sitkImage)
    
    print("Finding coordinates of non-empty pixels...")
    (depths,rows,cols) = np.where(array!=0)
    unique_depths = np.unique(depths)
    unique_rows = np.unique(rows)
    unique_cols = np.unique(cols)
    
    #For not having to loop over all the white pixels, only transform each relevant coordinate one time, and store the corresponding values in a dictionary
    print("Converting index to physical points...")
    x_dict = {}
    y_dict = {}
    z_dict = {}
    
    for xi in unique_cols:
        x_dict[xi] = sitkImage.TransformContinuousIndexToPhysicalPoint((int(xi),0,0))[0]
    for yi in unique_rows:
        y_dict[yi] = sitkImage.TransformContinuousIndexToPhysicalPoint((0,int(yi),0))[1]
    for zi in unique_depths:
        z_dict[zi] = sitkImage.TransformContinuousIndexToPhysicalPoint((0,0,int(zi)))[2]
    
    print("Setting up x,y,z")
    mass=1
    x = np.array([x_dict[i] for i in cols])
    y = np.array([y_dict[i] for i in rows])
    z = np.array([z_dict[i] for i in depths])
    
    x -= cog_x
    y -= cog_y
    z -= cog_z
            
    xx = x*x
    yy = y*y
    zz = z*z
    Ixx = np.sum(mass*(yy + zz))
    print(f"Ixx calculated: {Ixx}")
    Iyy = np.sum(mass*(xx + zz))
    print(f"Iyy calculated: {Iyy}")
    Izz = np.sum(mass*(yy + xx))
    print(f"Izz calculated: {Izz}")
    
    Ixy = -np.sum(mass*x*y)
    print(f"Ixy calculated: {Ixy}")
    Ixz = -np.sum(mass*x*z)
    print(f"Ixz calculated: {Ixz}")
    Iyz = -np.sum(mass*y*z)
    print(f"Iyz calculated: {Iyz}")
    
    I = np.array([[Ixx,Ixy,Ixz], [Ixy,Iyy,Iyz], [Ixz,Iyz,Izz]])
    Coordinates = np.array([x,y,z])

    
    return I, Coordinates


def switchEigenpairs(v,w,i,l):
    v[i],v[l] = v[l],v[i]
    w[:,[i,l]] = w[:,[l,i]]
    
def checkRHS(w):
    vec3 = np.round(w[:,2],4)
    cross = np.round(np.cross(w[:,0], w[:,1]), 4)
    return (vec3 == cross).all()

def MatrixToList(R):
    return [R[i,j] for i in range(0,3) for j in range(0,3)]

def MatrixToString(R):
    s = ''
    for i in range(0,3):
        for j in range(0,3):
            s += str(R[i,j])+' '
    return s


def RegisterMask(fix, mov, outputFolder, DEBUG = False, MIRROR_ORIENT = True):
    
    """
    This function calculates a rigid registration to transform the moving image onto the fixed image.
    It is implemented for mask images, and serves as a preprocessing step in a full registration workflow.
    Alignment is done by calculating and matching the principal axis of inertia.
    Measures are implemented such that the correct axes match in both image domain, and that the vectors point in the same direction.
    This is done by looking at the projections of the masks on the inertia axes and by exploiting the right hand coordinate system definition.
    
    Inputs:
        - fix: SimpleITK image. Image voxels are white (255) and background voxels black (0). Image that is not transformed
        - mov: SimpleITK image. Idem as fix. Image that is to be transformed onto fixed image.
        - outputFolder: string with path to folder to store output. 
        - DEBUG: boolean value. If initial results are not correct, try running again with DEBUG=True .
        - MIRROR_ORIENT: boolean value. If the two images domain have flipped the image with respect to another, set True. (This is the case for Bruker microCT - clinical CT)
    
    Output:
        - a parameter text file namd <MaskRegistrationParam.txt> is written in the outputFolder to execute the transformation with transformix. To execute the transformation, run from command line:
            >> transformix -in <path_to_moving_image> -out <outputFolder> -tp <outputFolder>/MaskRegistrationParam.txt
    """
        
    arr_fix = sitk.GetArrayViewFromImage(fix)
    arr_mov = sitk.GetArrayViewFromImage(mov)
    size_fix = len(np.where(arr_fix != 0)[0]) * fix.GetNumberOfComponentsPerPixel() * fix.GetSizeOfPixelComponent()
    size_mov = len(np.where(arr_mov != 0)[0]) * mov.GetNumberOfComponentsPerPixel() * mov.GetSizeOfPixelComponent()
    
    # make sure moving image lies in the array extent of fixed image, and keep track of original origin
    original_origin = mov.GetOrigin()
    new_origin = fix.GetOrigin()
    mov.SetOrigin(new_origin)
    
    # Calculation of inertia tensor can be computationally heavy, so it gets done in the smallest domain size
    if size_mov >= size_fix:
        filtI = sitk.ResampleImageFilter()
        filtI.SetInterpolator(sitk.sitkNearestNeighbor)
        filtI.SetReferenceImage(fix)
        filtI.SetOutputOrigin(mov.GetOrigin()) # avoid unwanted translations
        mov_f = filtI.Execute(mov)
                        
        cog_mov = CalculateCOG(mov_f)
        cog_fix = CalculateCOG(fix)

        mov_I, mov_Mask = CalculateInertiaTensor(mov_f, cog_mov)
        fix_I, fix_Mask = CalculateInertiaTensor(fix, cog_fix)
    
    else:
        filtI = sitk.ResampleImageFilter()
        filtI.SetInterpolator(sitk.sitkNearestNeighbor)
        filtI.SetReferenceImage(mov)
        filtI.SetOutputOrigin(fix.GetOrigin()) # avoid unwanted translations
        fix_f = filtI.Execute(fix)
        
        cog_mov = CalculateCOG(mov)
        cog_fix = CalculateCOG(fix_f)

        mov_I, mov_Mask = CalculateInertiaTensor(mov, cog_mov)
        fix_I, fix_Mask = CalculateInertiaTensor(fix_f, cog_fix)
    
    [v_mov, w_mov] = np.linalg.eig(mov_I)
    [v_fix, w_fix] = np.linalg.eig(fix_I)
    
    
    # eigenvectors and -values are not automatically sorted in decreasing order
    i_mov = np.abs(v_mov).argsort()[::-1]
    v_mov = v_mov[i_mov]
    w_mov = w_mov[:, i_mov]
    
    i_fix = np.abs(v_fix).argsort()[::-1]
    v_fix = v_fix[i_fix]
    w_fix = w_fix[:, i_fix]
    
    # eigenvectors can be inverted and corresponding vectors can thus be pointing in opposite direction
    
    #Step 1: make sure fixed image has right hand coordinate system with eigenvectors
    RHS = checkRHS(w_fix)
    if not RHS:
        print('fixed image eigenvectors are not right hand system. Inverting third vector')
        w_fix[:,2] *= -1
    
    # Step 2: make sure first and second eigenvector point in same direction:
    #         project coordinates on vector and look at maximum in both directions
    edge_projections_mov = np.zeros((3,2))
    edge_projections_fix = np.zeros((3,2))
    for nbVector in range(0,3): #only for first two vectors
        proj_mov = np.matmul(np.transpose(mov_Mask), w_mov[:,nbVector])
        proj_fix = np.matmul(np.transpose(fix_Mask), w_fix[:,nbVector])
        
        max_mov, min_mov = np.max(proj_mov), np.min(proj_mov)
        max_fix, min_fix = np.max(proj_fix), np.min(proj_fix)
        
        max_edge_mov = len(np.where(proj_mov > 0.75 * max_mov)[0])
        min_edge_mov = len(np.where(proj_mov < 0.75 * min_mov)[0])
        max_edge_fix = len(np.where(proj_fix > 0.75 * max_fix)[0])
        min_edge_fix = len(np.where(proj_fix < 0.75 * min_fix)[0])
        
        edge_projections_mov[nbVector,:] = max_edge_mov, min_edge_mov
        edge_projections_fix[nbVector,:] = max_edge_fix, min_edge_fix
    
    sorted_size_differences = np.abs(edge_projections_mov[:,0] - edge_projections_mov[:,1]).argsort()[::-1]
    
    if DEBUG:
        print('Debugging mode...')
        print('These are the current number of voxels on the 25% edge regions of projections onto eigenvectors: (rows=vector, cols=(max min)')
        print('MOVING IMAGE')
        print(edge_projections_mov)
        print()
        print('FIXED IMAGE')
        print(edge_projections_fix)
        print()
        print(f'Vector orientation was based on eigenvectors {sorted_size_differences[0]} and {sorted_size_differences[1]}')
        print(f'Changing to eigenvectors {sorted_size_differences[0]} and {sorted_size_differences[2]}')
        print('\nPlease run transformix with new parameters. If results are still incorrect, it is advised to manually check if your mask is well defined for the bone geometry')
        
        sorted_size_differences[1], sorted_size_differences[2] = sorted_size_differences[2], sorted_size_differences[1]
        
    for i in range(2):
        nbVector = sorted_size_differences[i]
        if edge_projections_fix[nbVector,:].argmax() != edge_projections_mov[nbVector,:].argmax():
            print(f'Inverting eigenvector {nbVector} in moving image')
            w_mov[:,nbVector] *= -1
    
    # Step 3: make sure moving image correctly oriented coordinate system with eigenvectors
    # !! images can be flipped in orientation, then RHS matches LHS
    RHS = checkRHS(w_mov)
    system = 'left hand system' if MIRROR_ORIENT else 'right hand system'
    if RHS == MIRROR_ORIENT: # if mirrored, moving must be LHS, if not mirrored moving must be RHS
        nbVector = sorted_size_differences[2]
        print(f'moving image eigenvectors are not {system}. Inverting eigenvector {nbVector}')
        w_mov[:,nbVector] *= -1
 
    
    # create output folder
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    
    if not DEBUG:
        # SimpleITK convention:
        # for coordinate matrix M = [c1 c2 ... cn]
        # transpose(M) * A = transpose(M')
        
        Rtot = np.matmul(w_mov,np.transpose(w_fix))

        
    if DEBUG:
        print('Debug mode...')
        
        # finding the main direction of each eigenvector
        unitmov = np.zeros((3,3))
        unitfix = np.zeros((3,3))
        for k in range(0,3):
            maxMOV = np.where(abs(w_mov[:,k]) == np.max(abs(w_mov[:,k])))[0][0]
            maxFIX = np.where(abs(w_fix[:,k]) == np.max(abs(w_fix[:,k])))[0][0]
            unitmov[maxMOV,k] = np.sign(w_mov[maxMOV,k])
            unitfix[maxFIX,k] = np.sign(w_fix[maxFIX,k])
            
        
        # --- Series of operation ---
        # 1) First rotation: principle axes moving image to main axes
        Rmov2x = np.matmul(w_mov,np.transpose(unitmov))

        # 2) Intermediate flip defined with input argument

        # 3) Aligning corresponding principal axes (can be flip based)
        Rx2x = np.matmul(unitmov, np.transpose(unitfix))
        
        # 4) Intermediate flip defined with input argument
        
        # 5) Second rotation: main axes to principle axes fixed image
        Rx2fix = np.matmul(unitfix, np.transpose(w_fix))
        
        # -- Resample filter
        filt = sitk.ResampleImageFilter()
        filt.SetInterpolator(sitk.sitkNearestNeighbor)
        filt.SetReferenceImage(fix)
        
        # -- Transformation setup
        MOV2X = sitk.AffineTransform(3)
        MOV2X.SetMatrix(MatrixToList(Rmov2x))
        
        FLIP1 = sitk.AffineTransform(3)
        FLIP1.SetMatrix(MatrixToList(FlipMatrixMoving))
        
        X2X = sitk.AffineTransform(3)
        X2X.SetMatrix(MatrixToList(Rx2x))
        
        FLIP2 = sitk.AffineTransform(3)
        FLIP2.SetMatrix(MatrixToList(FlipMatrixFixed))
        
        X2FIX = sitk.AffineTransform(3)
        X2FIX.SetMatrix(MatrixToList(Rx2fix))
        
        # -- Intermediate results
        print('Writing intermediate results...')
        # 1) First rotation
        filt.SetTransform(MOV2X)
        step_mov2x = filt.Execute(mov)
        sitk.WriteImage(step_mov2x, os.path.join(outputFolder, 'step1_mov2x.mha'))
        
        # 2) input flip
        filt.SetTransform(FLIP1)
        step_flip1 = filt.Execute(step_mov2x)
        sitk.WriteImage(step_flip1, os.path.join(outputFolder, 'step2_flip1.mha'))
        
        # 3) Unit axes alignment
        filt.SetTransform(X2X)
        step_x2x = filt.Execute(step_flip1)
        sitk.WriteImage(step_x2x, os.path.join(outputFolder, 'step3_x2x.mha'))
        
        # 4) input flip 2
        filt.SetTransform(FLIP2)
        step_flip2 = filt.Execute(step_x2x)
        sitk.WriteImage(step_flip2, os.path.join(outputFolder, 'step4_flip2.mha'))
        
        # 5) Second rotation
        filt.SetTransform(X2FIX)
        step_x2fix = filt.Execute(step_flip2)
        sitk.WriteImage(step_x2fix, os.path.join(outputFolder, 'step5_x2fix.mha'))
        
        
        # -- Total matrix
        Rtot = np.matmul(Rmov2x, np.matmul(FlipMatrixMoving, np.matmul(Rx2x, np.matmul(FlipMatrixFixed, Rx2fix))))
        
    # end debug
    
    print('Writing transformation parameter file.')
    # writing parameter file for transformation
    Rot_string = MatrixToString(Rtot)
    translation1 = -(np.asarray(new_origin) - np.asarray(original_origin))
    translation2 = -(cog_fix - cog_mov) # I find these negative signs counterintuitive, but that's apparatnly how it works...
    translation = translation1 + translation2
    
    Trans_string = f'{translation[0]} {translation[1]} {translation[2]}'
    
    
    
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    
    f = open(outputFolder + '/MaskRegistrationParam.txt','w')
    f.write('(Transform "AffineTransform")\n')
    f.write('(NumberOfParameters 12)\n')
    f.write(f'(TransformParameters {Rot_string} {Trans_string})\n')
    f.write('(InitialTransformParametersFileName "NoInitialTransform")\n')
    f.write('(HowToCombineTransforms "Compose")\n')
    f.write('(FixedImageDimension 3)\n')
    f.write('(MovingImageDimension 3)\n')
    f.write('(FixedInternalImagePixelType "short")\n')
    f.write('(MovingInternalImagePixelType "short")\n')
    
    (sx,sy,sz) = fix.GetSize()
    f.write(f'(Size {sx} {sy} {sz})\n')
    f.write('(Index 0 0 0)\n')
    
    (dx,dy,dz) = fix.GetSpacing()
    f.write(f'(Spacing {dx} {dy} {dz})\n')
    
    (ox,oy,oz) = fix.GetOrigin()
    f.write(f'(Origin {ox} {oy} {oz})\n')
    
    (d1,d2,d3,d4,d5,d6,d7,d8,d9) = fix.GetDirection()
    f.write(f'(Direction {d1} {d2} {d3} {d4} {d5} {d6} {d7} {d8} {d9})\n')
    
    f.write('(UseDirectionCosines "true")\n')
    f.write(f'(CenterOfRotationPoint {cog_fix[0]} {cog_fix[1]} {cog_fix[2]})\n')
    f.write('(ResampleInterpolator "FinalNearestNeighborInterpolator")\n')
    f.write('(Resampler "DefaultResampler")\n')
    f.write('(DefaultPixelValue 0.000000)\n')
    f.write('(ResultImageFormat "mha")\n')
    f.write('(ResultImagePixelType "short")\n')
    f.write('(CompressResultImage "false")\n')
    f.close()
    print('Done!')
    
def main():
    parser = argparse.ArgumentParser(description = \
                                     'Script to perform rigid registration on two full mask files. '\
                                     'Registration is performed based on the principal axes of inertia. '\
                                     '\nPrecautions have been taken to make sure all principal axes are corresponding and pointing in the same direction. '\
                                     'However, the script may fail. If so, use the debug mode to check intermediate results.')
    parser.add_argument('-fix', help='Fixed image', required=True)
    parser.add_argument('-mov', help='Moving image', required=True)
    parser.add_argument('-out', help='Output folder to store parameter files and any intermediate results', required=True)
    parser.add_argument('-debug', help='Execute script in debug mode. Intermediate image results will be written. Using the API two matrices can be given as input to perform intermediate flippings, but this is not possible via the command line!', action='store_true', default=False)

    args = parser.parse_args()
    
    if not os.path.isfile(args.fix) or not os.path.isfile(args.mov):
        raise RuntimeError('ERROR: you did not specify a valid file for the fixed and/or moving image.')
    
    else:
        print('Reading in images...')
        fixedIm = sitk.ReadImage(args.fix)
        movingIm = sitk.ReadImage(args.mov)
        
        RegisterMask(fixedIm, movingIm , args.out, DEBUG = args.debug)

if __name__ == '__main__':
    main()

