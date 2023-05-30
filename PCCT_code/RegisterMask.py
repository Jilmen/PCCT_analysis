# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:45:26 2023

@author: u0139075
"""

import numpy as np
import SimpleITK as sitk
import os
import argparse

def CalculateInertiaTensor(sitkImage):
    
    """
    Returns a numpy Tensor array with the moments of inertia according to image coordinate system.
    The image should have its center of geometry placed in the world coordinate system's origin.
    """
    
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


def RegisterMask(fix, mov, outputFolder, DEBUG = False, \
                 FlipMatrixMoving = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]), \
                 FlipMatrixFixed = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])):
    
    size_fix = fix.GetNumberOfPixels() * fix.GetNumberOfComponentsPerPixel() * fix.GetSizeOfPixelComponent()
    size_mov = mov.GetNumberOfPixels() * mov.GetNumberOfComponentsPerPixel() * mov.GetSizeOfPixelComponent()
    
    # Calculation of inertia tensor can be computationally heavy, so it gets done in the smallest domain size
    if size_mov >= size_fix:
        filtI = sitk.ResampleImageFilter()
        filtI.SetInterpolator(sitk.sitkNearestNeighbor)
        filtI.SetReferenceImage(fix)
        mov_f = filtI.Execute(mov)

        mov_I, mov_Mask = CalculateInertiaTensor(mov_f)
        fix_I, fix_Mask = CalculateInertiaTensor(fix)
    else:
        filtI = sitk.ResampleImageFilter()
        filtI.SetInterpolator(sitk.sitkNearestNeighbor)
        filtI.SetReferenceImage(mov)
        fix_f = filtI.Execute(fix)

        mov_I, mov_Mask = CalculateInertiaTensor(mov)
        fix_I, fix_Mask = CalculateInertiaTensor(fix_f)
    
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
    
    # Step 1: make sure fixed image has right hand coordinate system with eigenvectors
    RHS = checkRHS(w_fix)
    if not RHS:
        print('fixed image eigenvectors are not right hand system. Inverting third vector')
        w_fix[:,2] *= -1
    
    # Step 2: make sure first and second eigenvector point in same direction:
    #         project coordinates on vector and look at maximum in both directions
    for nbVector in range(0,2): #only for first two vectors
        proj_mov = np.matmul(np.transpose(mov_Mask), w_mov[:,nbVector])
        proj_fix = np.matmul(np.transpose(fix_Mask), w_fix[:,nbVector])
        
        max_mov, min_mov = np.max(proj_mov), np.min(proj_mov)
        max_fix, min_fix = np.max(proj_fix), np.min(proj_fix)
        
        if abs(max_mov - max_fix) > abs(max_mov - min_fix):
            print(f'invert moving image eigenvector {nbVector}')
            w_mov[:,nbVector] *= -1
    
    # Step 3: make sure moving image has right hand coordinate system with eigenvectors
    RHS = checkRHS(w_mov)
    if not RHS:
        print('moving image eigenvectors are not right hand system. Inverting third vector')
        w_mov[:,2] *= -1
    
    # create output folder
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    
    if not DEBUG:
        # SimpleITK convention:
        # for coordinate matrix M = [c1 c2 ... cn]
        # transpose(M) * A = transpose(M')
        
        Rtot = np.matmul(w_mov,np.transpose(w_fix))
        
        # translation along principal ax if bone out of FOV
        maxMOV = np.where(v_mov == np.max(v_mov))[0][0]
        maxFIX = np.where(v_fix == np.max(v_fix))[0][0]
        
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
    s = MatrixToString(Rtot)
    
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    
    f = open(outputFolder + '/MaskRegistrationParam.txt','w')
    f.write('(Transform "AffineTransform")\n')
    f.write('(NumberOfParameters 12)\n')
    f.write(f'(TransformParameters {s} 0 0 0)\n')
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
    f.write('(CenterOfRotationPoint 0.0000000000 0.0000000000 0.0000000000)\n')
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