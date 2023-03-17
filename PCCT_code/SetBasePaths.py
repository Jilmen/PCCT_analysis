import os
import SimpleITK as sitk

spec1 = 'D:\\PCCT_data\\specimens\\spec1_2701230006'

spec2 = 'D:\\PCCT_data\\specimens\\spec2_2701230007'

spec3 = 'D:\\PCCT_data\\specimens\\spec3_2701230008'

spec4 = 'D:\\PCCT_data\\specimens\\spec4_2701230009'

spec5 = 'D:\\PCCT_data\\specimens\\spec5_2701230010'

spec6 = 'D:\\PCCT_data\\specimens\\spec6_1105210053'

spec7 = 'D:\\PCCT_data\\specimens\\spec7_1105210052'

spec8 = 'D:\\PCCT_data\\specimens\\spec8_1105210054'

def SetIms(spec, bone):
    pathfix = os.path.join(spec, 'full_wrist', 'BMP_int8', bone, 'mask.mha')
    pathmov = os.path.join(spec, 'full_wrist_reproducability', 'BMP_int8', bone, 'mask.mha')
    outpath = os.path.join(spec, 'full_wrist_reproducability', 'Registration', bone)
    
    fix = sitk.ReadImage(pathfix)
    mov = sitk.ReadImage(pathmov)
    
    return fix, mov, outpath