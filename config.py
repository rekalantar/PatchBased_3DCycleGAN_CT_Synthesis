'''
File created by Reza Kalantar - 29/11/2022
'''

import torch
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    LoadImaged,
    Invertd,
    Orientationd,
    MapTransform,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

params = {
    'num_pool': 100, #number of images generated in image pool
    'roi_size': [128,128,128], #determines the patch size
    'pixdim':(0.8,0.8,8.0), #resampling pixel dimensions
    'imgA_intensity_range': (-1000,1000), #range of intensities for nomalization to range [-1,1]
    'imgB_intensity_range': (0,1500),
}

# Transformations for dynamic loading and sampling of Nifti files
train_transforms = Compose([
    LoadImaged(keys=['imgA', 'imgB']),
    EnsureChannelFirstd(keys=['imgA', 'imgB']),
#     Orientationd(keys=['imgA', 'imgB'], axcodes='RAS'),
    CropForegroundd(keys=['imgA'], source_key='imgA'),
    CropForegroundd(keys=['imgB'], source_key='imgB'),
    Spacingd(keys=['imgA', 'imgB'], pixdim=params['pixdim'], mode=("bilinear", "bilinear")),
    ScaleIntensityRanged(keys=['imgA'], a_min=params['imgA_intensity_range'][0], a_max=params['imgA_intensity_range'][1], b_min=-1.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=['imgB'], a_min=params['imgB_intensity_range'][0], a_max=params['imgB_intensity_range'][1], b_min=-1.0, b_max=1.0, clip=False),
    RandSpatialCropd(keys=['imgA'], roi_size=params['roi_size'], random_size=False, random_center=True),
    RandSpatialCropd(keys=['imgB'], roi_size=params['roi_size'], random_size=False, random_center=True),
    SpatialPadd(keys=["imgA", "imgB"], spatial_size=params['roi_size']),
])

test_transforms = Compose([
    LoadImaged(keys=['imgA', 'imgB']),
    EnsureChannelFirstd(keys=['imgA', 'imgB']),
    #     Orientationd(keys=['imgA', 'imgB'], axcodes='RAS'),
    CropForegroundd(keys=['imgA'], source_key='imgA'),
    CropForegroundd(keys=['imgB'], source_key='imgB'),
    Spacingd(keys=['imgA', 'imgB'], pixdim=params['pixdim'], mode=("bilinear", "bilinear")),
    ScaleIntensityRanged(keys=['imgA'], a_min=params['imgA_intensity_range'][0], a_max=params['imgA_intensity_range'][1], b_min=-1.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=['imgB'], a_min=params['imgB_intensity_range'][0], a_max=params['imgB_intensity_range'][1], b_min=-1.0, b_max=1.0, clip=False),
    RandSpatialCropd(keys=['imgA'], roi_size=params['roi_size'], random_size=False, random_center=True),
    RandSpatialCropd(keys=['imgB'], roi_size=params['roi_size'], random_size=False, random_center=True),
    SpatialPadd(keys=["imgA", "imgB"], spatial_size=params['roi_size']),
])