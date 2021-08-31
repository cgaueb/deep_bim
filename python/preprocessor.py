import os
import glob
import shutil

from src import dataset_builder
from src import utils

WD_PATH = os.path.join(os.getcwd(), 'python')
DATASET_DIR_PATH = os.path.join(WD_PATH, 'dataset')
CLEAN_DATASET_DIR_PATH = os.path.join(DATASET_DIR_PATH, 'BIM elements clean')
CUSTOM_VOXELIZER_PATH = os.path.join(os.getcwd(), 'voxelizer', 'bin', 'voxelizer')
BINVOX_PATH = os.path.join(os.getcwd(), 'voxelizer', 'binvox', 'binvox')

def BIM_dataset_builder() :
    dataset_builder.augment_classes(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean'),
        pOutFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pAugmentations={'rotations' : [([0, 1, 0, 40.0], 2), ]})
    
    dataset_builder.augment_class(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean', 'Base-column'),
        pOutFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pAugmentations={'rotations' : [([0, 1, 0, 40.0], 6), ]})

    dataset_builder.augment_class(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean', 'Wall'),
        pOutFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pAugmentations={'rotations' : [([0, 1, 0, 40.0], 3), ]})

    dataset_builder.augment_class(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean', 'Slab'),
        pOutFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pAugmentations={'rotations' : [([0, 1, 0, 40.0], 3), ]})

    dataset_builder.augment_class(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean', 'H-beam'),
        pOutFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pAugmentations={'rotations' : [('random', 40.0, 2), ]})

    dataset_builder.augment_class(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean', 'I-beam'),
        pOutFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pAugmentations={'rotations' : [('random', 40.0, 2), ]})

    dataset_builder.augment_class(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean', 'L-beam'),
        pOutFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pAugmentations={'rotations' : [('random', 40.0, 2), ]})

    dataset_builder.augment_class(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean', 'U-beam'),
        pOutFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pAugmentations={'rotations' : [('random', 40.0, 2), ]})

    dataset_builder.align_dataset(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pOutFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean aligned augmented'))

    dataset_builder.voxelize_elements(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pOutFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean vox32 uniform norm'),
        VoxelizerExePath=CUSTOM_VOXELIZER_PATH,
        isMaxNorm=False)
  
    dataset_builder.voxelize_elements(
        os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        os.path.join(DATASET_DIR_PATH, 'BIM elements clean vox32 max norm'),
        VoxelizerExePath=CUSTOM_VOXELIZER_PATH,
        isMaxNorm=True)

    dataset_builder.build_csv(
        pInFolder=os.path.join(DATASET_DIR_PATH, 'BIM elements clean augmented'),
        pOutFolder=DATASET_DIR_PATH,
        pFileName='vox32')

def main() :
    BIM_dataset_builder()

if __name__ == "__main__" :
    main()