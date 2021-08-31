import os
import glob
from . import data_analysis
import trimesh
import shutil
import numpy as np
import pandas as pd
import subprocess

from sklearn.model_selection import train_test_split

from . import utils
from . import binvox_rw

def build_csv(pInFolder, pOutFolder, pFileName) :
    test_file = os.path.join(pOutFolder, pFileName + "_test.csv")
    train_file = os.path.join(pOutFolder, pFileName + "_train.csv")

    if os.path.exists(test_file) :
        os.remove(test_file)

    if os.path.exists(train_file) :
        os.remove(train_file)

    model_tuples = list()
    root, classes, _ = next(os.walk(pInFolder))
    num_classes = len(classes)

    for i_cls, class_data in enumerate(classes) :
        root2, _, files = next(os.walk(os.path.join(root, class_data)))

        for i, model_file in enumerate(files) :
            data_lst = list()

            class_index = [0.0 for j in range(num_classes)]
            class_index[i_cls] = 1.0
            model_name = os.path.join(class_data, model_file[:model_file.rfind('.')] + '.npz')

            data_lst += [[model_name, ] + class_index, ]
            model_tuples.extend(data_lst)

    model_frame = pd.DataFrame(model_tuples, columns = ['Model',] + classes).sample(frac=1).reset_index(drop=True)

    split_size = int(len(model_frame.index) * 0.7)
    train_frame = model_frame.iloc[:split_size, :]
    test_frame = model_frame.iloc[split_size:, :]
    
    test_frame.to_csv(test_file, index=False)
    train_frame.to_csv(train_file, index=False)

def create_csv(pInCSV, pOutFolder, top_k, model_type) :
    cwd = os.getcwd()
    elem_df = pd.read_csv(cwd + pInCSV)
    desc_classes = dict()

    for x,y in zip(elem_df[model_type], elem_df['Model']) :
        if x in desc_classes :
            desc_classes[x] += [y,]
        else :
            desc_classes[x] = [y,]

    cls_tuples = [(key, len(desc_classes[key])) for key in desc_classes]
    cls_tuples = sorted(cls_tuples, key=lambda x : x[1], reverse=True)
    print(cls_tuples)
    cls_tuples = cls_tuples[:top_k]

    data_lst = list()

    for i_cls, cls_type in enumerate(cls_tuples) :
        class_index = [0.0 for i in range(top_k)]
        class_index[i_cls] = 1.0

        for item in desc_classes[cls_type[0]] :
            file_path = item
            data_lst += [[file_path, ] + class_index, ]

    dataset = pd.DataFrame(data_lst, columns = ['Model',] + [name[0] for name in cls_tuples])
    dataset.to_csv(cwd + pOutFolder + 'models_' + model_type + '_top' + str(top_k) + '.csv', index=False)

def split_train_test_bridges(pDataFolder) :
    cwd = os.getcwd()
    datafolder = cwd + pDataFolder

    root, class_folders, files1 = next(os.walk(datafolder))

    for class_folder in class_folders :
        root2, folders2, models = next(os.walk(datafolder + class_folder))

        train, test = train_test_split(models, test_size=0.3)
        path = datafolder + class_folder + '\\'
        utils.clear_make_dir(path + 'train')
        utils.clear_make_dir(path + 'test')

        for train_file in train :
            shutil.move(path + train_file, path + 'train\\' + train_file)
        
        for test_file in test :
            shutil.move(path + test_file, path + 'test\\' + test_file)

def augment_classes(pInFolder, pOutFolder, pAugmentations) :
    _, class_folders, _ = next(os.walk(pInFolder))
    utils.clear_make_dir(pOutFolder)

    for class_folder in class_folders :
        augment_class(
            os.path.join(pInFolder, class_folder),
            pOutFolder,
            pAugmentations)

def align_classes(pInFolder, pOutFolder) :
    _, class_folders, _ = next(os.walk(pInFolder))
    utils.clear_make_dir(pOutFolder)

    for class_folder in class_folders :
        align_class(
            os.path.join(pInFolder, class_folder),
            os.path.join(pOutFolder, class_folder),)

def align_class(pInFolder, pOutFolder, pAugmentations) :
    outFolder = os.path.join(pOutFolder, os.path.basename(pInFolder))
    utils.clear_make_dir(outFolder)

    for model_file in glob.glob(os.path.join(pInFolder, '*.obj')) :
        model_name = os.path.basename(model_file)
        mesh = trimesh.load(os.path.join(pInFolder, model_file))
        #mesh = utils.normalizeModel(mesh)
        mesh = utils.centerModel_toOrigin(mesh)
        onb = utils.getONB(mesh)
        if 'use_points' in pAugmentations:
            if pAugmentations['use_points'] == 1:
                onb = utils.getPointCloudONB(mesh, 10000)

        mesh.apply_transform(trimesh.transformations.inverse_matrix(onb))

        obj_file = trimesh.exchange.obj.export_obj(mesh, False, False, False, False, False)
     
        out_model_dir = os.path.join(outFolder, model_name)
        with open(out_model_dir[:out_model_dir.rfind('.')] + '_rot' + '.obj', 'w') as f : 
            f.write(obj_file)

def augment_class(pInFolder, pOutFolder, pAugmentations) :
    outFolder = os.path.join(pOutFolder, os.path.basename(pInFolder))
    utils.clear_make_dir(outFolder)

    for model_file in glob.glob(os.path.join(pInFolder, '*.obj')) :
        model_name = os.path.basename(model_file)
        mesh = trimesh.load(os.path.join(pInFolder, model_file))
        #mesh = utils.normalizeModel(mesh)
        mesh = utils.centerModel_toOrigin(mesh)

        out_model_dir = os.path.join(outFolder, model_name)

        if 'rotations' in pAugmentations :
            obj_file = trimesh.exchange.obj.export_obj(mesh, False, False, False, False, False)

            with open(out_model_dir, 'w') as f :
                f.write(obj_file)

            for rot in pAugmentations['rotations'] :
                if rot[0] == 'random' :
                    degrees = np.random.randint(0, 360 // (rot[1] - 1), rot[2]) * rot[1]
                    for degree in degrees :
                        angle = np.radians(degree)
                        v = np.random.uniform(size=3)
                        direction = v / np.linalg.norm(v)
                        rot_mat = trimesh.transformations.rotation_matrix(angle, direction)
                        mesh.apply_transform(rot_mat)
                        obj_file = trimesh.exchange.obj.export_obj(mesh, False, False, False, False, False)

                        with open(os.path.splitext(out_model_dir)[0] + '_rot' + str(int(degree)) + '.obj', 'w') as f :
                            f.write(obj_file)
                else :
                    degrees = np.random.randint(0, 360 // (rot[0][3] - 1), rot[1]) * rot[0][3]

                    for degree in degrees :
                        angle = np.radians(degree)
                        direction = rot[0][:3]
                        rot_mat = trimesh.transformations.rotation_matrix(angle, direction)
                        mesh.apply_transform(rot_mat)
                        obj_file = trimesh.exchange.obj.export_obj(mesh, False, False, False, False, False)

                        with open(os.path.splitext(out_model_dir)[0] + '_rot' + str(int(degree)) + '.obj', 'w') as f :
                            f.write(obj_file)
        if 'scaling' in pAugmentations :
            value = pAugmentations['scaling']
            #mesh = utils.normalizeModel(mesh)
            scales = np.reshape(np.random.uniform(0.7, 1.5, value[0][1] * 2), (-1, 2))

            for i, scale in enumerate(scales) :
                mesh.apply_scale([scale[0], scale[1], 1])   
                obj_file = trimesh.exchange.obj.export_obj(mesh, False, False, False, False, False)

                with open(os.path.join(pInFolder, 'wall_' + str(i) + '.obj'), 'w') as f :
                    f.write(obj_file)
                
                mesh.apply_scale([1.0 / scale[0], 1.0 / scale[1], 1])

def augment_classes_2(pInFolder, pOutFolder, pAugmentations) :
    _, class_folders, _ = next(os.walk(pInFolder))
    utils.clear_make_dir(pOutFolder)

    for class_folder in class_folders :
        _, subsets, _ = next(os.walk(os.path.join(pInFolder, class_folder)))

        for subset in subsets :
            dst_folder = os.path.join(pOutFolder, class_folder)
            utils.clear_make_dir(dst_folder)

            augment_class(
                os.path.join(pInFolder, class_folder, subset),
                os.path.join(pOutFolder, class_folder),
                pAugmentations)
            
            for model_file in glob.glob(os.path.join(dst_folder, subset, '*.obj')) :
                shutil.move(model_file, os.path.join(dst_folder, os.path.basename(model_file)))

            shutil.rmtree(os.path.join(dst_folder, subset))

def transfer_samples_from_folder(pImageFolder, pInFolder, pOutFolder) :
    _, folders, _ = next(os.walk(pImageFolder))

    for folder in folders :
        utils.clear_make_dir(os.path.join(pOutFolder, folder))
        for model in glob.glob(os.path.join(pImageFolder, folder, '*.jpg')) :
            model_name = os.path.basename(os.path.splitext(model)[0]) + '.obj' 
            shutil.copy(
                os.path.join(pInFolder, model_name),
                os.path.join(pOutFolder, folder, model_name))

def load_asc_file(file, voxDim) :
    voxel_grid = np.zeros(shape=(voxDim, voxDim, voxDim))

    with open(file, 'r') as f:
        lines = f.readlines()

        for line in lines :
            coords = [int(c) for c in line.split() if c.isdigit()]
            voxel_grid[coords[0], coords[1], coords[2]] = 1.0

    return voxel_grid

def voxelize_elements(pInFolder, pOutFolder, VoxelizerExePath, isMaxNorm) :
    _, folders, _ = next(os.walk(pInFolder))
    temp = os.path.join(pOutFolder, 'temp')
    utils.clear_make_dir(pOutFolder)
    utils.clear_make_dir(temp)

    for folder in folders :
        outFolder = os.path.join(temp, folder)
        inFolder = os.path.join(pInFolder, folder)
        utils.clear_make_dir(outFolder)
        utils.clear_make_dir(os.path.join(pOutFolder, folder))

        for model_file in glob.glob(os.path.join(inFolder, '*.obj')) :
            mesh = trimesh.load(model_file)
        
            obb = utils.obb_vertices(mesh) / 1000.0
            np.savez_compressed(file=os.path.join(pOutFolder, folder, os.path.basename(os.path.splitext(model_file)[0])),
                b=obb)

            if not isMaxNorm :
                mesh = utils.align_model_obb(mesh)

            mesh = utils.normalizeModel(mesh)
            mesh_name = os.path.basename(model_file)

            obj_file = trimesh.exchange.obj.export_obj(mesh, False, False, False, False, False)

            with open(os.path.join(outFolder, mesh_name), 'w') as f :
                f.write(obj_file)

        subprocess.run([VoxelizerExePath, '-d', outFolder, outFolder, '1' if isMaxNorm else '0'])

    _, folders, _ = next(os.walk(temp))

    for folder in folders :
        temp_folder = os.path.join(pOutFolder, 'temp', folder)
        outFolder = os.path.join(pOutFolder, folder)

        for model_file in glob.glob(os.path.join(temp_folder, '*.asc')) :
            voxel_grid = load_asc_file(model_file, 32)
            model_name = os.path.basename(os.path.splitext(model_file)[0])
            
            with np.load(os.path.join(outFolder, model_name + '.npz'), allow_pickle=True) as f :
                ar = f['b']

            np.savez_compressed(file=os.path.join(outFolder, model_name),
                a=voxel_grid,
                b=ar)

            os.remove(model_file)

        for model_file in glob.glob(os.path.join(temp_folder, '*.obj')) :
            os.remove(model_file)

        os.rmdir(temp_folder)

    os.rmdir(temp)

def voxelize_elements_binvox(pInFolder, pOutFolder, VoxelizerExePath, useSolidVoxelization) :
    _, folders, _ = next(os.walk(pInFolder))
    utils.clear_make_dir(pOutFolder)

    for folder in folders :
        inFolder = os.path.join(pInFolder, folder)

        outFolder = os.path.join(pOutFolder, folder)
        utils.clear_make_dir(outFolder)

        for model_file in glob.glob(os.path.join(inFolder, '*.obj')) :

            if useSolidVoxelization :
                subprocess.run([VoxelizerExePath, '-dc', '-cb', '-aw', '-d', '32', model_file])
            else :
                subprocess.run([VoxelizerExePath, '-dc', '-cb', '-aw', '-ri', '-d', '32', model_file])

            vox_file = os.path.splitext(model_file)[0] + '.binvox'
            vox_name = os.path.splitext(model_file)[0]
            out_vox_file = os.path.join(outFolder, vox_name)

            with open(vox_file, 'rb') as f :
                voxel_grid = binvox_rw.read_as_3d_array(f)

            np.savez_compressed(out_vox_file, a=voxel_grid.data)
            os.remove(vox_file)