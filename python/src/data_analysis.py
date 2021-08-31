import os
import glob
import trimesh
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from . import utils

def plot_csv_column(pInFolder, pOutFolder, pCSV, pColName, pClip = 1000, label_size=5) :
    cwd = os.getcwd()
    inFolder = cwd + pInFolder
    csv = inFolder + pCSV

    models_df = pd.read_csv(csv)
    column = models_df[pColName]

    labels = dict()

    for c in column :
        if c in labels :
            labels[c] += 1
        else :
            labels[c] = 1

    for key in labels.keys() :
        if labels[key] > pClip :
            labels[key] = pClip

    fig = plt.figure()
    plt.barh(list(labels.keys()), list(labels.values()))
    plt.tick_params(axis='y', which='major', labelsize=label_size)
    plt.tight_layout()
    plt.grid(True)
    plt.title('{0} classes'.format(len(labels.keys())))
    fig.savefig(cwd + pOutFolder + pColName + '.pdf', bbox_inches='tight')

def plot_csv_one_hot(pInFolder, pOutFolder, pCSV) :
    cwd = os.getcwd()
    inFolder = cwd + pInFolder
    csv = inFolder + pCSV

    models_df = pd.read_csv(csv)
    models_df = models_df.iloc[:, 1:]

    labels = { label : 0 for label in models_df.columns.values }

    for key, value in labels.items() :
        labels[key] = (models_df[key] == 1.0).sum()

    fig = plt.figure()
    plt.barh(list(labels.keys()), list(labels.values()))
    plt.tick_params(axis='y', which='major', labelsize=10)
    plt.tight_layout()
    plt.grid(True)
    plt.title('{0} classes'.format(len(labels.keys())))
    fig.savefig(cwd + pOutFolder + pCSV[:pCSV.rfind('.')] + '_distr.pdf', bbox_inches='tight')

def plot_elements_from_classes(pInfolder, pOutFolder, pCSV, pColumn, pFraction, pClasses) :
    fig = plt.figure()
    utils.clear_make_dir(pOutFolder)

    if pCSV is not None :
        elem_df = pd.read_csv(pCSV)

        if pClasses is None :
            pClasses = set()
            for sample in elem_df.loc[:, pColumn] :
                pClasses.add(sample)

        for class_name in pClasses :
            class_name = class_name.replace('\\', '_')
            class_name = class_name.replace('/', '_')
            class_name = class_name.replace('//', '_')
            class_name = class_name.rstrip()

            utils.clear_make_dir(os.path.join(pOutFolder, class_name))
            samples = elem_df.loc[elem_df[pColumn] == class_name].sample(frac=pFraction).reset_index(drop=True)

            for i, sample in samples.iterrows() :
                mesh_file = sample['Model']
                
                model = trimesh.load(os.path.join(pInfolder, mesh_file))
                model = utils.align_model_obb(model)
                model = utils.normalizeModel_unorm(model)

                plt.clf()
                ax = fig.gca(projection='3d')
                plot_model(model, ax)
                mesh_name = os.path.basename(os.path.splitext(mesh_file)[0] + '.jpg')
                fig.savefig(os.path.join(pOutFolder, class_name, mesh_name), bbox_inches='tight')
    else :
        _, folders, _ = next(os.walk(pInfolder))

        for folder in folders :
            plot_elements_from_class(
                os.path.join(pInfolder, folder),
                os.path.join(pOutFolder, folder))

def plot_elements_from_class(pInfolder, pOutFolder) :
    utils.clear_make_dir(pOutFolder)
    fig = plt.figure()

    for model_file in glob.glob(os.path.join(pInfolder, '*.obj')) :
        model = trimesh.load(model_file)
        model = utils.align_model_obb(model)
        model = utils.normalizeModel_unorm(model)

        plt.clf()
        ax = fig.gca(projection='3d')
        plot_model(model, ax)
        mesh_name = os.path.basename(os.path.splitext(model_file)[0] + '.jpg')
        fig.savefig(os.path.join(pOutFolder, mesh_name), bbox_inches='tight')

def visualize_sample(pInfolder, pFile) :
    cwd = os.getcwd()
    infolder = cwd + pInfolder
    voxDim = 32
    model = trimesh.load(infolder + pFile)
    model = fp_utils.normalizeModel(model)
    samples = model.sample(5000)
    voxel_grid = fp_utils.binary_voxels(samples, (voxDim - 2, voxDim - 2, voxDim - 2))

    dim = np.ones(shape=(3,), dtype=np.int) * voxDim - (voxDim - 2)

    voxel_grid = np.pad(voxel_grid, [
        (math.floor(dim[0]/2), math.ceil(dim[0]/2)),
        (math.floor(dim[1]/2), math.ceil(dim[1]/2)),
        (math.floor(dim[2]/2), math.ceil(dim[2]/2))], mode='constant')

    fig = plt.figure()

    axs = fig.add_subplot(311, projection='3d')
    axs.plot_trisurf(model.vertices[:, 0], model.vertices[:, 1], model.vertices[:, 2], triangles=model.faces, alpha=0.5, cmap=cm.cool)
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_zlabel('z')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.set_zlim([0, 1])

    axs = fig.add_subplot(312, projection='3d')
    x, y, z = np.indices((voxDim + 1, voxDim + 1, voxDim + 1))
    axs.voxels(x, y, z, voxel_grid, alpha=0.5)
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_zlabel('z')
    axs.set_xlim([0, 31])
    axs.set_ylim([0, 31])
    axs.set_zlim([0, 31])

    axs = fig.add_subplot(313, projection='3d')
    axs.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=2)
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_zlabel('z')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.set_zlim([0, 1])

    plt.tight_layout()
    plt.show()

def plot_voxels_from_classes(pInFolder, pOutFolder) :
    utils.clear_make_dir(pOutFolder)
    _, folders, _ = next(os.walk(pInFolder))

    for folder in folders :
        outFolder = os.path.join(pOutFolder, folder)
        plot_voxels_from_class(os.path.join(pInFolder, folder), outFolder)

def plot_voxels_from_class(pInFolder, pOutFolder) :
    utils.clear_make_dir(pOutFolder)
    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')

    for model_file in glob.glob(os.path.join(pInFolder, '*.npz')) :
        with np.load(model_file, allow_pickle=True) as grid :
            voxel_grid = grid['a']

        plt.cla()
        plot_voxelgrid(voxel_grid, axs)
        mesh_name = os.path.basename(os.path.splitext(model_file)[0] + '.jpg')
        plt.savefig(os.path.join(pOutFolder, mesh_name), bbox_inches='tight')

def plot_voxelgrid_file(pInFile, ax=None) :
    with np.load(pInFile, allow_pickle=True) as grid :
        voxel_grid = grid['a']

    plot_voxelgrid(voxel_grid, ax)

def plot_voxelgrid(pGrid, ax=None) :
    plotGrid = True if ax is None else False

    if plotGrid :
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    voxDim = pGrid.shape[0]
    ax.voxels(pGrid, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0, voxDim])
    ax.set_ylim([0, voxDim])
    ax.set_zlim([0, voxDim])

    if plotGrid :
        plt.show()

def plot_model_file(pFile, ax=None) :
    model = trimesh.load(pFile)
    plot_model(model)

def plot_model(pModel, ax=None) :
    plotModel = True if ax is None else False
    if plotModel :
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(
        pModel.vertices[:, 0],
        pModel.vertices[:, 1],
        pModel.vertices[:, 2],
        triangles=pModel.faces, alpha=0.5, cmap=cm.cool)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    if plotModel :
        plt.show()