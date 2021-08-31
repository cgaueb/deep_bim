
import numpy as np
import trimesh
import os
import shutil
import glob

def getPointCloudONB(pModel, sample_count=1000) :
    samples, _ = trimesh.sample.sample_surface(pModel, sample_count)
    point_mesh = trimesh.points.PointCloud(samples)
    return point_mesh.bounding_box_oriented.primitive.transform

def getONB(pModel):
    return pModel.bounding_box_oriented.primitive.transform

def align_model_obb(pModel) :
    pModel = centerModel_toOrigin(pModel)
    onb = getONB(pModel)
    pModel.apply_transform(trimesh.transformations.inverse_matrix(onb))
    return pModel

def getAABB(pVertices) :
    aabb = np.empty((2, 3))

    aabb[0, 0] = np.min(pVertices[:, 0])
    aabb[0, 1] = np.min(pVertices[:, 1])
    aabb[0, 2] = np.min(pVertices[:, 2])

    aabb[1, 0] = np.max(pVertices[:, 0])
    aabb[1, 1] = np.max(pVertices[:, 1])
    aabb[1, 2] = np.max(pVertices[:, 2])

    return aabb

def normalizePoints(pPoints) :
    aabb = getAABB(pPoints)

    diag = aabb[1] - aabb[0]
    minP = np.min(diag)
    maxP = np.max(diag)
    scale = 1.0 / maxP
    normPoints = (pPoints - minP) * scale

    normAABB = (aabb - minP) * scale
    normCent = (normAABB[1] + normAABB[0]) * 0.5
    
    return normPoints + (np.array([0.5, 0.5, 0.5]) - normCent)

def aspect_ratio(pModel) :
    aabb = getAABB(pModel.vertices)
    diag = aabb[1] - aabb[0]
    xy = diag[0] / diag[1]
    xz = diag[0] / diag[2]
    yz = diag[1] / diag[2]
    
    return np.array([xy, xz, yz])

def obb_vertices(pModel) :
    mesh = centerModel_toOrigin(pModel)
    return mesh.bounding_box_oriented.vertices

def normalizeModel(pModel) :
    aabb = getAABB(pModel.vertices)

    diag = aabb[1] - aabb[0]
    minP = np.min(diag)
    maxP = np.max(diag)
    scale = 1.0 / maxP

    pModel.apply_translation(-np.ones(shape=(3,)) * minP)
    pModel.apply_scale(scale)

    normAABB = (aabb - minP) * scale
    normCent = (normAABB[1] + normAABB[0]) * 0.5

    pModel.apply_translation(np.array([0.5, 0.5, 0.5]) - normCent)
    
    return pModel

def normalizeModel_unorm(pModel) :
    aabb = getAABB(pModel.vertices)

    diag = aabb[1] - aabb[0]
    minP = np.min(diag)
    scale = 1.0 / diag

    pModel.apply_translation(-np.ones(shape=(3,)) * minP)
    pModel.apply_scale(scale)

    normAABB = (aabb - minP) * scale
    normCent = (normAABB[1] + normAABB[0]) * 0.5

    pModel.apply_translation(np.array([0.5, 0.5, 0.5]) - normCent)
    
    return pModel

def binary_voxels(norm_points, resolution) :
    voxels = np.zeros(resolution, dtype=np.float32)
    x = np.floor(norm_points[:, 0] * (resolution[0] - 1)).astype(np.int32)
    y = np.floor(norm_points[:, 1] * (resolution[1] - 1)).astype(np.int32)
    z = np.floor(norm_points[:, 2] * (resolution[2] - 1)).astype(np.int32)
    voxels[x, y, z] = 1.0
    return voxels

def centerModel_toOrigin(pModel) :
    aabb = getAABB(pModel.vertices)    
    center = (aabb[1] + aabb[0]) * 0.5
    pModel.apply_translation(-center)
    return pModel

def clear_make_dir(pFolderDir) :
    if not os.path.exists(pFolderDir) :
        os.mkdir(pFolderDir)
    else :
        for file in glob.glob(os.path.join(pFolderDir, '*')) :
            if os.path.isdir(file) :
                shutil.rmtree(file)
            else :
                os.remove(file)
        #shutil.rmtree(pFolderDir)
        #os.mkdir(pFolderDir)