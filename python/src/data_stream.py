import os
import glob
import math
import trimesh

import numpy as np
import random

from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence

class dataStream_VX(Sequence) :
    def __init__(self, X, y, pVoxelDim, pBatchSize, pPathToData, isUniformNorm) :
        super(dataStream_VX, self).__init__()
        self.batchSize = pBatchSize
        self.voxelDim = pVoxelDim
        self.numX = X.shape[0]
        self.X = X
        self.y = y
        self.path = pPathToData
        self.isUniformNorm = isUniformNorm

    def __len__(self) :
        return int(math.ceil(self.numX / self.batchSize))

    def __getitem__(self, index) :
        batch_x = self.X[index * self.batchSize: (index + 1) * self.batchSize]
        batch_y = self.y[index * self.batchSize: (index + 1) * self.batchSize]
        labels = np.array(batch_y, dtype=np.float)

        if self.isUniformNorm :
            samples = [self.__load_sample_unorm(sampleFile) for sampleFile in batch_x]
            grids = np.array([sample[0] for sample in samples])
            ar = np.array([sample[1] for sample in samples])
            return [grids, ar], labels
        else :
            grids = np.array([self.__load_sample_mnorm(sampleFile) for sampleFile in batch_x])
            return grids, labels

    def __load_sample_unorm(self, pSampleFile) :
        with np.load(os.path.join(self.path, pSampleFile), allow_pickle=True) as f :
            grid = f['a']
            ar = f['b']

        return grid, ar

    def __load_sample_mnorm(self, pSampleFile) :
        with np.load(os.path.join(self.path, pSampleFile)) as grid :
            grid = grid['a']

        return grid

    def on_epoch_end(self) :
        pass