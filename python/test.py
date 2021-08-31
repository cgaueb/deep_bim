import os
import tensorflow as tf

from src import model
from tensorflow.keras import backend

WD_PATH = os.path.join(os.getcwd(), 'python', 'dataset')
PATH_TO_DATA_UNIFORM_NORM = os.path.join(WD_PATH, 'BIM elements clean vox32 uniform norm')
PATH_TO_DATA_MAX_NORM = os.path.join(WD_PATH, 'BIM elements clean vox32 max norm')

def main() :
    net1 = model.voxel_model(pPathToData=PATH_TO_DATA_UNIFORM_NORM,
        pCSVTrainFile=os.path.join(WD_PATH, 'vox32_train.csv'),
        pCSVTestFile=os.path.join(WD_PATH, 'vox32_test.csv'),
        pName='uniform_norm', pIsUniformNorm=True)

    net2 = model.voxel_model(pPathToData=PATH_TO_DATA_UNIFORM_NORM,
        pCSVTrainFile=os.path.join(WD_PATH, 'vox32_train.csv'),
        pCSVTestFile=os.path.join(WD_PATH, 'vox32_test.csv'),
        pName='uniform_norm_no_obb', pIsUniformNorm=False)
    
    net3 = model.voxel_model(pPathToData=PATH_TO_DATA_MAX_NORM,
        pCSVTrainFile=os.path.join(WD_PATH, 'vox32_train.csv'),
        pCSVTestFile=os.path.join(WD_PATH, 'vox32_test.csv'),
        pName='max_norm', pIsUniformNorm=False)

    net1.load_trained_model()
    net2.load_trained_model()
    net3.load_trained_model()

    print('Max norm network Train', net3.test_model((
        os.path.join(WD_PATH, 'vox32_train.csv'),
        os.path.join(WD_PATH, 'vox32_test.csv'))))

    print('Uniform norm no obb network', net2.test_model((
        os.path.join(WD_PATH, 'vox32_train.csv'),
        os.path.join(WD_PATH, 'vox32_test.csv'))))

    print('Uniform norm network', net1.test_model((
        os.path.join(WD_PATH, 'vox32_train.csv'),
        os.path.join(WD_PATH, 'vox32_test.csv'))))

    net1.visualize_features(pMethod='UMAP')
    net2.visualize_features(pMethod='UMAP')
    net3.visualize_features(pMethod='UMAP')

if __name__ == "__main__" :
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.run_functions_eagerly(False)
    print(tf.version.VERSION)
    main()