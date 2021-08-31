import os
import tensorflow as tf

from src import model
from tensorflow.keras import backend

def train_BIM() :
    WD_PATH = os.path.join(os.getcwd(), 'python', 'dataset')
    PATH_TO_DATA_UNIFORM_NORM = os.path.join(WD_PATH, 'BIM elements clean vox32 uniform norm')
    PATH_TO_DATA_MAX_NORM = os.path.join(WD_PATH, 'BIM elements clean vox32 max norm')

    net1 = model.voxel_model(PATH_TO_DATA_UNIFORM_NORM,
        pCSVTrainFile=os.path.join(WD_PATH, 'vox32_train.csv'),
        pCSVTestFile=os.path.join(WD_PATH, 'vox32_test.csv'),
        pName='uniform_norm',
        pIsUniformNorm=True)
    
    net1.build_network()
    net1.train_model(pNumEpochs=50)

    net2 = model.voxel_model(PATH_TO_DATA_UNIFORM_NORM,
        pCSVTrainFile=os.path.join(WD_PATH, 'vox32_train.csv'),
        pCSVTestFile=os.path.join(WD_PATH, 'vox32_test.csv'),
        pName='uniform_norm_no_obb',
        pIsUniformNorm=False)
    
    net2.build_network()
    net2.train_model(pNumEpochs=50)

    net3 = model.voxel_model(PATH_TO_DATA_MAX_NORM,
        pCSVTrainFile=os.path.join(WD_PATH, 'vox32_train.csv'),
        pCSVTestFile=os.path.join(WD_PATH, 'vox32_test.csv'),
        pName='max_norm',
        pIsUniformNorm=False)
    
    net3.build_network()
    net3.train_model(pNumEpochs=50)    

def main() :
    train_BIM()

if __name__ == "__main__" :
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.run_functions_eagerly(False)
    print(tf.version.VERSION)
    main()