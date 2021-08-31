import os
import multiprocessing

from . import data_stream
from . import layers

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
import umap
from sklearn.manifold import TSNE

from tensorflow import keras
from tensorflow.keras import Input, losses
from tensorflow.keras.layers import Flatten, Dense, Concatenate

class voxel_model() :
    def __init__(self, pPathToData, pCSVTrainFile = None, pCSVTestFile = None, pBatchSize = 32, pName = '', pIsUniformNorm = True) :
        self.mBatchSize = pBatchSize
        self.mModel = None
        self.mName = pName
        self.mWeightFile = None
        self.mVoxDim = (32, 32, 32)
        self.mPathToData = pPathToData
        self.isUniformNorm = pIsUniformNorm
        self.mRootFolder = os.path.join(os.getcwd(), 'python', 'metadata')

        if not os.path.exists(self.mRootFolder) :
            os.mkdir(self.mRootFolder)

        self.mRootFolder = os.path.join(self.mRootFolder, pName + '_voxel_' + str(self.mVoxDim[0]), '')

        if not os.path.exists(self.mRootFolder) :
            os.mkdir(self.mRootFolder)

        if pCSVTrainFile is not None :
            self.mTrain_csv = pd.read_csv(pCSVTrainFile)
            self.mTest_csv = pd.read_csv(pCSVTestFile)

            X_train, y_train, labels = self.__csvToDataset(self.mTrain_csv)
            X_test, y_test, labels = self.__csvToDataset(self.mTest_csv)

            self.mLabels = labels
            self.mTrainGen = data_stream.dataStream_VX(X_train, y_train, self.mVoxDim, self.mBatchSize, self.mPathToData, self.isUniformNorm)
            self.mValidGen = data_stream.dataStream_VX(X_test, y_test, self.mVoxDim, self.mBatchSize, self.mPathToData, self.isUniformNorm)

            self.mNumClasses = labels.shape[0]

    def build_network(self) :
        inputGrid = Input(shape=self.mVoxDim)
        grid = tf.expand_dims(inputGrid, -1)

        if self.isUniformNorm :
            grid = layers.voxel_encoder(
                name            = '32_to_8',
                filters         = 2,
                kernel_size     = 8,
                stride          = 4,
                padding         = 'same',
                pool_size       = 8,
                pool_stride     = 4,
                pool_padding    = 'same')(grid)
        else :
            grid = layers.voxel_encoder(
                name            = '32_to_8',
                filters         = 3,
                kernel_size     = 8,
                stride          = 4,
                padding         = 'same',
                pool_size       = 8,
                pool_stride     = 4,
                pool_padding    = 'same')(grid)

        features = Flatten(name='features_grid')(grid)

        if self.isUniformNorm :
            inputAR = Input(shape=(8,3))
            features_ar = layers.point_encoder(name = 'point_enc')(inputAR)
            features_ar = Flatten(name='features_ar')(features_ar)
            features = Concatenate(name='features_global')([features, features_ar])
            inputs = [inputGrid, inputAR,]
        else :
            features = Flatten(name='features_global')(features)
            inputs = [inputGrid,]

        classification = layers.classifier(
            name        = 'classifier',
            units       = 32,
            drop_rate   = 0.3,
            num_classes = self.mNumClasses)(features)

        self.mModel = keras.Model(inputs=inputs, outputs=[classification], name="bim_vox")
        self.mModel.summary()

    def __compile_model(self) :
        self.mModel.compile(
            loss=losses.CategoricalCrossentropy(),
            metrics=['accuracy',],
            optimizer=tf.keras.optimizers.Adam())

    def train_model(self, pNumEpochs) :
        self.__compile_model()

        termination_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.02, mode='min', verbose=1)
        checkpoint_cb = keras.callbacks.ModelCheckpoint(self.mRootFolder, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)

        history = self.mModel.fit(x = self.mTrainGen,
            validation_data = self.mValidGen,
            callbacks=[termination_cb, checkpoint_cb,],
            epochs = pNumEpochs,
            verbose = 1,)

        self.__export_history(history)
        self.__export_metrics()
        print('Exporting model...')
        keras.utils.plot_model(self.mModel, to_file=os.path.join(self.mRootFolder, "voxel_" + str(self.mVoxDim[0]) + ".png"), show_shapes=True, dpi=300)
        self.__export_features(self.mTrainGen)

    def __get_features_from_layer(self, pLayerName, pStream) :
        features_model = keras.Model(inputs=self.mModel.input, outputs=self.mModel.get_layer(pLayerName).output)
        features = features_model.predict(x = pStream)
        return features

    def __export_features(self, pStream) :
        print('Exporting features...')

        np.savez_compressed(os.path.join(self.mRootFolder, 'features'),
            a=self.__get_features_from_layer('features_global', pStream),
            b=pStream.y,
            c=self.mLabels)

    def __export_history(self, history) :
        print('Exporting train history...')
        fig = plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Train/Val Loss')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['train', 'val'])
        plt.xticks(np.arange(len(history.history['loss'])))
        plt.grid()
        fig.savefig(os.path.join(self.mRootFolder, 'history.pdf'), bbox_inches='tight')

        plt.clf()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Train/Val accuracy')
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['train', 'val'])
        plt.xticks(np.arange(len(history.history['accuracy'])))
        plt.grid()
        fig.savefig(os.path.join(self.mRootFolder, 'accuracy.pdf'), bbox_inches='tight')

    def __export_metrics(self) :
        print('Exporting metrics...')
        self.mModel.load_weights(self.mRootFolder)
        predictions = self.mModel.predict(x = self.mTrainGen, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(self.mTrainGen.y, axis=1)
        train_cm = confusion_matrix(y_true, y_pred)

        predictions = self.mModel.predict(x = self.mValidGen, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(self.mValidGen.y, axis=1)
        val_cm = confusion_matrix(y_true, y_pred)

        df_val_cm = pd.DataFrame(val_cm, index=self.mLabels, columns=self.mLabels)
        df_train_cm = pd.DataFrame(train_cm, index=self.mLabels, columns=self.mLabels)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        plt.ticklabel_format(style='plain', axis='both')

        axes[0].set_title('Train set confusion matrix')
        axes[1].set_title('Test set confusion matrix')

        sns.heatmap(ax=axes[0], data=df_train_cm, annot=True, fmt='g', cbar=False)
        sns.heatmap(ax=axes[1], data=df_val_cm, annot=True, fmt='g', cbar=False)

        fig.savefig(os.path.join(self.mRootFolder, 'confusion_matrix.pdf'), bbox_inches='tight', dpi=300)
        
        val_cm = val_cm.diagonal() / np.sum(val_cm, axis=1)
        train_cm = train_cm.diagonal() / np.sum(train_cm, axis=1)

        val_cm = val_cm[:, np.newaxis]
        train_cm = train_cm[:, np.newaxis]
        df_val_cm = pd.DataFrame(val_cm, index=self.mLabels, columns=['accuracy'])
        df_train_cm = pd.DataFrame(train_cm, index=self.mLabels, columns=['accuracy'])

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].set_title('Train set class accuracy')
        axes[1].set_title('Test set class accuracy')
        axes[0].grid(True)
        axes[1].grid(True)

        sns.barplot(ax=axes[0], data=df_train_cm, x='accuracy', y=df_train_cm.index)
        sns.barplot(ax=axes[1], data=df_val_cm, x='accuracy', y=df_val_cm.index)
        plt.grid(True)
        fig.savefig(os.path.join(self.mRootFolder, 'class_accuracy.pdf'), bbox_inches='tight')

    def load_trained_model(self) :
        with np.load(os.path.join(self.mRootFolder, 'features.npz'), allow_pickle=True) as f :
            self.mLabels = f['c']

        self.mNumClasses = len(self.mLabels)
        self.build_network()
        self.__compile_model()
        self.mModel.load_weights(self.mRootFolder)
    
    def test_model(self, pCSVFiles) :
        per_cls_acc = {'class names' : list(self.mLabels) + ['overall acc.',]}
        diags = []

        for csvFile in pCSVFiles :
            csv_name = os.path.basename(os.path.splitext(csvFile)[0])
            csv_df = pd.read_csv(csvFile)
            X, y, labels = self.__csvToDataset(csv_df)

            data_gen = data_stream.dataStream_VX(X, y,
                self.mVoxDim, self.mBatchSize,
                self.mPathToData, self.isUniformNorm)

            predictions = self.mModel.predict(x = data_gen, verbose=1)
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(y, axis=1)

            cm = confusion_matrix(y_true, y_pred)
            cm = cm.astype('float') / np.sum(cm, axis=1)[:, np.newaxis]
            diag = cm.diagonal()
            per_cls_acc[self.mName + '_' + csv_name] = list(diag) + [np.mean(diag),]

        return pd.DataFrame(per_cls_acc)

    def evaluate_model_from_folder(self, pPathToData, pCSVValFile) :
        X_val, y_val, labels = self.csvToDataset(pd.read_csv(pCSVValFile))
        valGen = data_stream.dataStream_VX(X_val, y_val, self.mVoxDim, self.mBatchSize, pPathToData)

        history = self.mModel.evaluate(x = valGen,
            verbose = 1,
            workers = multiprocessing.cpu_count())

    def visualize_features(self, pMethod = 'UMAP') :
        if pMethod == 'UMAP' :
            self.__visualize_features_umap('features.npz')
        elif pMethod == 'TSNE' :
            self.__visualize_features_tsne('features.npz')

    def __visualize_features_umap(self, pFeatureFile) :
        with np.load(os.path.join(self.mRootFolder, pFeatureFile), allow_pickle=True) as f :
            features = f['a']
            one_hot = f['b']
            labels = f['c']

        print('Appling dimensionality reduction...')

        one_hot = one_hot.astype(np.int32)

        a = [5, 10, 50, 100, 200]
        b = [0.01, 0.1, 0.5, 1.0]
        fig = plt.figure(figsize=(15, 10))

        for ai in a :
            for bi in b :
                print('Computing UMAP a={0}, b={1}'.format(ai, bi))
                transformer = umap.UMAP(n_components=2, n_neighbors=ai, min_dist=bi)
                X_embeddings = transformer.fit_transform(features)
                df_embeddings = pd.DataFrame({'x' : X_embeddings[:, 0], 'y' : X_embeddings[:, 1], 'label' : [labels[label == 1][0] for label in one_hot] })

                plt.clf()                
                sns.scatterplot(data=df_embeddings, x='x', y='y', hue="label", style='label', s=10)
                plt.legend()
                plt.grid(True)
                fig.savefig(os.path.join(self.mRootFolder, 'umap_{0}_{1}.png'.format(ai, bi)), bbox_inches='tight')

    def __visualize_features_tsne(self, pFeatureFile) :
        with np.load(os.path.join(self.mRootFolder, pFeatureFile), allow_pickle=True) as f :
            features = f['a']
            one_hot = f['b']
            labels = f['c']

        print('Appling dimensionality reduction...')

        p = [2, 5, 10, 20, 50, 100, 200]
        fig = plt.figure()

        for pi in p :
            print('Computing TSNE perplexity={0}'.format(pi))
            transformer = TSNE(n_components=2, perplexity=pi, n_iter=5000)
            X_embeddings = transformer.fit_transform(features)
            df_embeddings = pd.DataFrame({'x' : X_embeddings[:, 0], 'y' : X_embeddings[:, 1], 'label' : [labels[label == 1.0][0] for label in one_hot] })

            plt.clf()                
            sns.scatterplot(data=df_embeddings, x='x', y='y', hue="label", style='label', s=4)
            plt.legend()
            plt.grid(True)
            fig.savefig(os.path.join(self.mRootFolder, 'tsne_{0}.png'.format(pi)), bbox_inches='tight')

    def map_predictions(self, pPred, pLabels) :
        if pLabels is None :
            return pPred
        else :
            return [pLabels[i] for i in pPred]

    def __csvToDataset(self, pDataframe) :
        dataset = pDataframe.to_numpy()        
        X_test = dataset[:, 0]
        y_test = dataset[:, 1:]
        labels = list(pDataframe.columns)[1:]

        return X_test, y_test, np.array(labels)