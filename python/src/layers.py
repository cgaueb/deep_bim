import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Conv3D, MaxPooling3D, Activation

class voxel_encoder(tf.keras.layers.Layer) :
    def __init__(self, **pConfig) :
        super(voxel_encoder, self).__init__(name=pConfig['name'])
        self.config = pConfig

        self.encoder = Conv3D(
            filters     = self.config['filters'],
            kernel_size = self.config['kernel_size'],
            strides     = self.config['stride'],
            padding     = self.config['padding'],
            activation  = 'linear')

        self.activation = Activation(activation='elu')
        self.batchNorm = BatchNormalization()

        self.pooling = MaxPooling3D(
            pool_size   = self.config['pool_size'],
            strides     = self.config['pool_stride'],
            padding     = self.config['pool_padding'])

    def build(self, input_shape) :
        pass

    @tf.function
    def call(self, input):
        x = self.activation(self.batchNorm(self.encoder(input)))
        x = self.pooling(x)
        return x

class point_encoder(tf.keras.layers.Layer) :
    def __init__(self, **pConfig) :
        super(point_encoder, self).__init__(name=pConfig['name'])
        self.config = pConfig
        self.dense_layer1 = Dense(units=8, activation='elu')
        self.dense_layer2 = Dense(units=8, activation='elu')
        self.dense_layer3 = Dense(units=8, activation='elu')

    def build(self, input_shape) :
        pass

    @tf.function
    def call(self, input):
        per_point_f = self.dense_layer3(self.dense_layer2(self.dense_layer1(input)))
        return tf.reduce_max(per_point_f, axis=1)

class classifier(tf.keras.layers.Layer) :
    def __init__(self, **pConfig) :
        super(classifier, self).__init__(name=pConfig['name'])
        self.config = pConfig
        self.dense_layer = Dense(units=self.config['units'], activation='relu')
        self.drop_out = Dropout(rate=self.config['drop_rate'])
        self.classifier = Dense(units=self.config['num_classes'], activation='softmax')

    def build(self, input_shape) :
        pass

    @tf.function
    def call(self, input):
        return self.classifier(self.dense_layer(self.drop_out(input)))