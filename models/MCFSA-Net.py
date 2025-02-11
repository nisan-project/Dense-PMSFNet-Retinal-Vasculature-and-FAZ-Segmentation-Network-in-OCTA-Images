#MCSFA-Net
import tensorflow as tf
from tensorflow.keras import models, layers, backend as K
from tensorflow.keras.utils import register_keras_serializable
from tensorflow_addons.layers import GroupNormalization

'''
MCSFA-Net
'''

def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)))

@register_keras_serializable(package="Custom")
class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True,name=None, **kwargs):
        super(DropBlock2D, self).__init__(name="DropBlock2D")
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.names = name
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale
        super(DropBlock2D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update( {"block_size": self.block_size,"keep_prob": self.keep_prob,"name": self.names })

        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        self.h = input_shape[1]  
        self.w = input_shape[2] 
        self.channel = input_shape[3]  
        #_, self.h, self.w, self.channel = input_shape.as_list()
        #_, self.h, self.w, self.channel = input_shape
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output *tf.cast(tf.size(mask), dtype=tf.float32)  / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.cast(self.w, dtype=tf.float32), tf.cast(self.h, dtype=tf.float32)

        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask


# HC-Drop Module
def HCDropModule(x, filter_size, size, block_size=3, keep_prob=0.9, dilation_rate=2):
    conv1 = layers.Conv2D(size, (filter_size, filter_size), dilation_rate=dilation_rate, strides=(1,1), padding="same")(x)
    conv2 = layers.Conv2D(size, (filter_size, filter_size), strides=(1,1), padding="same")(x)
    dropblock1 = DropBlock2D(keep_prob=0.9, block_size=3)(conv1,training=True)
    dropblock2 = DropBlock2D(keep_prob=0.9, block_size=3)(conv2,training=True)
    addblock = layers.Add()([dropblock1, dropblock2])
    group_norm =  GroupNormalization(groups=8)(addblock)
    linear_correction = layers.Conv2D(size, 1, activation='relu')(group_norm)
    return linear_correction


# Squeeze-and-Excitation (SE) Channel Attention
def se_channel_attention(feature_map, reduction_ratio=16):
    avg_pool = layers.GlobalAveragePooling2D()(feature_map)
    max_pool = layers.GlobalMaxPooling2D()(feature_map)

    avg_dense = layers.Dense(feature_map.shape[-1] // reduction_ratio, activation='relu')(avg_pool)
    max_dense = layers.Dense(feature_map.shape[-1] // reduction_ratio, activation='relu')(max_pool)

    combined = layers.Add()([avg_dense, max_dense])
    attention = layers.Dense(feature_map.shape[-1], activation='sigmoid')(combined)

    attention = layers.Reshape((1, 1, feature_map.shape[-1]))(attention)

    refined_feature_map = layers.Multiply()([feature_map, attention])
    return refined_feature_map

# Channel Cooperative Attention Fusion (CCAF) Module
def ccaf_module(F1, F2, F3, reduction_ratio=16):
    F1_se = se_channel_attention(F1, reduction_ratio)
    F2_se = se_channel_attention(F2, reduction_ratio)
    F3_se = se_channel_attention(F3, reduction_ratio)

    stacked_features = layers.Concatenate(axis=-1)([F1_se, F2_se, F3_se])

    softmax_weights = layers.Softmax(axis=-1)(stacked_features)

    refined_feature_map = layers.Multiply()([stacked_features, softmax_weights])

    return refined_feature_map


# Gaussian-like spatial activation function
def spatial_activation(prob_map):
    return tf.exp(tf.square(prob_map - 0.5) / 0.25) + 1

# Global Spatial Activation (GSA) Module
def gsa_module(feature_map):
    avg_pool = layers.GlobalAveragePooling2D()(feature_map)
    max_pool = layers.GlobalMaxPooling2D()(feature_map)

    avg_conv = layers.Reshape((1, 1, feature_map.shape[-1]))(avg_pool)
    max_conv = layers.Reshape((1, 1, feature_map.shape[-1]))(max_pool)

    prob_map = layers.Concatenate(axis=-1)([avg_conv, max_conv])
    prob_map = layers.Conv2D(1, (1, 1), padding='same')(prob_map) 
    prob_map = layers.Activation('sigmoid')(prob_map)  

    spatial_map = spatial_activation(prob_map)

    activated_feature_map = layers.Multiply()([feature_map, spatial_map])

    return activated_feature_map


def conv_block(x, filter_size, size, dropout, batch_norm=False):

    conv_1 = layers.Conv2D(size, (filter_size, filter_size), strides=(1,1), padding="same")(x)
    conv_1 = layers.BatchNormalization(axis=3)(conv_1)
    conv_1 = layers.Activation("relu")(conv_1)

    conv_2 = layers.Conv2D(size, (filter_size, filter_size), strides = (1,1), padding="same")(conv_1)
    conv_2 = layers.BatchNormalization(axis=3)(conv_2)
    conv_2 = layers.Activation("relu")(conv_2)

    return conv_2


def MCSFA_Net(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    
    FILTER_NUM = 64  
    FILTER_SIZE = 3   

    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # Down 1
    conv_128 = HCDropModule(inputs, FILTER_SIZE, FILTER_NUM, block_size=3, keep_prob=0.8, dilation_rate=2)
    conv_128 = layers.Conv2D(FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(conv_128)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    # Down 2
    conv_64 = HCDropModule(pool_64, FILTER_SIZE, 2*FILTER_NUM, block_size=3, keep_prob=0.8, dilation_rate=2)
    conv_64 = layers.Conv2D(2*FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(conv_64)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    # Down 3
    conv_32 = HCDropModule(pool_32, FILTER_SIZE, 4*FILTER_NUM, block_size=3, keep_prob=0.8, dilation_rate=2)
    conv_32 = layers.Conv2D(4*FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(conv_32)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)

    # Bottleneck
    conv_16 = layers.Conv2D(8*FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(pool_16)
    Feature3 = layers.Conv2D(8*FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(conv_16)
    Feature1 = layers.MaxPooling2D(pool_size=(8, 8))(conv_128)
    Feature1 = conv_block(Feature1, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    Feature2 = layers.MaxPooling2D(pool_size=(4, 4))(conv_64)
    Feature2 = conv_block(Feature2, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    ccaf = ccaf_module(Feature1, Feature2, Feature3, reduction_ratio=16)

    # Up 4
    up_32 = layers.Conv2DTranspose(4*FILTER_NUM, kernel_size=2, strides=2, padding='same')(ccaf)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    layers.Conv2D(4*FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(up_32)
    up_32 = gsa_module(up_32)
    up_conv_32 = layers.Conv2D(4*FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(up_32)

    # Up 5
    up_64 = layers.Conv2DTranspose(2*FILTER_NUM, kernel_size=2, strides=2, padding='same')(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_64 = layers.Conv2D(2*FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(up_64)
    up_64 = gsa_module(up_64)
    up_conv_64 = layers.Conv2D(2*FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(up_64)

    # Up 7
    up_128 = layers.Conv2DTranspose(FILTER_NUM, kernel_size=2, strides=2, padding='same')(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_128 = layers.Conv2D(8*FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(up_128)
    up_128 = gsa_module(up_128)
    up_conv_128 = layers.Conv2D(8*FILTER_NUM, (FILTER_SIZE,FILTER_SIZE), strides=(1,1), activation='relu', padding="same")(up_128)

    # 1*1 convolutional layer
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1, 1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  

    model = models.Model(inputs, conv_final, name="MCSFA-Net")
    print(model.summary())
    return model

#input_shape = (304,304,3)
#MCSFA_Net(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True)