import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K

'''
PCAM-PRDC-Net
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

def PRDC_block(x, filter_size, size, dropout=0.1, batch_norm=False):
    
    conv_1 = layers.BatchNormalization(axis=3)(x)
    conv_1 = layers.Activation("relu")(conv_1)
    conv_1 = layers.Conv2D(size, (filter_size, filter_size), strides=(1,1), padding="same")(conv_1)
    conv_1 = DropBlock2D(keep_prob=0.5, block_size=3)(conv_1,training=True)
    conv_1 = layers.Dropout(dropout)(conv_1)
    
    conv_2 = layers.BatchNormalization(axis=3)(conv_1)
    conv_2 = layers.Activation("relu")(conv_2)
    conv_2 = layers.Conv2D(size, (filter_size, filter_size), strides = (1,1), padding="same")(conv_2)
    conv_2 = DropBlock2D(keep_prob=0.5, block_size=3)(conv_2,training=True)
    conv_2 = layers.Dropout(dropout)(conv_2)
    
    conv_res = layers.Conv2D(size, kernel_size=(1, 1))(x)
        
    residual = layers.Add()([conv_res, conv_2])

    return residual

def spatial_pyramid_pooling(input_tensor):
   
    input_shape = tf.keras.backend.int_shape(input_tensor)  
    height, width, channels = input_shape[1], input_shape[2], input_shape[3]

    pool_sizes = [2, 3, 5, 6]

    pooled_tensors = [
        layers.MaxPooling2D(pool_size=(size, size), strides=(size, size), padding='same')(input_tensor)
        for size in pool_sizes
    ]

    convolved_tensors = [
        layers.Conv2D(filters=channels // 2, kernel_size=1, activation='relu')(pooled)
        for pooled in pooled_tensors
    ]

    upsampled_tensors = [
        tf.image.resize(convolved, size=(height, width), method='bilinear')
        for convolved in convolved_tensors
    ]

    concatenated_features = layers.Concatenate(axis=-1)(upsampled_tensors)

    return layers.Concatenate(axis=-1)([input_tensor, concatenated_features])


# Channel Attention Module (CAM)
def channel_attention_module(input_tensor, reduction_ratio=8):

    avg_pooled = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    max_pooled = tf.reduce_max(input_tensor, axis=[1, 2], keepdims=True)

    channel_dim = input_tensor.shape[-1]
    mlp = tf.keras.Sequential([
        layers.Dense(channel_dim // reduction_ratio, activation='relu'),
        layers.Dense(channel_dim)
    ])

    avg_out = mlp(avg_pooled)
    max_out = mlp(max_pooled)

    attention = tf.nn.sigmoid(avg_out + max_out)
    return input_tensor * attention

# Pyramid Channel Attention Module (PCAM)
def PCAM_block(input_tensor):
  
    spp_features = spatial_pyramid_pooling(input_tensor)
    cam_features = channel_attention_module(spp_features)
    enhanced_features = spp_features * cam_features
    output = layers.Concatenate(axis=-1)([input_tensor, enhanced_features])

    return output


def PCAM_PRDC_Net(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
   
    FILTER_NUM = 64  
    FILTER_SIZE = 3  
    UP_SAMP_SIZE = 2 
       
    inputs = layers.Input(input_shape, dtype=tf.float32)
    
    # Downsampling layers
    # Down 1
    conv_128 = PRDC_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    conv_128 = PCAM_block(conv_128)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    
    # Down 2
    conv_64 = PRDC_block(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    conv_64 = PCAM_block(conv_64)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    
    # Down 3
    conv_32 = PRDC_block(pool_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    conv_32 = PCAM_block(conv_32)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
    
    # Down 4
    conv_16 = PRDC_block(pool_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    conv_16 = PCAM_block(conv_16)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)
    
    # Down 5
    conv_8 = PRDC_block(pool_8, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)
    
    # Upsampling layers
    # Up 6
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE),interpolation='bilinear', data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)
    up_conv_16 = PRDC_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    
    # Up 7
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE),interpolation='bilinear', data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = PRDC_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    
    # Up 8
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE),interpolation='bilinear', data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = PRDC_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    
    # Up 9
    up_128 = p_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE),interpolation='bilinear', data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = PRDC_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layer
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1, 1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)

    model = models.Model(inputs, conv_final, name="PCAM-PRDC-Net")
    print(model.summary())
    return model

#input_shape = (304,304,3)
#PCAM_PRDC_Net(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True)
