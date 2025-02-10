import tensorflow as tf
from tensorflow.keras import models, layers, backend as K
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D, Concatenate, Add
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply
from tensorflow.keras.applications import DenseNet121

def mspfm(input_tensor, filters):
    """
    Multi-Scale Pyramidal Fusion Module (MSPFM).
    """
    input_shape = K.int_shape(input_tensor) 
    input_height, input_width = input_shape[1], input_shape[2]  

    conv1x1 = Conv2D(filters, (1, 1), padding="same", activation="relu")(input_tensor)
    conv3x3_1 = Conv2D(filters, (3, 3), padding="same", activation="relu")(input_tensor)
    conv5x5_1 = Conv2D(filters, (5, 5), padding="same", activation="relu")(input_tensor)

    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2))(input_tensor)
    conv3x3_2 = Conv2D(filters, (3, 3), padding="same", activation="relu")(avg_pool_2x2)

    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4))(input_tensor)
    conv3x3_3 = Conv2D(filters, (3, 3), padding="same", activation="relu")(avg_pool_4x4)

    avg_pool_6x6 = AveragePooling2D(pool_size=(6, 6))(input_tensor)
    conv3x3_4 = Conv2D(filters, (3, 3), padding="same", activation="relu")(avg_pool_6x6)

    upsample_2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(conv3x3_2)
    upsample_4 = UpSampling2D(size=(4, 4), interpolation="bilinear")(conv3x3_3)
    upsample_6 = UpSampling2D(size=(6, 6), interpolation="bilinear")(conv3x3_4)
    
    def get_padding(upsampled_layer, input_height, input_width):
        upsampled_shape = K.int_shape(upsampled_layer)  
        up_h, up_w = upsampled_shape[1], upsampled_shape[2]
        pad_h = max(0, input_height - up_h)
        pad_w = max(0, input_width - up_w)
        return ((0, pad_h), (0, pad_w))

    if input_height and input_width:  
        padding1 = get_padding(upsample_2, input_height, input_width)
        padding2 = get_padding(upsample_4, input_height, input_width)
        padding3 = get_padding(upsample_6, input_height, input_width)

        upsample_2 = layers.ZeroPadding2D(padding=padding1)(upsample_2)
        upsample_4 = layers.ZeroPadding2D(padding=padding2)(upsample_4)
        upsample_6 = layers.ZeroPadding2D(padding=padding3)(upsample_6)

    concatenated1 = Concatenate()([input_tensor, conv1x1, conv3x3_1, conv5x5_1])
    concatenated2 = Concatenate()([input_tensor, upsample_2, upsample_4, upsample_6])
    concat_output = Add()([concatenated1, concatenated2])

    final_conv = Conv2D(filters, (1, 1), padding="same", activation="relu")(concat_output)
    return final_conv


def sSE_block(input_tensor):
    """
    Spatial Squeeze and Excitation (sSE) block.
    """
    conv1x1 = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(input_tensor)
    return Multiply()([input_tensor, conv1x1])


def cSE_block(input_tensor, filters):
    """
    Channel Squeeze and Excitation (cSE) block.
    """
    gap = GlobalAveragePooling2D()(input_tensor)
    dense1 = Dense(filters // 2, activation="relu")(gap)  
    dense2 = Dense(filters, activation="sigmoid")(dense1)  
    channel_weights = Reshape((1, 1, filters))(dense2)
    return Multiply()([input_tensor, channel_weights])


def scSE_block(input_tensor, filters):
    """
    Combined sSE and cSE block.
    """
    sse = sSE_block(input_tensor)
    cse = cSE_block(input_tensor, filters)
    return Add()([sse, cse])  


def module_with_scSE(input_tensor, filters):
    """
    The overall module containing:
    - Two 3x3 Conv layers
    - scSE block
    """

    conv1 = Conv2D(filters, (3, 3), padding="same", activation="relu")(input_tensor)
    conv1 = layers.BatchNormalization(axis=3)(conv1)
    conv1 = layers.Activation("relu")(conv1)
 
    conv2 = Conv2D(filters, (3, 3), padding="same", activation="relu")(conv1)
    conv2 = layers.BatchNormalization(axis=3)(conv2)
    conv2 = layers.Activation("relu")(conv2)

    scse_output = scSE_block(conv2, filters)

    return scse_output


def DensePMSFNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    FILTER_NUM = 64   
    UP_SAMP_SIZE = 2  

    Backbone = DenseNet121(include_top=False,weights="imagenet",input_shape=input_shape)
    Backbone.trainable = True
    for layer in Backbone.layers[:52]:
        layer.trainable = False

    inputs = Backbone.input
    s1 = Backbone.layers[4].output
    s2 = Backbone.layers[51].output
    s3 = Backbone.layers[139].output
    s4 = Backbone.layers[311].output
    b1= Backbone.layers[426].output

    sapp_bn = mspfm(b1, 4*FILTER_NUM)

    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), interpolation='bilinear', data_format="channels_last")(sapp_bn)
    sapp_4 = mspfm(s4, 4*FILTER_NUM)
    up_32 = layers.concatenate([up_32, sapp_4], axis=3)
    up_conv_32 = module_with_scSE(up_32, 4*FILTER_NUM)
    
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), interpolation='bilinear', data_format="channels_last")(up_conv_32)
    sapp_3 = mspfm(s3, 4*FILTER_NUM)
    up_64 = layers.concatenate([up_64, sapp_3], axis=3)
    up_conv_64 = module_with_scSE(up_64, 4*FILTER_NUM)

    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE),interpolation='bilinear', data_format="channels_last")(up_conv_64)
    sapp_2 = mspfm(s2, 2*FILTER_NUM)
    up_128 = layers.concatenate([up_128, sapp_2], axis=3)
    up_conv_128 = module_with_scSE(up_128, 2*FILTER_NUM)

    up_256 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE),interpolation='bilinear', data_format="channels_last")(up_conv_128)
    sapp_1 = mspfm(s1, FILTER_NUM)
    up_256 = layers.concatenate([up_256, sapp_1], axis=3)
    up_conv_256 = module_with_scSE(up_256, FILTER_NUM)

    up_conv_32_adjusted = layers.UpSampling2D(size=(8*UP_SAMP_SIZE, 8*UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_conv_64_adjusted = layers.UpSampling2D(size=(4*UP_SAMP_SIZE, 4*UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_conv_128_adjusted = layers.UpSampling2D(size=(2*UP_SAMP_SIZE, 2*UP_SAMP_SIZE), data_format="channels_last")(up_conv_128)
    up_conv_256_adjusted = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_256)

    deepfusion = layers.concatenate([up_conv_32_adjusted, up_conv_64_adjusted, up_conv_128_adjusted, up_conv_256_adjusted])

    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1, 1))(deepfusion)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  

    model = models.Model(inputs, conv_final, name="DensePMSFNet")
    print(model.summary())
    return model

#input_shape = (304,304,3)
#DensePMSFNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True)
