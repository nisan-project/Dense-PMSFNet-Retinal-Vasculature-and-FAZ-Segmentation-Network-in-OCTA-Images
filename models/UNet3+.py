import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Activation, BatchNormalization, concatenate, Dropout

'''
UNet3+
'''

def conv_block(x, filter_size, size, dropout, batch_norm=False):
    conv = Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    conv = Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv

def encoder_block(x, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm):
    conv = conv_block(x, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool

def UNet3Plus(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True, deep_sup=True):
    FILTER_NUM = 64  
    FILTER_SIZE = 3  
    UP_SAMP_SIZE = 2  

    inputs = Input(input_shape, dtype=tf.float32)

    # Encoder 
    e1, p1 = encoder_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    e2, p2 = encoder_block(p1,FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    e3, p3 = encoder_block(p2, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    e4, p4 = encoder_block(p3, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    """ Bottleneck """
    e5 = Conv2D(16*FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(p4)
    e5 = Conv2D(16*FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e5)

    """ Decoder 4 """
    e1_d4 = MaxPooling2D(pool_size=(8, 8))(e1)
    e1_d4 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e1_d4)

    e2_d4 = MaxPooling2D(pool_size=(4, 4))(e2)
    e2_d4 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e2_d4)

    e3_d4 = MaxPooling2D(pool_size=(2, 2))(e3)
    e3_d4 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e3_d4)

    e4_d4 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e4)

    e5_d4 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(e5)
    e5_d4 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e5_d4)

    d4 = concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], axis=3)
    d4 = Conv2D(5*FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(d4)

    """ Decoder 3 """
    e1_d3 = MaxPooling2D(pool_size=(4, 4))(e1)
    e1_d3 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e1_d3)

    e2_d3 = MaxPooling2D(pool_size=(2, 2))(e2)
    e2_d3 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e2_d3)

    e3_d3 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e3)

    d4_d3 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(d4)
    d4_d3 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(d4_d3)

    e5_d3 = UpSampling2D(size=(2*UP_SAMP_SIZE, 2*UP_SAMP_SIZE), data_format="channels_last")(e5)
    e5_d3 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e5_d3)

    d3 = concatenate([e1_d3, e2_d3, e3_d3, d4_d3, e5_d3], axis=3) 
    d3 = Conv2D(5*FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(d3)

    """ Decoder 2 """
    e1_d2 = MaxPooling2D(pool_size=(2, 2))(e1)
    e1_d2 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e1_d2)

    e2_d2 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e2)

    d3_d2 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(d3)
    d3_d2 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(d3_d2)

    d4_d2 = UpSampling2D(size=(2*UP_SAMP_SIZE, 2*UP_SAMP_SIZE), data_format="channels_last")(d4)
    d4_d2 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(d4_d2)

    e5_d2 = UpSampling2D(size=(4*UP_SAMP_SIZE, 4*UP_SAMP_SIZE), data_format="channels_last")(e5)
    e5_d2 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e5_d2)

    d2 = concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2], axis=3)
    d2 = Conv2D(5*FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(d2)

    """ Decoder 1 """
    e1_d1 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e1)

    d2_d1 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(d2)
    d2_d1 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(d2_d1)

    d3_d1 = UpSampling2D(size=(2*UP_SAMP_SIZE, 2*UP_SAMP_SIZE), data_format="channels_last")(d3)
    d3_d1 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(d3_d1)

    d4_d1 = UpSampling2D(size=(4*UP_SAMP_SIZE, 4*UP_SAMP_SIZE), data_format="channels_last")(d4)
    d4_d1 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(d4_d1)

    e5_d1 = UpSampling2D(size=(8*UP_SAMP_SIZE, 8*UP_SAMP_SIZE), data_format="channels_last")(e5)
    e5_d1 = Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(e5_d1)

    d1 = concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1], axis=3)
    d1 = Conv2D(5*FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), padding="same")(d1)

    """ Deep Supervision """
    if deep_sup == True:
        y1 = Conv2D(NUM_CLASSES, kernel_size=(1, 1))(d1)
        y1 = Activation('sigmoid')(y1)

        y2 = Conv2D(NUM_CLASSES, kernel_size=(1, 1))(d2)
        y2 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(y2)
        y2 = Activation('sigmoid')(y2)

        y3 = Conv2D(NUM_CLASSES, kernel_size=(1, 1))(d3)
        y3 = UpSampling2D(size=(2*UP_SAMP_SIZE, 2*UP_SAMP_SIZE), data_format="channels_last")(y3)
        y3 = Activation('sigmoid')(y3)

        y4 = Conv2D(NUM_CLASSES, kernel_size=(1, 1))(d4)
        y4 = UpSampling2D(size=(4*UP_SAMP_SIZE, 4*UP_SAMP_SIZE), data_format="channels_last")(y4)
        y4 = Activation('sigmoid')(y4)

        y5 = Conv2D(NUM_CLASSES, kernel_size=(1, 1))(e5)
        y5 = UpSampling2D(size=(8*UP_SAMP_SIZE, 8*UP_SAMP_SIZE), data_format="channels_last")(y5)
        y5 = Activation('sigmoid')(y5)

        outputs = [y1, y2, y3, y4, y5]

    else:
        y1 = Conv2D(NUM_CLASSES, kernel_size=(1, 1))(d1)
        y1 = Activation("sigmoid")(y1)
        outputs = [y1]

    model = models.Model(inputs, outputs, name="UNet3+")
    print(model.summary())
    return model

input_shape = (304, 304, 3)
UNet3Plus(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True)
