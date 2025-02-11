import tensorflow as tf
from tensorflow.keras import layers, models

'''
UNet++
'''

def conv_block(x, filter_size, num_filters, dropout, batch_norm=False):
    conv = layers.Conv2D(num_filters, (filter_size, filter_size), padding="same")(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(num_filters, (filter_size, filter_size), padding="same")(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv

def up_conv_block(x, skip_connections, filter_size, num_filters, dropout, batch_norm):
    up = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(x)
    up = layers.concatenate([up] + skip_connections, axis=3)
    return conv_block(up, filter_size, num_filters, dropout, batch_norm)

def UNetPlusPlus(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    FILTER_NUM = 64  
    FILTER_SIZE = 3  

    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Encoder 
    X_00 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_00 = layers.MaxPooling2D(pool_size=(2, 2))(X_00)

    X_10 = conv_block(pool_00, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_10 = layers.MaxPooling2D(pool_size=(2, 2))(X_10)

    X_20 = conv_block(pool_10, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_20 = layers.MaxPooling2D(pool_size=(2, 2))(X_20)

    X_30 = conv_block(pool_20, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_30 = layers.MaxPooling2D(pool_size=(2, 2))(X_30)

    X_40 = conv_block(pool_30, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Decoder 
    X_01 = up_conv_block(X_10, [X_00], FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    X_11 = up_conv_block(X_20, [X_10], FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    X_21 = up_conv_block(X_30, [X_20], FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    X_31 = up_conv_block(X_40, [X_30], FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    X_02 = up_conv_block(X_11, [X_00, X_01], FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    X_12 = up_conv_block(X_21, [X_10, X_11], FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    X_22 = up_conv_block(X_31, [X_20, X_21], FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    X_03 = up_conv_block(X_12, [X_00, X_01, X_02], FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    X_13 = up_conv_block(X_22, [X_10, X_11, X_12], FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    X_04 = up_conv_block(X_13, [X_00, X_01, X_02, X_03], FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Final 1x1 Convolution Layer
    final = layers.concatenate([X_01, X_02, X_03, X_04], axis=3)
    final = layers.Conv2D(NUM_CLASSES, kernel_size=(1, 1))(final)
    final = layers.BatchNormalization(axis=3)(final)
    final = layers.Activation('sigmoid')(final)

    model = models.Model(inputs, final, name="UNet++")
    print(model.summary())
    return model

#input_shape = (304, 304, 3)
#UNetPlusPlus(input_shape, NUM_CLASSES=3, dropout_rate=0.1, batch_norm=True)