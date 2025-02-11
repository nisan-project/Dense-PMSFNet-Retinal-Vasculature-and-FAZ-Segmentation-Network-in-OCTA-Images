import tensorflow as tf
from tensorflow.keras import layers, models

'''
AV-Net
'''

def conv_block(x, channels, batch_norm=False):
    conv = layers.Conv2D(channels, kernel_size=3, padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(channels, kernel_size=3, padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    return conv

def dense_block(x, channels, n):
    for _ in range(n):
        x1 = conv_block(x, channels)
        x = layers.concatenate([x, x1], axis=-1)
    x = layers.Conv2D(channels, kernel_size=1, padding="same")(x)
    x = layers.Activation("relu")(x)
    return x

def decoder_block(x1, x2, channels):
    x = layers.concatenate([x1, x2], axis=-1)
    x = layers.Conv2D(channels, kernel_size=3, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(channels, kernel_size=3, padding="same")(x)
    x = layers.Activation("relu")(x)
    return x

def AVNet(input_shape, NUM_CLASSES=1, channels=64):

    inputs = layers.Input(input_shape, dtype=tf.float32)
    
    # Downsampling
    conv1 = layers.Conv2D(channels, kernel_size=7, padding="same")(inputs)
    conv1 = layers.Activation("relu")(conv1)
    maxpool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    dense1 = dense_block(maxpool1, channels, 6)
    avgpool1 = layers.AvgPool2D(pool_size=(2, 2))(dense1)

    dense2 = dense_block(avgpool1, channels, 12)
    avgpool2 = layers.AvgPool2D(pool_size=(2, 2))(dense2)

    dense3 = dense_block(avgpool2, channels, 24)
    avgpool3 = layers.AvgPool2D(pool_size=(2, 2))(dense3)

    dense4 = dense_block(avgpool3, channels, 16)

    # Upsampling
    upsample1 = layers.UpSampling2D(size=(2, 2))(dense4)
    decode1 = decoder_block(upsample1, dense3, channels)

    upsample2 = layers.UpSampling2D(size=(2, 2))(decode1)
    decode2 = decoder_block(upsample2, dense2, channels)

    upsample3 = layers.UpSampling2D(size=(2, 2))(decode2)
    decode3 = decoder_block(upsample3, dense1, channels)

    upsample4 = layers.UpSampling2D(size=(2, 2))(decode3)
    decode4 = decoder_block(upsample4, conv1, channels)

    output = layers.Conv2D(NUM_CLASSES=1, kernel_size=1)(decode4)
    
    model = models.Model(inputs, output)
    print(model.summary())
    return model

input_shape = (304, 304, 3)
model = AVNet(input_shape, NUM_CLASSES=1, channels=64)