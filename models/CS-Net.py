import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

'''
CS-Net
'''

def downsample():
    return layers.MaxPooling2D(pool_size=(2, 2), strides=2)

def deconv(out_channels):
    return layers.Conv2DTranspose(out_channels, kernel_size=2, strides=2, padding="same")

def res_encoder(in_channels, out_channels):
    inputs = layers.Input(shape=(None, None, in_channels))
    residual = layers.Conv2D(out_channels, kernel_size=1)(inputs)
    x = layers.Conv2D(out_channels, kernel_size=3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(out_channels, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Add()([x, residual])
    x = layers.Activation("relu")(x)
    return keras.Model(inputs, x)

def decoder(in_channels, out_channels):
    inputs = layers.Input(shape=(None, None, in_channels))
    x = layers.Conv2D(out_channels, kernel_size=3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(out_channels, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return keras.Model(inputs, x)

def affinity_attention(in_channels):
    inputs = layers.Input(shape=(None, None, in_channels))
    gamma = tf.Variable(tf.zeros(1), trainable=True)
    x = layers.Lambda(lambda x: gamma * x + x)(inputs)
    return keras.Model(inputs, x)

def CS_Net(input_shape, NUM_CLASSES=1):
    inputs = layers.Input(input_shape, dtype=tf.float32)
    enc_input = res_encoder(3, 32)(inputs)
    down1 = downsample()(enc_input)
    enc1 = res_encoder(32, 64)(down1)
    down2 = downsample()(enc1)
    enc2 = res_encoder(64, 128)(down2)
    down3 = downsample()(enc2)
    enc3 = res_encoder(128, 256)(down3)
    down4 = downsample()(enc3)
    input_feature = res_encoder(256, 512)(down4)
    attention = affinity_attention(512)(input_feature)
    attention_fuse = layers.Conv2D(512, kernel_size=1)(layers.Concatenate()([input_feature, attention]))
    up4 = deconv(256)(attention_fuse)
    up4 = layers.Concatenate()([enc3, up4])
    dec4 = decoder(512, 256)(up4)
    up3 = deconv(128)(dec4)
    up3 = layers.Concatenate()([enc2, up3])
    dec3 = decoder(256, 128)(up3)
    up2 = deconv(64)(dec3)
    up2 = layers.Concatenate()([enc1, up2])
    dec2 = decoder(128, 64)(up2)
    up1 = deconv(32)(dec2)
    up1 = layers.Concatenate()([enc_input, up1])
    dec1 = decoder(64, 32)(up1)
    final = layers.Conv2D(NUM_CLASSES, kernel_size=1, activation="sigmoid")(dec1)
    model = models.Model(inputs, final, name="CS-Net")
    print(model.summary())
    return model

#input_shape = (304, 304, 3)
#model = CS_Net(input_shape, NUM_CLASSES=1)