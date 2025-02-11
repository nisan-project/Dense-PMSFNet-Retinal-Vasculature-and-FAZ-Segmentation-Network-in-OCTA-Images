import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import  Conv2D, GlobalAveragePooling2D, Concatenate, Reshape, Add, Multiply, BatchNormalization, ReLU, Softmax

'''
MS-Net
'''

def conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    conv_1 = layers.Conv2D(size, (filter_size, filter_size), strides=(1,1), padding="same")(x)
    if batch_norm is True:
        conv_1 = layers.BatchNormalization(axis=3)(conv_1)
    conv_1 = layers.Activation("relu")(conv_1)

    conv_2 = layers.Conv2D(size, (filter_size, filter_size), strides = (1,1), padding="same")(conv_1)
    if batch_norm is True:
        conv_2 = layers.BatchNormalization(axis=3)(conv_2)
    conv_2 = layers.Activation("relu")(conv_2)

    return conv_2

def multi_scale_attention_module(input_tensor, dilation_rates):
    
    pooled = GlobalAveragePooling2D(keepdims=True)(input_tensor)
    pooled = Reshape((1, 1, input_tensor.shape[-1]))(pooled)
    conv1_i = Conv2D(filters=input_tensor.shape[-1], kernel_size=3, padding="same", activation="relu")(pooled)
    conv1_ii = Conv2D(filters=input_tensor.shape[-1], kernel_size=3, padding="same", activation="relu")(conv1_i)

    dilated_features = []
    for rate in dilation_rates:
        dilated_conv = Conv2D(filters=input_tensor.shape[-1], 
                              kernel_size=3, 
                              padding="same", 
                              dilation_rate=rate, 
                              activation="relu")(conv1_ii)
        dilated_features.append(dilated_conv)

    multi_scale_features = Concatenate()(dilated_features)

    fused_features = Conv2D(filters=input_tensor.shape[-1], kernel_size=3, padding="same")(multi_scale_features)
    fused_features = BatchNormalization()(fused_features)
    fused_features = ReLU()(fused_features)

    output = Add()([input_tensor, fused_features])
    output = ReLU()(output)

    return output

def feature_perception_module(Fa, Fb, name="FPM"):
    
    F_concat = Concatenate()([Fa, Fb])

    F_conv1 = Conv2D(Fa.shape[-1], kernel_size=3, padding="same")(F_concat)
    F_conv1 = BatchNormalization()(F_conv1)
    F_conv1 = ReLU()(F_conv1)
    F_conv1 = Conv2D(Fa.shape[-1], kernel_size=3, padding="same")(F_conv1)
    F_conv1 = BatchNormalization()(F_conv1)
    F_conv1 = ReLU()(F_conv1)

    F_conv2 = Conv2D(Fa.shape[-1], kernel_size=1, padding="same", activation="relu")(F_concat)

    F_conv3 = Conv2D(Fa.shape[-1], kernel_size=3, padding="same", activation="relu")(F_conv1 + F_conv2)
    F_conv3 = Conv2D(Fa.shape[-1], kernel_size=3, padding="same")(F_conv3)

    F_weights = Softmax(axis=-1)(F_conv3)

    fout1 = F_weights
    fout2 = F_weights
    
    return fout1, fout2


def stacked_feature_pyramid_module(input_tensor, name="SFPM"):
    
    F1 = Conv2D(input_tensor.shape[-1], kernel_size=3, dilation_rate=1, padding="same", activation="relu")(input_tensor)
    F2 = Conv2D(input_tensor.shape[-1], kernel_size=3, dilation_rate=2, padding="same", activation="relu")(input_tensor)
    F4 = Conv2D(input_tensor.shape[-1], kernel_size=3, dilation_rate=4, padding="same", activation="relu")(input_tensor)

    f1, f2 = feature_perception_module(F1, F2, name="FPM_12")
    F12 = Add()([Multiply()([F1, f1]), Multiply()([F2, f2])])

    f2, f4 = feature_perception_module(F2, F4, name="FPM_24")
    F24 = Add()([Multiply()([F2, f2]), Multiply()([F4, f4])])

    f12, f24 = feature_perception_module(F12, F24, name="FPM_1224")
    F_final = Add()([input_tensor, Add()([Multiply()([F12, f12]), Multiply()([F24, f24])])])
    
    return F_final


def MS_Net(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):

    FILTER_NUM = 16
    FILTER_SIZE = 3 
       
    inputs = layers.Input(input_shape, dtype=tf.float32)
    
    # Downsampling layers
    # Down 1
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    
    # Down 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    
    # Down 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
    
    # Down 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)
    
    # Down 5
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)
    sfpm = stacked_feature_pyramid_module(conv_8)
    
    # Upsampling layers
    # Up 6
    up_16 = layers.Conv2DTranspose(8*FILTER_NUM, kernel_size=2, strides=2, padding='same')(sfpm)
    msam_3 = multi_scale_attention_module(conv_16, dilation_rates=[5,7])
    up_16 = layers.concatenate([up_16, msam_3], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    
    # Up 7
    up_32 = layers.Conv2DTranspose(4*FILTER_NUM, kernel_size=2, strides=2, padding='same')(up_conv_16)
    msam_2 = multi_scale_attention_module(conv_32, dilation_rates=[3,5])
    up_32 = layers.concatenate([up_32, msam_2], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    
    # Up 8
    up_64 = layers.Conv2DTranspose(2*FILTER_NUM, kernel_size=2, strides=2, padding='same')(up_conv_32)
    msam_1 = multi_scale_attention_module(conv_64, dilation_rates=[1,3])
    up_64 = layers.concatenate([up_64, msam_1], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    
    # Up 9
    up_128 = layers.Conv2DTranspose(FILTER_NUM, kernel_size=2, strides=2, padding='same')(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layer
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1, 1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  
    model = models.Model(inputs, conv_final, name="MS-Net")
    print(model.summary())
    return model

#input_shape = (304,304,3)
#MS_Net(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True)
