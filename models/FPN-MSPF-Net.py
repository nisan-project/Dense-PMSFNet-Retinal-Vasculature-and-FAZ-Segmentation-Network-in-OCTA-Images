import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import backend as K

'''
FPN-MSPF-Net
'''
def fpn_dual_branch_module(FPN5, FPN4, num_classes=1, num_filters=128):

    # Low-scale branch
    # FPN5: Contextual supplier with ASPP
    aspp_1x1 = layers.Conv2D(num_filters, (1, 1), padding="same", activation="relu")(FPN5)
    aspp_3x3_r6 = layers.Conv2D(num_filters, (3, 3), dilation_rate=6, padding="same", activation="relu")(FPN5)
    aspp_3x3_r12 = layers.Conv2D(num_filters, (3, 3), dilation_rate=12, padding="same", activation="relu")(FPN5)
    aspp_3x3_r18 = layers.Conv2D(num_filters, (3, 3), dilation_rate=18, padding="same", activation="relu")(FPN5)
    image_pool = layers.GlobalAveragePooling2D()(FPN5)
    image_pool = layers.Reshape((1, 1, FPN5.shape[-1]))(image_pool)
    image_pool = layers.Conv2D(num_filters, (1, 1), padding="same", activation="relu")(image_pool)
    image_pool = layers.UpSampling2D(size=(FPN5.shape[1], FPN5.shape[2]), interpolation="bilinear")(image_pool)

    aspp_output = layers.Concatenate()([aspp_1x1, aspp_3x3_r6, aspp_3x3_r12, aspp_3x3_r18, image_pool])
    context_low = layers.Conv2D(num_filters, (1, 1), padding="same", activation="relu")(aspp_output)
    context_low = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(context_low)

    # FPN4: Spatial supplier
    spatial_low = layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(FPN4)
    
    if context_low.shape[1] != spatial_low.shape[1] or context_low.shape[2] != spatial_low.shape[2]:
            context_low = layers.Cropping2D(cropping=((1, 0), (0, 1)))(context_low)

    # Combine low-scale branch
    low_scale_output = layers.Concatenate()([context_low, spatial_low])
    low_scale_output = layers.Conv2D(num_classes, (3, 3), padding="same", activation="relu")(low_scale_output)
     
    return low_scale_output

def multi_scale_fusion(low_scale_output, high_scale_output,  num_classes=1):
    
    # Calculate attention mask: SA = 1 - S_lo
    attention_mask = 1.0 - low_scale_output
    low_scale_output_upsampled = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(low_scale_output)
    attention_mask_upsampled = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(attention_mask)

    # High-scale attention: SA_hi = (1 - S_lo) * S_hi
    high_scale_attention = attention_mask_upsampled * high_scale_output

    # Final segmentation: S = S_lo + SA_hi
    final_segmentation = low_scale_output_upsampled + high_scale_attention

    return final_segmentation

# def simple_encoder(input_tensor):

#     def encoder_block(x, num_filters):
#         x = layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
#         x = layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
#         return x, layers.MaxPooling2D((2, 2))(x)

#     c1, p1 = encoder_block(input_tensor, 64)   # Level 1: 1/2 resolution
#     c2, p2 = encoder_block(p1, 128)           # Level 2: 1/4 resolution
#     c3, p3 = encoder_block(p2, 256)           # Level 3: 1/8 resolution
#     c4, p4 = encoder_block(p3, 512)           # Level 4: 1/16 resolution
#     c5 = layers.Conv2D(1024, (3, 3), padding="same", activation="relu")(p4)  # Level 5: Bottleneck
#     return c1, c2, c3, c4, c5

def fpn_block(input_feature, skip_feature):
   
    upsampled = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(input_feature)
    skip = layers.Conv2D(256, (1, 1), padding="same", activation="relu")(skip_feature)

    upsampled_shape = K.int_shape(upsampled) 
    upsampled_height, upsampled_width = upsampled_shape[1], upsampled_shape[2]  

    def get_padding(upsampled_layer, input_height, input_width):
        upsampled_shape = K.int_shape(upsampled_layer)  
        up_h, up_w = upsampled_shape[1], upsampled_shape[2]
        pad_h = max(0, input_height - up_h)
        pad_w = max(0, input_width - up_w)
        return ((0, pad_h), (0, pad_w))

    padding = get_padding(skip, upsampled_height, upsampled_width)
    skip = layers.ZeroPadding2D(padding=padding)(skip)
    combined = layers.Add()([upsampled, skip])
    combined = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(combined)
    return combined


def FPN_MSPF_Net(input_shape, NUM_CLASSES=1):
   
    inputs = layers.Input(shape=input_shape)
    
    # Simple encoder
    #c1, c2, c3, c4, c5 = simple_encoder(inputs)

    backbone = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    c2 = backbone.get_layer('conv2_block3_out').output  # 1/4 resolution
    c3 = backbone.get_layer('conv3_block4_out').output  # 1/8 resolution
    c4 = backbone.get_layer('conv4_block6_out').output  # 1/16 resolution
    c5 = backbone.get_layer('conv5_block3_out').output  # 1/32 resolution


    # FPN layers
    p5 = layers.Conv2D(256, (1, 1), padding="same", activation="relu")(c5)  # FPN 5
    p4 = fpn_block(p5, c4)                                                  # FPN 4
    p3 = fpn_block(p4, c3)                                                  # FPN 3
    p2 = fpn_block(p3, c2) 
    
    low_scale = fpn_dual_branch_module(p5, p4, num_classes=1, num_filters=128)
    high_scale = fpn_dual_branch_module(p3, p2, num_classes=1, num_filters=128)
    
    final_output = multi_scale_fusion(low_scale, high_scale,  num_classes=1)
    final_output = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(final_output)
    #final_output = layers.Conv2D(NUM_CLASSES, (1, 1), padding='same', activation='sigmoid')(final_output)
    
    model = Model(inputs=inputs, outputs=final_output, name="FPN-MSPF-Net")
    print(model.summary())
    return model

#input_shape = (304, 304, 3)
#model = FPN_MSPF_Net(input_shape, NUM_CLASSES=1)
