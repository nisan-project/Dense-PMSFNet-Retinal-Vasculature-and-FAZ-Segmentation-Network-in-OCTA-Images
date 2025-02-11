import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers, Sequential, models
from tensorflow.keras.applications import ResNet50V2
import tensorflow_addons as tfa

'''
TransUNet
'''
L2_WEIGHT_DECAY = 1e-4

class PositionalEmbedding(layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.trainable = trainable

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected input shape with 3 dimensions, got {len(input_shape)}")
        
        self.position_embedding = self.add_weight(
            shape=(1, input_shape[1], input_shape[2]),
            initializer=tf.random_normal_initializer(stddev=0.06),
            trainable=self.trainable,
            name="position_embedding",
        )

    def call(self, inputs):
        return inputs + tf.cast(self.position_embedding, dtype=inputs.dtype)


class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        if hidden_size % self.num_heads != 0:
            raise ValueError(
                f"Hidden size ({hidden_size}) must be divisible by number of heads ({self.num_heads})"
            )
        
        self.projection_dim = hidden_size // self.num_heads
        self.query_dense = layers.Dense(hidden_size, name="query")
        self.key_dense = layers.Dense(hidden_size, name="key")
        self.value_dense = layers.Dense(hidden_size, name="value")
        self.output_dense = layers.Dense(hidden_size, name="output")

    def compute_attention(self, query, key, value):
        scores = tf.matmul(query, key, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(tf.shape(key)[-1], scores.dtype))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(attention_weights, value), attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.split_heads(self.query_dense(inputs), batch_size)
        key = self.split_heads(self.key_dense(inputs), batch_size)
        value = self.split_heads(self.value_dense(inputs), batch_size)
        
        attention_output, weights = self.compute_attention(query, key, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concatenated_attention = tf.reshape(attention_output, (batch_size, -1, self.projection_dim * self.num_heads))
        return self.output_dense(concatenated_attention), weights


class TransformerLayer(layers.Layer):

    def __init__(self, num_heads, mlp_units, dropout_rate, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.num_heads = num_heads
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attention_layer = MultiHeadAttention(num_heads=self.num_heads, name="multihead_attention")
        self.mlp_block = layers.Sequential(
            [
                layers.Dense(self.mlp_units, activation="linear", name="mlp_dense_0"),
                layers.Activation(lambda x: tfa.activations.gelu(x, approximate=False), name="gelu_activation"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(input_shape[-1], name="mlp_dense_1"),
                layers.Dropout(self.dropout_rate),
            ],
            name="mlp_block",
        )
        self.norm_layer1 = layers.LayerNormalization(epsilon=1e-6, name="layer_norm_1")
        self.norm_layer2 = layers.LayerNormalization(epsilon=1e-6, name="layer_norm_2")
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        x = self.norm_layer1(inputs)
        attention_output, _ = self.attention_layer(x)
        attention_output = self.dropout(attention_output, training=training)
        x = x + attention_output
        
        y = self.norm_layer2(x)
        y = self.mlp_block(y)
        return x + y

def ws_reg(kernel):
    kernel_mean, kernel_std = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
    kernel = (kernel - kernel_mean) / (kernel_std + 1e-5)

def conv3x3(cout, stride=1, groups=1, bias=False, name=""):
    return layers.Conv2D(
        cout, kernel_size=3, strides=stride, padding="same", use_bias=bias, groups=groups,
        name=name, kernel_regularizer=ws_reg
    )

def conv1x1(cout, stride=1, groups=1, bias=False, name=""):
    return layers.Conv2D(
        cout, kernel_size=1, strides=stride, padding="same", use_bias=bias, groups=groups,
        name=name, kernel_regularizer=ws_reg
    )

class SegmentationHead(layers.Layer):
    def __init__(self, num_classes=9, kernel_size=1, activation='sigmoid', name="segmentation_head", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.activation = activation

    def build(self, input_shape):
        self.conv_layer = layers.Conv2D(
            filters=self.num_classes, kernel_size=self.kernel_size, padding="same",
            kernel_regularizer = regularizers.L2(L2_WEIGHT_DECAY), 
            kernel_initializer = initializers.LecunNormal())
        self.activation_layer = layers.Activation(self.activation)

    def call(self, inputs):
        x = self.conv_layer(inputs)
        x = self.activation_layer(x)
        return x

class Conv2DReLU(layers.Layer):
    def __init__(self, filters, kernel_size, padding="same", strides=1, name="conv2d_relu", **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

    def build(self, input_shape):
        self.conv_layer = layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
            padding=self.padding, use_bias=False, kernel_regularizer = regularizers.L2(L2_WEIGHT_DECAY), 
            kernel_initializer="lecun_normal")
        self.batch_norm = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

    def call(self, inputs):
        x = self.conv_layer(inputs)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        return x

class DecoderBlock(layers.Layer):
    def __init__(self, filters, name="decoder_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv_layer_1 = Conv2DReLU(filters=self.filters, kernel_size=3)
        self.conv_layer_2 = Conv2DReLU(filters=self.filters, kernel_size=3)
        self.upsampling_layer = layers.UpSampling2D(size=2, interpolation="bilinear")

    def call(self, inputs, skip=None):
        x = self.upsampling_layer(inputs)
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        return x

class DecoderCup(layers.Layer):
    def __init__(self, decoder_channels, num_skip_connections=3, name="decoder_cup", **kwargs):
        super().__init__(name=name, **kwargs)
        self.decoder_channels = decoder_channels
        self.num_skip_connections = num_skip_connections

    def build(self, input_shape):
        self.conv_more = Conv2DReLU(filters=512, kernel_size=3)
        self.decoder_blocks = [DecoderBlock(filters=out_ch) for out_ch in self.decoder_channels]

    def call(self, hidden_states, features):
        x = self.conv_more(hidden_states)
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[i] if (features is not None and i < self.num_skip_connections) else None
            x = decoder_block(x, skip=skip)
        return x

class PreActBottleneck(layers.Layer):
    
    def __init__(self, cin, cout=None, cmid=None, stride=1, name="preact", **kwargs):
        super().__init__(name=name, **kwargs)
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = tfa.layers.GroupNormalization(32, epsilon=1e-6)
        self.conv1 = conv1x1(cmid, bias=False)
        self.gn2 = tfa.layers.GroupNormalization(32, epsilon=1e-6)
        self.conv2 = conv3x3(cmid, stride, bias=False)
        self.gn3 = tfa.layers.GroupNormalization(32, epsilon=1e-6)
        self.conv3 = conv1x1(cout, bias=False)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cout, stride, bias=False)
            self.gn_proj = tfa.layers.GroupNormalization(cout, epsilon=1e-5)

    def call(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = tf.nn.relu(self.gn1(self.conv1(x)))
        y = tf.nn.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return tf.nn.relu(residual + y)

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = [weights[f"{n_block}/{n_unit}/conv1/kernel"]]
        conv2_weight = [weights[f"{n_block}/{n_unit}/conv2/kernel"]]
        conv3_weight = [weights[f"{n_block}/{n_unit}/conv3/kernel"]]

        gn1_weight = [
            np.squeeze(weights[f"{n_block}/{n_unit}/gn1/scale"], axis=(0, 1, 2)),
            np.squeeze(weights[f"{n_block}/{n_unit}/gn1/bias"], axis=(0, 1, 2))
        ]
        gn2_weight = [
            np.squeeze(weights[f"{n_block}/{n_unit}/gn2/scale"], axis=(0, 1, 2)),
            np.squeeze(weights[f"{n_block}/{n_unit}/gn2/bias"], axis=(0, 1, 2))
        ]
        gn3_weight = [
            np.squeeze(weights[f"{n_block}/{n_unit}/gn3/scale"], axis=(0, 1, 2)),
            np.squeeze(weights[f"{n_block}/{n_unit}/gn3/bias"], axis=(0, 1, 2))
        ]

        self.conv1.set_weights(conv1_weight)
        self.conv2.set_weights(conv2_weight)
        self.conv3.set_weights(conv3_weight)
        self.gn1.set_weights(gn1_weight)
        self.gn2.set_weights(gn2_weight)
        self.gn3.set_weights(gn3_weight)

        if hasattr(self, 'downsample'):
            proj_conv_weight = [weights[f"{n_block}/{n_unit}/conv_proj/kernel"]]
            proj_gn_weight = [
                np.squeeze(weights[f"{n_block}/{n_unit}/gn_proj/scale"], axis=(0, 1, 2)),
                np.squeeze(weights[f"{n_block}/{n_unit}/gn_proj/bias"], axis=(0, 1, 2))
            ]
            self.downsample.set_weights(proj_conv_weight)
            self.gn_proj.set_weights(proj_gn_weight)

class ResNetV2(models.Model):
    """Pre-activation (v2) ResNet model."""
    def __init__(self, block_units, width_factor=1, trainable=True, name="resnet_v2", **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        width = int(64 * width_factor)
        self.width = width

        self.root = Sequential([
            layers.Conv2D(width, kernel_size=7, strides=2, use_bias=False, padding="same", name="conv", kernel_regularizer=ws_reg),
            tfa.layers.GroupNormalization(32, epsilon=1e-6),
            layers.ReLU()
        ])

        self.body = [
            Sequential([PreActBottleneck(cin=width, cout=width*4, cmid=width, name="block1_unit1")]),
            Sequential([PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2, name="block2_unit1")]),
            Sequential([PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2, name="block3_unit1")])
        ]

    def call(self, x):
        features = []
        x = self.root(x)
        features.append(x)
        x = layers.MaxPool2D(pool_size=3, strides=2, padding="valid")(x)
        for layer in self.body:
            x = layer(x)
            features.append(x)
        return x, features[::-1]

def resnet_embeddings(x, image_size=224, n_skip=3, pretrain=True):
    resnet50v2 = ResNet50V2(
        weights='imagenet' if pretrain else None,
        include_top=False, 
        input_shape=(image_size, image_size, 3)
    )
    _ = resnet50v2(x)
    layers = ["conv3_block4_preact_relu", "conv2_block3_preact_relu", "conv1_conv"]
    features = [resnet50v2.get_layer(l).output for l in layers] if n_skip > 0 else []
    return resnet50v2, features

def TransUNet(
    image_size=224,
    patch_size=16,
    hybrid=True,
    grid=(14, 14),
    hidden_size=768,
    n_layers=12,
    n_heads=12,
    mlp_dim=3072,
    dropout=0.1,
    decoder_channels=[256, 128, 64, 16],
    n_skip=3,
    num_classes=3,
    final_act="sigmoid",
    pretrain=True,
    freeze_enc_cnn=True,
    name="TransUNet",
):
    """Builds the TransUNet model."""
    # Transformer Encoder
    assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    # Embedding
    if hybrid:
        grid_size = grid
        patch_size = image_size // 16 // grid_size[0]
        patch_size = max(patch_size, 1)  # Ensure patch size is at least 1

        resnet50v2, features = resnet_embeddings(
            x, image_size=image_size, n_skip=n_skip, pretrain=pretrain
        )

        if freeze_enc_cnn:
            resnet50v2.trainable = False

        y = resnet50v2.get_layer("conv4_block6_preact_relu").output
        x = resnet50v2.input
    else:
        y = x
        features = None

    y = layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
        trainable=True,
    )(y)

    y = layers.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    y = PositionalEmbedding(
        name="Transformer/posembed_input", trainable=True
    )(y)
    y = layers.Dropout(dropout)(y)

    # Transformer/Encoder
    for n in range(n_layers):
        y, _ = TransformerLayer(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
            trainable=True,
        )(y)

    y = layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(y)

    n_patch_sqrt = int(math.sqrt(y.shape[1]))
    y = layers.Reshape(target_shape=[n_patch_sqrt, n_patch_sqrt, hidden_size])(y)

    # Decoder CUP
    if decoder_channels:
        y = DecoderCup(
            decoder_channels=decoder_channels, n_skip=n_skip
        )(y, features)

    # Segmentation Head
    y = SegmentationHead(num_classes=num_classes, final_act=final_act)(y)

    model = models.Model(inputs=x, outputs=y, name=name)

    return model
