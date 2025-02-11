import tensorflow as tf
from tensorflow.keras import layers, models
import math

'''
UNETR
'''

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = tf.shape(patches)[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.position_embedding.input_dim, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

def transformer_mlp_block(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def conv_block(x, filters, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, dropout=0.0):
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer=kernel_initializer)(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    return x

def upsample_block(x, filters, name=None):
    return layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=name)(x)

def upsample_conv_block(x, filters, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, dropout=0.0):
    x = upsample_block(x, filters)
    return conv_block(x, filters, activation=activation, kernel_initializer=kernel_initializer, batch_norm=batch_norm, dropout=dropout)

def double_conv_block(x, filters, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, dropout=0.0):
    x = conv_block(x, filters, activation=activation, kernel_initializer=kernel_initializer, batch_norm=batch_norm, dropout=dropout)
    return conv_block(x, filters, activation=activation, kernel_initializer=kernel_initializer, batch_norm=batch_norm, dropout=0.0)

def unetr_2d(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    transformer_layers,
    num_heads,
    transformer_units,
    data_augmentation=None,
    num_filters=16,
    num_classes=1,
    decoder_activation='relu',
    decoder_kernel_init='he_normal',
    vit_hidden_mult=3,
    batch_norm=True,
    dropout=0.0,
):
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs) if data_augmentation else inputs
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    hidden_states_out = []

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = transformer_mlp_block(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])
        hidden_states_out.append(encoded_patches)

    total_upscale_factor = int(math.log2(patch_size))
    dropout = [dropout] * total_upscale_factor if isinstance(dropout, float) else dropout
    
    z = layers.Reshape([
        input_shape[0] // patch_size, input_shape[1] // patch_size, projection_dim
    ])(encoded_patches)
    x = upsample_block(z, num_filters * (2 ** (total_upscale_factor - 1)))

    for layer in reversed(range(1, total_upscale_factor)):
        z = layers.Reshape([
            input_shape[0] // patch_size, input_shape[1] // patch_size, projection_dim
        ])(hidden_states_out[(vit_hidden_mult * layer) - 1])
        for _ in range(total_upscale_factor - layer):
            z = upsample_conv_block(
                z, num_filters * (2 ** layer), activation=decoder_activation,
                kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[layer]
            )
        x = layers.concatenate([x, z])
        x = double_conv_block(
            x, num_filters * (2 ** layer), activation=decoder_activation,
            kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[layer]
        )
        x = upsample_block(x, num_filters * (2 ** (layer - 1)))

    first_skip = double_conv_block(
        augmented, num_filters, activation=decoder_activation,
        kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[0]
    )
    x = layers.concatenate([first_skip, x])
    x = double_conv_block(
        x, num_filters, activation=decoder_activation,
        kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[0]
    )
    output = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=output, name="UNETR")
    return model
