"""Visual transformers network implementation"""
import shutil
from pathlib import Path
from typing import Dict, Tuple, Sequence

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Dropout,
    Layer,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
    Input,
    Add,
    Normalization,
    Conv2D,
    Resizing,
)
from tensorflow.keras.models import Model, Sequential

from config import PreProcessing
from networks.base import NetworkInterface


class generate_patch(Layer):
    def __init__(self, patch_size):
        super(generate_patch, self).__init__()
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
        patch_dims = patches.shape[-1]
        patches = tf.reshape(
            patches, [batch_size, -1, patch_dims]
        )  # here shape is (batch_size, num_patches, patch_h*patch_w*c)
        return patches


class PatchEncode_Embed(Layer):
    """
    2 steps happen here
    1. flatten the patches
    2. Map to dim D; patch embeddings
    """

    def __init__(self, num_patches, projection_dim):
        super(PatchEncode_Embed, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)  # activation = linear
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class generate_patch_conv(Layer):
    """
    this is an example to generate conv patches comparable with the image patches
    generated using tf extract image patches. This wasn't the original implementation, specially
    the number of filters in the conv layer has nothing to do with patch size. It must be same as
    hidden dim (query/key dim) in relation to multi-head attention layer.
    """

    def __init__(self, patch_size, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size

    def call(self, images):
        patches = Conv2D(
            self.hidden_size,
            self.patch_size,
            self.patch_size,
            padding="valid",
            name="Embedding",
        )(images)
        # kernels and strides = patch size
        # the weights of the convolutional layer will be learned.
        rows_axis, cols_axis = (1, 2)  # channels last images
        seq_len = (images.shape[rows_axis] // self.patch_size) * (
            images.shape[cols_axis] // self.patch_size
        )
        x = tf.reshape(patches, [-1, seq_len, self.hidden_size])
        return x


class AddPositionEmbs(Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.posemb_init = tf.keras.initializers.RandomNormal(stddev=0.02)
        # posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input') # used in original code

    def build(self, inputs_shape):
        pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
        self.pos_embedding = self.add_weight(
            "pos_embedding", pos_emb_shape, initializer=self.posemb_init
        )

    def call(self, inputs, inputs_positions=None):
        # inputs.shape is (batch_size, seq_len, emb_dim).
        pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

        return inputs + pos_embedding


def mlp_block_f(mlp_dim, inputs):
    x = Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
    x = Dropout(rate=0.1)(x)  # dropout rate is from original paper,
    x = Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x)
    x = Dropout(rate=0.1)(x)
    return x


def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
    x = LayerNormalization(dtype=inputs.dtype)(inputs)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(
        x, x
    )  # self attention multi-head, dropout_rate is from original implementation
    x = Add()([x, inputs])  # 1st residual part

    y = LayerNormalization(dtype=x.dtype)(x)
    y = mlp_block_f(mlp_dim, y)
    y_1 = Add()([y, x])  # 2nd residual part
    return y_1


def Encoder_f(num_layers, mlp_dim, num_heads, inputs):
    x = AddPositionEmbs(name="posembed_input")(inputs)
    x = Dropout(rate=0.2)(x)
    for _ in range(num_layers):
        x = Encoder1Dblock_f(num_heads, mlp_dim, x)

    encoded = LayerNormalization(name="encoder_norm")(x)
    return encoded


def generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs):
    patches = Conv2D(
        filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding="valid"
    )(inputs)
    row_axis, col_axis = (1, 2)  # channels last images
    seq_len = (inputs.shape[row_axis] // patch_size) * (
        inputs.shape[col_axis] // patch_size
    )
    x = tf.reshape(patches, [-1, seq_len, hidden_size])
    return x


class Network(NetworkInterface):
    """Visual transformers implementation"""

    @staticmethod
    def name() -> str:
        return "ViT"

    @property
    def batch_size(self) -> int:
        return 32

    def _create_model(self, output_size: int) -> tf.keras.Model:
        """Create visual transformers model

        Parameters
        ----------
        output_size: int
            Output vector size

        Returns
        ------
        Sequential
            New model
        """
        # Settings
        pre_processing = PreProcessing()
        patch_size = 12
        projection_dim = 64
        transformer_layers = 8
        mlp_dim = 128
        num_heads = 4
        # Network
        inputs = Input(shape=pre_processing.input_shape)
        resize = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
        # for layer in self._get_augment_layers():
        #     inputs = layer(inputs)
        patches = generate_patch_conv_orgPaper_f(patch_size, projection_dim, resize)

        ######################################
        # ready for the transformer blocks
        ######################################
        encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, patches)

        #####################################
        #  final part (mlp to classification)
        #####################################
        # encoder_out_rank = int(tf.experimental.numpy.ndim(encoder_out))
        im_representation = tf.reduce_mean(encoder_out, axis=1)  # (1,) or (1,2)

        logits = Dense(
            units=output_size, name="head", kernel_initializer=tf.keras.initializers.zeros, activation="softmax"
        )(
            im_representation
        )  # !!! important !!! activation is linear

        final_model = tf.keras.Model(inputs=inputs, outputs=logits)
        final_model.compile(
            loss=tf.keras.metrics.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        return final_model
