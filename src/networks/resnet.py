"""LeNet-5 implementation"""
import tensorflow as tf

from networks.base import NetworkInterface
from config import PreProcessing


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(
            x
        )
        shortcut = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = tf.keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="SAME", name=name + "_2_conv")(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = tf.keras.layers.Activation("relu", name=name + "_2_relu")(x)

    x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_3_bn")(
        x
    )

    x = tf.keras.layers.Add(name=name + "_add")([shortcut, x])
    x = tf.keras.layers.Activation("relu", name=name + "_out")(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
    return x


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """A residual block.

    Args:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
          otherwise identity shortcut.
        name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

    preact = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_preact_bn"
    )(x)
    preact = tf.keras.layers.Activation("relu", name=name + "_preact_relu")(preact)

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(
            preact
        )
    else:
        shortcut = tf.keras.layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = tf.keras.layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name + "_1_conv")(
        preact
    )
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = tf.keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = tf.keras.layers.Activation("relu", name=name + "_2_relu")(x)

    x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = tf.keras.layers.Add(name=name + "_out")([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    Returns:
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + "_block1")
    for i in range(2, blocks):
        x = block2(x, filters, name=name + "_block" + str(i))
    x = block2(x, filters, stride=stride1, name=name + "_block" + str(blocks))
    return x


def block3(
    x,
    filters,
    kernel_size=3,
    stride=1,
    groups=32,
    conv_shortcut=True,
    name=None,
):
    """A residual block.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      groups: default 32, group size for grouped convolution.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(
            (64 // groups) * filters,
            1,
            strides=stride,
            use_bias=False,
            name=name + "_0_conv",
        )(x)
        shortcut = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 1, use_bias=False, name=name + "_1_conv")(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = tf.keras.layers.Activation("relu", name=name + "_1_relu")(x)

    c = filters // groups
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        depth_multiplier=c,
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x_shape = tf.keras.backend.shape(x)[:-1]
    x = tf.keras.backend.reshape(
        x, tf.keras.backend.concatenate([x_shape, (groups, c, c)])
    )
    x = tf.keras.layers.Lambda(
        lambda x: sum(x[:, :, :, :, i] for i in range(c)),
        name=name + "_2_reduce",
    )(x)
    x = tf.keras.backend.reshape(x, tf.keras.backend.concatenate([x_shape, (filters,)]))
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = tf.keras.layers.Activation("relu", name=name + "_2_relu")(x)

    x = tf.keras.layers.Conv2D(
        (64 // groups) * filters, 1, use_bias=False, name=name + "_3_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_3_bn")(
        x
    )

    x = tf.keras.layers.Add(name=name + "_add")([shortcut, x])
    x = tf.keras.layers.Activation("relu", name=name + "_out")(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      groups: default 32, group size for grouped convolution.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = block3(x, filters, stride=stride1, groups=groups, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block3(
            x,
            filters,
            groups=groups,
            conv_shortcut=False,
            name=name + "_block" + str(i),
        )
    return x


class Network(NetworkInterface):
    """LeNet-5 implementation"""

    @staticmethod
    def name() -> str:
        return "ResNet50"

    @property
    def batch_size(self) -> int:
        return 32

    def _create_model(self, output_size: int) -> tf.keras.Model:
        """Create LeNet-5 model

        Parameters
        ----------
        output_size: int
            Output vector size

        Returns
        ------
        Sequential
            New model
        """
        pre_processing = PreProcessing()
        img_input = tf.keras.layers.Input(shape=pre_processing.input_shape)
        img_input = tf.keras.layers.Rescaling(1.0 / 255)(img_input)
        for layer in self._get_augment_layers():
            img_input = layer(img_input)
        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 6, name="conv4")
        x = stack1(x, 512, 3, name="conv5")

        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=True, name="conv1_conv")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = tf.keras.layers.Dense(output_size, activation="softmax", name="predictions")(x)
        model = tf.keras.Model(img_input, x, name="resnet50")
        model.compile(
            loss=tf.keras.metrics.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        return model
