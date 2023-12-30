"""VGG-16 implementation"""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential

from networks.base import NetworkInterface


class Network(NetworkInterface):
    """VGG-16 implementation"""

    @staticmethod
    def name() -> str:
        """Network name"""
        return "VGG16"

    @property
    def batch_size(self) -> int:
        """Batch size"""
        return 8

    def _create_model(self, output_size: int) -> tf.keras.Model:
        """Return a VGG-16 model

        Parameters
        ----------
        output_size: int
            Output vector size

        Returns
        ------
        Sequential
            New model
        """
        model = Sequential(name="VGG-16")
        model.add(tf.keras.layers.Rescaling(1.0 / 255.0))
        for layer in self._get_augment_layers():
            model.add(layer)
        model.add(
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=(3, 3), padding="same", activation="relu"
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same"))
        # Block 2
        model.add(
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same"))
        # Block 3
        model.add(
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same"))
        # Block 4
        model.add(
            tf.keras.layers.Conv2D(
                filters=512,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(
            tf.keras.layers.Conv2D(
                filters=512,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same"))
        # Block 5
        model.add(
            tf.keras.layers.Conv2D(
                filters=512,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(
            tf.keras.layers.Conv2D(
                filters=512,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same"))
        # FC
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
        model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
        model.add(tf.keras.layers.Dense(units=output_size, activation="softmax"))
        model.compile(
            loss=tf.keras.metrics.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Nadam(),
            metrics=["accuracy"],
        )
        return model
