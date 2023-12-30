"""LeNet-5 implementation"""

import tensorflow as tf

from networks.base import NetworkInterface


class Network(NetworkInterface):
    """LeNet-5 implementation"""
    @staticmethod
    def name() -> str:
        return "LeNet5"

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
        model = tf.keras.models.Sequential(name="LeNet-5")
        model.add(tf.keras.layers.Rescaling(1.0 / 255, input_shape=(self.config.image_size[0],
                                                                    self.config.image_size[0],
                                                                    1)))
        for layer in self._get_augment_layers():
            model.add(layer)
        model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=120, activation="relu"))
        model.add(tf.keras.layers.Dense(units=84, activation="relu"))
        model.add(tf.keras.layers.Dense(units=output_size, activation="softmax"))
        model.compile(
            loss=tf.keras.metrics.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        return model
