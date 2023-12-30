"""AlexNet implementation"""

import tensorflow as tf

from networks.base import NetworkInterface


class Network(NetworkInterface):
    """AlexNet implementation"""
    @staticmethod
    def name() -> str:
        return "AlexNet"

    @property
    def batch_size(self) -> int:
        return 32

    def _create_model(self, output_size: int) -> tf.keras.Model:
        """Create AlexNet model

        Parameters
        ----------
        output_size: int
            Output vector size

        Returns
        ------
        Sequential
            New model
        """
        model = tf.keras.models.Sequential(name="AlexNet")
        model.add(tf.keras.layers.Rescaling(1.0 / 255, input_shape=(self.config.image_size[0],
                                                                    self.config.image_size[0],
                                                                    1)))
        for layer in self._get_augment_layers():
            model.add(layer)
        model.add(tf.keras.layers.Conv2D(96, 11, strides=4, padding='same', activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(3, strides=2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(256, 5, strides=4, padding='same', activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(3, strides=2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(384, 3, strides=4, padding='same', activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(384, 3, strides=4, padding='same', activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(256, 3, strides=4, padding='same', activation="relu"))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
        model.compile(
            loss=tf.keras.metrics.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        return model
