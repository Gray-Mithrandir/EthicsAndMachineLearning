"""Basing network"""
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict
from collections import namedtuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from numpy.typing import NDArray

from config import AugmentationSettings, PreProcessing

EvaluationResult = namedtuple("EvaluationResult", ["accuracy", "loss"])


class NetworkInterface(ABC):
    """Neural network interface"""

    def __init__(self, output_size: int) -> None:
        """Initialize network

        Parameters
        ----------
        output_size: int
            Output vector size
        """
        self.logger = logging.getLogger("raido.network")
        self.config = PreProcessing()
        self.model = self._create_model(output_size)


    @staticmethod
    @abstractmethod
    def name() -> str:
        """Network name"""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Batch size"""

    @property
    def epochs(self) -> int:
        """Number of train epochs"""
        return 150

    @abstractmethod
    def _create_model(self, output_size: int) -> tf.keras.Model:
        """Create a new network

        Parameters
        ----------
        output_size: int
            Output vector size

        Returns
        -------
        tf.keras.Model
            New model
        """

    def train(self, train_ds: tf.data.Dataset, validation_ds: tf.data.Dataset) -> tf.keras.callbacks.History:
        """Train model with given train and validation dataset

        Parameters
        ----------
        train_ds: tf.data.Dateset
            Train dataset
        validation_ds: tf.data.Dataset
            Validation dataset

        Returns
        -------
        tf.keras.callbacks.History
            Train history object
        """
        self.logger.info("Starting model training!")
        check_point_path = Path("data", "cooked", "checkpoint", f"{self.name().lower()}", "weights").absolute()
        shutil.rmtree(check_point_path.parent, ignore_errors=True)
        check_point_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            f"{check_point_path}",
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            save_freq='epoch',
        )
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            verbose=1,
            cooldown=5,
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", verbose=1, patience=7, start_from_epoch=10
        )
        history = self.model.fit(
            train_ds,
            epochs=self.epochs,
            verbose=1,
            validation_data=validation_ds,
            callbacks=[reduce_lr_callback, early_stop_callback, checkpoint_callback],
        )
        self.logger.info("Training done.Loading best weights")
        self.model.load_weights(f"{check_point_path}")
        return history

    def evaluate(self, eval_ds: tf.data.Dataset) -> EvaluationResult:
        """Evaluate network performance with given dataset

        Parameters
        ----------
        eval_ds: tf.data.Dataset
            Dataset to evaluate

        Returns
        -------
        EvaluationResult
            Network evaluation results
        """
        self.logger.info("Starting evaluation")
        loss, accuracy = self.model.evaluate(eval_ds, verbose=0)
        result = EvaluationResult(accuracy=accuracy, loss=loss)
        self.logger.info("Evaluation results: %s", result)
        return result

    def classification_report(self, eval_ds: tf.data.Dataset, labels: Tuple[str, ...]) -> Dict[str, Dict[str, float]]:
        """Build classification report

        Parameters
        ----------
        eval_ds: tf.data.Dataset
            Dataset to evaluate
        labels: Tuple[str, ...]
            Class labels

        Returns
        -------
        Dict[str, Dict[str, float]]
            Precision, recall, F1 score for each class
        """
        self.logger.info("Build classification report")
        y_true = []
        y_pred = []
        for _data, _labels in eval_ds:
            result = self.model.predict_on_batch(_data)
            y_true.append(np.argmax(np.squeeze(np.array(_labels))))
            y_pred.append(np.argmax(result))
        _report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels, output_dict=True)
        self.logger.info("Classification report %s", _report)
        return _report

    def confusion_matrix(self, eval_ds: tf.data.Dataset) -> NDArray:
        """Build multilabel confusion matrix

        Parameters
        ----------
        eval_ds: tf.data.Dataset
            Dataset to evaluate

        Returns
        -------
        NDArray
            A 2x2 confusion matrix corresponding to each output in the input.
        """
        self.logger.info("Build confusion matrix")
        y_true = []
        y_pred = []
        for _data, _labels in eval_ds:
            result = self.model.predict_on_batch(_data)
            y_true.append(np.argmax(np.squeeze(np.array(_labels))))
            y_pred.append(np.argmax(result))
        _report = confusion_matrix(y_true=y_true, y_pred=y_pred)
        self.logger.info("Confusion matrix %s", _report)
        return _report

    @staticmethod
    def _get_augment_layers() -> Tuple[tf.keras.layers.Layer, ...]:
        """Return augmentation layers"""
        settings = AugmentationSettings()
        image_sizes = PreProcessing().image_size
        return (
            tf.keras.layers.RandomTranslation(
                height_factor=settings.height_shift_range / 100.0,
                width_factor=settings.width_shift_range / 100.0,
                fill_mode="constant",
                fill_value=0,
                input_shape=[*image_sizes, 1],
            ),
            tf.keras.layers.RandomZoom(
                height_factor=settings.zoom_range / 100.0,
                fill_mode="constant",
                fill_value=0,
            ),
            tf.keras.layers.RandomRotation(
                factor=settings.rotation_angle / 100.0,
                fill_mode="constant",
                fill_value=0,
            ),
        )
