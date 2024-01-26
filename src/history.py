"""Train history and evaluation metrics"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from config import settings


class Report:
    """Classification report"""

    def __init__(self):
        self._y_pred = []
        self._y_true = []
        self._loss = []
        self._report = None  # type: Union[None, Dict[str, Dict[str, float]]]

    def update(self, y_predicted: torch.Tensor, y_true: torch.Tensor, loss: Optional[torch.Tensor] = None):
        """Update classification report

        Parameters
        ----------
        y_predicted: torch.Tensor
            Network output
        y_true: torch.Tensor
            True labels
        loss: Optional[torch.Tensor]
            Loss value
        """
        self._report = None
        for y_p, y_l in zip(F.softmax(y_predicted, dim=1), y_true):
            self._y_pred.append(tuple(y_pp.item() for y_pp in y_p))
            self._y_true.append(y_l.item())
        if loss is not None:
            self._loss.append(loss.item())

    def precision(self, class_name: Optional[str] = None) -> float:
        """Precision score

        Parameters
        ----------
        class_name: Optional[str]
            Class name. If None overall precision if return

        Returns
        -------
        float
            Precision
        """
        _name = class_name
        if _name is None:
            _name = "macro avg"
        return self.report[_name]["precision"]

    def recall(self, class_name: Optional[str] = None) -> float:
        """Recall score

        Parameters
        ----------
        class_name: Optional[str]
            Class name. If None overall recall if return

        Returns
        -------
        float
            Recall
        """
        _name = class_name
        if _name is None:
            _name = "macro avg"
        return self.report[_name]["recall"]

    def f1_score(self, class_name: Optional[str] = None) -> float:
        """F1-score

        Parameters
        ----------
        class_name: Optional[str]
            Class name. If None overall f1-score if return

        Returns
        -------
        float
            F1-score
        """
        _name = class_name
        if _name is None:
            _name = "macro avg"
        return self.report[_name]["f1-score"]

    def probabilities(self, class_name: str) -> Tuple[float, ...]:
        """Return class probabilities

        Parameters
        ----------
        class_name: str
            Class name to return

        Returns
        -------
        Tuple[float, ...]
            List of predicted probabilities
        """
        search_index = sorted(settings.preprocessing.target_diagnosis).index(class_name)
        _y_pred_index = [np.argmax(_y_pred) for _y_pred in self._y_pred]
        return tuple(_pred[_label] for _pred, _label in zip(self._y_pred, _y_pred_index) if _label == search_index)

    @property
    def accuracy(self) -> float:
        """Accuracy score"""
        return self.report["accuracy"]

    @property
    def loss(self) -> float:
        """Average loss"""
        if len(self._loss) == 0:
            return 0.0
        return sum(self._loss) / len(self._loss)

    @property
    def confusion_matrix(self) -> Any:
        """Evaluation confusion_matrix"""
        _y_pred_index = [np.argmax(_y_pred) for _y_pred in self._y_pred]
        return confusion_matrix(y_true=self._y_true, y_pred=_y_pred_index, normalize="true")

    @property
    def report(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Classification report as dictionary"""
        if self._report is None:
            y_pred = [np.argmax(_pred) for _pred in self._y_pred]
            self._report = classification_report(
                y_true=self._y_true,
                y_pred=y_pred,
                target_names=sorted(settings.preprocessing.target_diagnosis),
                output_dict=True,
            )
        return self._report


class TrainHistory:
    """Train history"""

    def __init__(self):
        self._epoch = 0
        self._train = []  # type: List[Report]
        self._validation = []  # type: List[Report]
        self._learning_rate = []  # type: List[float]
        self._keep_epoch = 0
        # Last train epoch to keep

    def train_update(self, predicted: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> None:
        """Update train statistics

        Parameters
        ----------
        predicted: torch.Tensor
            Predicted values. Raw output from network
        labels: torch.Tensor
            True labels
        loss: torch.Tensor
            Loss value
        """
        self._train[-1].update(y_predicted=predicted, y_true=labels, loss=loss)

    def validation_update(self, predicted: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> None:
        """Update validation statistics

        Parameters
        ----------
        predicted: torch.Tensor
            Predicted values. Raw output from network
        labels: torch.Tensor
            True labels
        loss: torch.Tensor
            Loss value
        """
        self._validation[-1].update(y_predicted=predicted, y_true=labels, loss=loss)

    def record_learning_rate(self, rate: float) -> None:
        """Save current learning rate"""
        self._learning_rate.append(rate)

    def keep_epoch(self) -> None:
        """Keep current epoch for plots"""
        self._keep_epoch = self._epoch

    def next_epoch(self) -> None:
        """Start new epoch"""
        self._train.append(Report())
        self._validation.append(Report())
        self._epoch += 1

    @property
    def epoch(self) -> int:
        """Return current epoch"""
        return self._epoch

    @property
    def train_accuracy(self) -> float:
        """Return last train accuracy"""
        return self._train[-1].accuracy

    @property
    def train_loss(self) -> float:
        """Return last train loss"""
        return self._train[-1].loss

    @property
    def validation_accuracy(self) -> float:
        """Return last validation accuracy"""
        return self._validation[-1].accuracy

    @property
    def validation_loss(self) -> float:
        """Return last validation loss"""
        return self._validation[-1].loss

    @property
    def accuracy_history(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Return train and validation accuracy history

        Returns
        -------
        Tuple[Tuple[float, ...], Tuple[float, ...]]
            List of train accuracy and validation accuracy over all epochs
        """
        train_acc = tuple(train.accuracy for train in self._train[: self._keep_epoch])
        val_add = tuple(val.accuracy for val in self._validation[: self._keep_epoch])
        return train_acc, val_add

    @property
    def loss_history(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Return train and validation loss history

        Returns
        -------
        Tuple[Tuple[float, ...], Tuple[float, ...]]
            List of train loss and validation loss over all epochs
        """
        train_loss = tuple(train.loss for train in self._train[: self._keep_epoch])
        val_loss = tuple(val.loss for val in self._validation[: self._keep_epoch])
        return train_loss, val_loss

    @property
    def learning_rate_history(self) -> Tuple[float, ...]:
        """Return learning rate history

        Returns
        -------
        Tuple[float]
            Learning rate history
        """
        return tuple(rate for rate in self._learning_rate[: self._keep_epoch])

    @property
    def epoch_summary(self) -> str:
        """Return train/validation epoch summary"""
        return (
            f"Accuracy (train/validation/delta): "
            f"{self.train_accuracy:.5f}/"
            f"{self.validation_accuracy:.5f}/"
            f"{self.train_accuracy - self.validation_accuracy:.5f}. "
            f"Loss (train/validation/delta): "
            f"{self.train_loss:.5f}/"
            f"{self.validation_loss:.5f}/"
            f"{self.train_loss - self.validation_loss:.5f}. "
            f"Learning rate: {self._learning_rate[-1]:.8f}"
        )

    def __iter__(self):
        yield from list(
            (train, validation)
            for train, validation in zip(self._train[: self._keep_epoch], self._validation[: self._keep_epoch])
        )


@dataclass
class EvaluationReport:
    """Evaluation report"""

    common: Report = None
    """Evaluation on balanced dataset"""
    male: Report = None
    """Evaluation on male only dataset"""
    female: Report = None
    """Evaluation on female only dataset"""

    @property
    def summary(self) -> str:
        """Return evaluation summary"""
        return (
            f"Accuracy (common/male/female):"
            f" {self.common.accuracy:.5f}/{self.male.accuracy:.5f}/{self.female.accuracy:.5f}"
        )


@dataclass
class EvaluationSample:
    """Single evaluation sample"""

    image: Path
    """Image path"""
    probabilities: Tuple[float, ...]
    """Prediction probabilities"""

    @property
    def predicted_label(self) -> str:
        """Predicted class name"""
        return sorted(settings.preprocessing.target_diagnosis)[np.argmax(self.probabilities)]

    @property
    def true_label(self) -> str:
        """True class name"""
        return self.image.parent.name

    @property
    def is_correct(self) -> bool:
        """Set if predicted label same as true"""
        return self.true_label == self.predicted_label

    @property
    def patient_sex(self) -> str:
        """Return patient sex"""
        return "Female" if self.image.name.startswith("F") else "Male"
