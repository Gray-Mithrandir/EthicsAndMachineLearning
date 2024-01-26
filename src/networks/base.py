"""Basing network"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import timedelta
from time import monotonic
from typing import Any, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import v2
from tqdm import tqdm

from config import settings
from dataset import get_sample_images, get_test_dataset, get_train_dataset, get_validation_dataset
from history import EvaluationReport, EvaluationSample, Report, TrainHistory


class EarlyStopper:  # pylint: disable=too-few-public-methods
    """Early stopper implementation"""

    def __init__(self, patience: int = 3, min_delta: float = 0, start_epoch: int = 20):
        """Initialize early stopper

        Parameters
        ----------
        patience: int
            Number of epoch without improvement
        min_delta: float
            Minimal improvement
        start_epoch: int
            Start epoch, allow warmup period

        """
        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch
        self.counter = 0
        self.current_epoch = 0
        self.min_validation_loss = float("inf")
        self.logger = logging.getLogger("raido")

    def is_early_stop(self, validation_loss: float) -> bool:
        """Check if condition for early stop

        Parameters
        ----------
        validation_loss: float
            Validation loss of current epoch

        Returns
        -------
        bool
            Set if condition for early stop
        """
        self.current_epoch += 1
        if self.current_epoch < self.start_epoch:
            return False
        if validation_loss < self.min_validation_loss:
            self.logger.info("Validation loss reduced %.5f < %.5f", validation_loss, self.min_validation_loss)
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.logger.info("Early stop. Minimal validation loss %.5f", self.min_validation_loss)
                return True
            self.logger.info("Validation loss not reduced. Count %s/%s", self.counter, self.patience)
        return False


class ModelCheckpoint:
    """Save best model during training"""

    def __init__(self):
        """Initialize checkpoint"""
        self.logger = logging.getLogger("raido")
        self._best_accuracy = float("-inf")
        self._state_dict = {}

    def checkpoint(self, model: nn.Module, accuracy: float) -> bool:
        """Save model state if current accuracy is better that saved

        Parameters
        ---------
        model: nn.Model
            Model to save
        accuracy: float
            Validation accuracy

        Returns
        -------
        bool
            Set if current model is better that saved
        """
        if accuracy > self._best_accuracy:
            self.logger.info("Model accuracy improved %.5f > %.5f", accuracy, self._best_accuracy)
            self._best_accuracy = accuracy
            self._state_dict = model.state_dict()
            return True
        return False

    @property
    def best_accuracy(self) -> float:
        """Return best saved accuracy"""
        return self._best_accuracy

    @property
    def state(self) -> Dict[Any, Any]:
        """Return saved model state dictionary"""
        return self._state_dict


class NetworkInterface(ABC):
    """Neural network interface"""

    def __init__(self) -> None:
        """Initialize network"""
        self.logger = logging.getLogger("raido")
        self.output_size = len(settings.preprocessing.target_diagnosis)
        self.model_checkpoint = ModelCheckpoint()

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Network name"""

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return a neural network model"""

    def get_loss_function(self) -> Any:
        """Return Loss function for network"""
        return nn.CrossEntropyLoss()

    def get_optimizer(self, model: nn.Module) -> Any:
        """Return Optimizer function"""
        return torch.optim.Adagrad(model.parameters(), lr=settings[self.name()].optimizer.learning_rate)

    def get_scheduler(self, optimizer: Any) -> Any:
        """Return learning rate scheduler"""
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=settings[self.name()].scheduler.factor,
            patience=settings[self.name()].scheduler.patience,
        )

    def get_early_stop(self) -> EarlyStopper:
        """Early stopper"""
        return EarlyStopper(
            patience=settings[self.name()].earlystop.patience, start_epoch=settings[self.name()].earlystop.start_epoch
        )

    @property
    def epochs(self) -> int:
        """Number of train epochs"""
        return settings[self.name()].epochs

    @property
    def batch_size(self) -> int:
        """Batch size"""
        return settings[self.name()].batch_size

    def train(self, device: Any) -> TrainHistory:  # pylint: disable=too-many-locals
        """Train model with given train and validation dataset

        Parameters
        ----------
        device: Any
            Torch device

        Returns
        -------
        TrainHistory
            Train history
        """
        self.logger.info("Starting model training!")

        model = self.get_model().to(device)
        self.logger.info("Compiling model")
        model.compile()

        loss_function = self.get_loss_function()
        optimizer = self.get_optimizer(model=model)
        scheduler = self.get_scheduler(optimizer)
        early_stop = self.get_early_stop()

        self.logger.info("Loading train dataset")
        train_ds = get_train_dataset(batch_size=self.batch_size)
        val_ds = get_validation_dataset(batch_size=self.batch_size)

        start_time = monotonic()
        history = TrainHistory()
        for epoch in range(self.epochs):
            history.next_epoch()
            self.logger.info("Starting training. Epoch %s/%s", epoch + 1, self.epochs)
            model.train()
            for images, labels in tqdm(train_ds, desc=f"Epoch {epoch + 1}/ {self.epochs}", miniters=0, unit="batch"):
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(images)
                loss = loss_function(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                history.train_update(outputs, labels, loss)
            self.logger.info("Starting validation")
            model.eval()
            with torch.no_grad():
                for images, labels in tqdm(val_ds, desc="Validation", miniters=0, unit="batch"):
                    images = images.to(device)
                    labels = labels.to(device)
                    # Forward pass
                    outputs = model(images)
                    history.validation_update(outputs, labels, loss_function(outputs, labels))
            scheduler.step(history.validation_loss)
            history.record_learning_rate(scheduler.get_last_lr()[0])
            self.logger.info(history.epoch_summary)
            if self.model_checkpoint.checkpoint(model=model, accuracy=history.validation_accuracy):
                history.keep_epoch()
            if early_stop.is_early_stop(history.validation_loss):
                break
        self.logger.info("Training complete. Run time: %s", timedelta(seconds=monotonic() - start_time))
        return history

    def evaluate(self, device: Any) -> EvaluationReport:
        """Return evaluation status

        Parameters
        ----------
        device: Any
            Torch device

        Returns
        -------
        EvaluationReport
            Evaluation report
        """
        model = self.get_model().to(device)
        model.load_state_dict(self.model_checkpoint.state)
        model.eval()
        evaluation_report = EvaluationReport()
        start_time = monotonic()
        for name, only_male in (("common", None), ("male", True), ("female", False)):
            test_ds = get_test_dataset(batch_size=1, only_male=only_male)
            with torch.no_grad():
                report = Report()
                for images, labels in tqdm(test_ds, desc="Evaluating", miniters=0, unit="batch"):
                    images = images.to(device)
                    labels = labels.to(device)
                    # Forward pass
                    outputs = model(images)
                    report.update(y_predicted=outputs, y_true=labels)
                setattr(evaluation_report, name, report)
        self.logger.info("Evaluation complete. Run time: %s", timedelta(seconds=monotonic() - start_time))
        self.logger.info(evaluation_report.summary)
        return evaluation_report

    def evaluation_sample(self, device: Any) -> Dict[str, Dict[str, EvaluationSample]]:
        """Evaluate image

        Parameters
        ----------
        device: Any
            Computation device

        Returns
        -------
        Dict[str, Dict[str, EvaluationSample]]
            Evaluation samples as {Male|Female: {Diagnosis: EvaluationSample}}
        """
        model = self.get_model().to(device)
        model.load_state_dict(self.model_checkpoint.state)
        model.eval()
        data_transform = v2.Compose(
            [v2.Grayscale(num_output_channels=1), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        predictions = {"Male": {}, "Female": {}}
        with torch.no_grad():
            model.eval()
            for image in tqdm(get_sample_images(), desc="Evaluating", miniters=0, unit="image"):
                img_normalized = data_transform(Image.open(f"{image}")).float()
                img_normalized = img_normalized.unsqueeze_(0)
                img_normalized = img_normalized.to(device)
                output = model(img_normalized)
                sample = EvaluationSample(
                    image=image, probabilities=tuple(pred for pred in F.softmax(output, dim=1).cpu().numpy()[0])
                )
                predictions[sample.patient_sex][sample.true_label] = sample
        return predictions
