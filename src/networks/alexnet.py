"""AlexNet implementation"""
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.base import NetworkInterface, EarlyStopper


class AlexNet(nn.Module):
    def __init__(self, num_classes, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Network(NetworkInterface):
    """AlexNet implementation"""

    @property
    def loss_function(self) -> Any:
        return nn.CrossEntropyLoss()

    def optimizer(self, model: nn.Module) -> Any:
        return torch.optim.Adagrad(model.parameters(), lr=1e-3)

    def scheduler(self, optimizer: Any) -> Any:
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    @property
    def model(self) -> nn.Module:
        return AlexNet(num_classes=self.output_size)

    @property
    def early_stop(self) -> EarlyStopper:
        return EarlyStopper(patience=5, start_epoch=30)

    @property
    def name(self) -> str:
        return "AlexNet"
