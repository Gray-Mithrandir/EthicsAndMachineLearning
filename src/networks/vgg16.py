"""VGG-16 implementation"""
from typing import Any, List, Union

import torch
from torch import nn
from torchvision.models.vgg import VGG, cfgs

from config import settings
from networks.base import NetworkInterface


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    """Modified version from `torchvision.models.vgg` with single input channel"""
    layers: List[nn.Module] = []
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Network(NetworkInterface):
    """VGG-16 interface"""

    def get_model(self) -> nn.Module:
        """Return VGG-16 model with batch normalization"""
        return VGG(make_layers(cfgs["D"], batch_norm=True), num_classes=self.output_size)

    def get_optimizer(self, model: nn.Module) -> Any:
        """Return SGD optimizer"""
        return torch.optim.SGD(
            model.parameters(),
            lr=settings[self.name()].optimizer.learning_rate,
            weight_decay=settings[self.name()].optimizer.weight_decay,
            momentum=settings[self.name()].optimizer.momentum,
        )

    @classmethod
    def name(cls) -> str:
        """Model name"""
        return "VGG16"
