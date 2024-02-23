"""LeNet-5 implementation"""
from torch import nn

from networks.base import NetworkInterface


class LeNet5(nn.Module):
    """LeNet-5 network

    References
    ----------
    Lecun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. (1998).
     "Gradient-based learning applied to document recognition" (PDF).
      Proceedings of the IEEE. 86 (11): 2278–2324. doi:10.1109/5.726791. S2CID 14542261.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16384, 120), nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        return self.classifier(self.layer2(self.layer1(x)))


class Network(NetworkInterface):
    """LeNet-5 interface"""

    def get_model(self) -> nn.Module:
        """Return new model object"""
        return LeNet5(self.output_size)

    @classmethod
    def name(cls) -> str:
        """Model name"""
        return "LENET5"
