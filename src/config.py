"""Settings file"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PreProcessing:
    """Pre-processing settings"""

    image_size = (128, 128)
    """Image size"""
    test_images = 20
    """Number of test images used for performance testing"""
    validation_images_fraction = 0.2
    """Number of test images used for validation testing as fraction per class"""

    @property
    def input_shape(self):
        """Return image input shape (image size, one channel)"""
        return [self.image_size[0], self.image_size[1], 1]


@dataclass(frozen=True)
class PlotSettings:
    """Plot Settings class"""

    plot_dpi = 300
    """Plot DPI"""
    plot_size = (10, 10)
    """Plot size in inches"""


@dataclass(frozen=True)
class AugmentationSettings:
    """Augmentation and pre-processing settings"""

    rotation_angle = 2.5
    """Rotate image in range [degrees]"""
    width_shift_range = 5.5
    """Shift image horizontally in range [%]"""
    height_shift_range = 7.5
    """Vertical shift range - [%]"""
    zoom_range = 5
    """Zoom range - [%]"""
