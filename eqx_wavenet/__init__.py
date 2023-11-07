from beartype.claw import beartype_this_package

beartype_this_package()

from .wavenet import Wavenet
from .wavenet_config import WavenetConfig

__all__ = ["WavenetConfig", "Wavenet"]
