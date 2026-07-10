from __future__ import annotations

from . import core, inference
from .core import (
    SPSBlock,
    SPSConfig,
    SPSFlashAttention,
    SPSModelBase,
    IGNORE_INDEX,
    ModelConfig,
)
from .inference import SPSInferenceMixin

class SPSModel(SPSInferenceMixin, SPSModelBase):
    pass


__all__ = [
    "core",
    "inference",
    "SPSBlock",
    "SPSConfig",
    "SPSFlashAttention",
    "SPSInferenceMixin",
    "SPSModel",
    "SPSModelBase",
    "IGNORE_INDEX",
    "ModelConfig",
]
