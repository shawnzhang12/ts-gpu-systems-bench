from __future__ import annotations

import torch.nn as nn

from .mamba_model import MambaForecaster
from .transformer_model import FlashTransformerForecaster


def build_model(model_type: str, in_features: int, **kwargs) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "mamba":
        return MambaForecaster(in_features=in_features, **kwargs)
    if model_type == "transformer":
        return FlashTransformerForecaster(in_features=in_features, **kwargs)
    raise ValueError("model.type must be one of: mamba, transformer")
