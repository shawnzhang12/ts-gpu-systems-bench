from __future__ import annotations

import torch

from src.models.factory import build_model


def test_mamba_forward_shape() -> None:
    model = build_model("mamba", in_features=7, d_model=32, n_layers=1, d_state=8, d_conv=2, expand=2)
    x = torch.randn(4, 64, 7)
    y = model(x)
    assert y.shape == (4,)


def test_transformer_forward_shape() -> None:
    model = build_model("transformer", in_features=7, d_model=32, n_heads=4, n_layers=1, dropout=0.0)
    x = torch.randn(4, 64, 7)
    y = model(x)
    assert y.shape == (4,)
