from __future__ import annotations

import pytest
import torch

from src.preprocess.pytorch_backend import causal_rolling_zscore_compile, causal_rolling_zscore_eager
from src.preprocess.triton_backend import TRITON_AVAILABLE, causal_rolling_zscore_triton


def test_eager_output_shape() -> None:
    x = torch.randn(4, 32, 7)
    y = causal_rolling_zscore_eager(x, window=8)
    assert y.shape == x.shape


def test_compile_matches_eager_cpu() -> None:
    x = torch.randn(2, 24, 7)
    ref = causal_rolling_zscore_eager(x, window=6)
    out = causal_rolling_zscore_compile(x, window=6)
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not (TRITON_AVAILABLE and torch.cuda.is_available()), reason="triton+cuda not available")
def test_triton_matches_eager_cuda() -> None:
    x = torch.randn(2, 32, 7, device="cuda")
    ref = causal_rolling_zscore_eager(x, window=8)
    out = causal_rolling_zscore_triton(x, window=8)
    assert torch.allclose(ref, out, atol=2e-3, rtol=2e-3)
