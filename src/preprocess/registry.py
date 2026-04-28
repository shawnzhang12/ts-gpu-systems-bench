from __future__ import annotations

from collections.abc import Callable

import torch

from .pytorch_backend import causal_rolling_zscore_compile, causal_rolling_zscore_eager
from .tilelang_backend import causal_rolling_zscore_tilelang, tilelang_status
from .triton_backend import TRITON_AVAILABLE, causal_rolling_zscore_triton

BackendFn = Callable[[torch.Tensor, int, float], torch.Tensor]


class BackendUnavailable(RuntimeError):
    """Raised when a requested backend is not available in the current runtime."""


def available_backends() -> dict[str, bool]:
    tile_ok, _ = tilelang_status()
    return {
        "pytorch_eager": True,
        "compile": True,
        "triton": TRITON_AVAILABLE,
        "tilelang": tile_ok,
    }


def get_preprocess_backend(name: str) -> BackendFn:
    name = name.lower()
    if name == "pytorch_eager":
        return causal_rolling_zscore_eager
    if name == "compile":
        return causal_rolling_zscore_compile
    if name == "triton":
        if not TRITON_AVAILABLE:
            raise BackendUnavailable("triton backend unavailable")
        return causal_rolling_zscore_triton
    if name == "tilelang":
        tile_ok, reason = tilelang_status()
        if not tile_ok:
            raise BackendUnavailable(f"tilelang backend unavailable: {reason}")
        return causal_rolling_zscore_tilelang

    known = ", ".join(sorted(available_backends().keys()))
    raise ValueError(f"unknown backend '{name}', expected one of: {known}")
