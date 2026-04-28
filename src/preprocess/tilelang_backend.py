from __future__ import annotations

import torch

from .pytorch_backend import causal_rolling_zscore_eager

try:
    import tilelang as _tilelang  # noqa: F401
except Exception:
    _tilelang = None


class TileLangUnavailable(RuntimeError):
    """Raised when TileLang runtime is unavailable for kernel execution."""


def tilelang_status() -> tuple[bool, str]:
    if _tilelang is None:
        return False, "tilelang package not installed"
    if not torch.cuda.is_available():
        return False, "CUDA device unavailable"
    return True, "available"


def causal_rolling_zscore_tilelang(x: torch.Tensor, window: int, eps: float = 1e-5) -> torch.Tensor:
    available, reason = tilelang_status()
    if not available:
        raise TileLangUnavailable(reason)

    # Scaffold path: keep the API and benchmark wiring stable, while using the
    # numerically equivalent eager reference until a full TileLang kernel lands.
    return causal_rolling_zscore_eager(x, window=window, eps=eps)
