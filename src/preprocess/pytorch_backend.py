from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


def causal_rolling_zscore_eager(x: torch.Tensor, window: int, eps: float = 1e-5) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"expected [B, L, C], got shape={tuple(x.shape)}")
    if window < 1:
        raise ValueError("window must be >= 1")

    x_f = x.float()
    x_t = x_f.transpose(1, 2)  # [B, C, L]

    padded = F.pad(x_t, (window - 1, 0), mode="constant", value=0.0)
    windows = padded.unfold(-1, window, 1)  # [B, C, L, window]

    valid = torch.ones_like(x_t)
    valid = F.pad(valid, (window - 1, 0), mode="constant", value=0.0)
    valid_windows = valid.unfold(-1, window, 1)

    count = valid_windows.sum(dim=-1).clamp_min(1.0)
    mean = (windows * valid_windows).sum(dim=-1) / count

    centered = windows - mean.unsqueeze(-1)
    var = (centered.square() * valid_windows).sum(dim=-1) / count
    std = torch.sqrt(var + eps)

    out = (x_t - mean) / std
    return out.transpose(1, 2).to(dtype=x.dtype)


@lru_cache(maxsize=64)
def _compiled_kernel(window: int, eps: float):
    def _fn(x: torch.Tensor) -> torch.Tensor:
        return causal_rolling_zscore_eager(x, window=window, eps=eps)

    try:
        return torch.compile(_fn, dynamic=True)
    except Exception:
        return _fn


def causal_rolling_zscore_compile(x: torch.Tensor, window: int, eps: float = 1e-5) -> torch.Tensor:
    fn = _compiled_kernel(window=window, eps=eps)
    return fn(x)
