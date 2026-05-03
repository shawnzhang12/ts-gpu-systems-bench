from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import torch

if "TILELANG_CACHE_DIR" not in os.environ:
    os.environ["TILELANG_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "tilelang_cache")

try:
    import tilelang as _tilelang
    import tilelang.language as T
except Exception:
    _tilelang = None
    T = None


class TileLangUnavailable(RuntimeError):
    """Raised when TileLang runtime is unavailable for kernel execution."""


def tilelang_status() -> tuple[bool, str]:
    if _tilelang is None:
        return False, "tilelang package not installed"
    if not torch.cuda.is_available():
        return False, "CUDA device unavailable"
    return True, "available"


def _tilelang_target() -> str:
    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    return f"cuda -arch=sm_{props.major}{props.minor}"


@lru_cache(maxsize=128)
def _build_tilelang_kernel(
    batch_size: int,
    features: int,
    seq_len: int,
    window: int,
    block_t: int,
    eps: float,
    target: str,
):
    if _tilelang is None or T is None:  # pragma: no cover
        raise TileLangUnavailable("tilelang package not installed")

    span = block_t + window - 1

    @_tilelang.jit(out_idx=[-1], target=target)
    def kernel_builder():
        @T.prim_func
        def kernel(
            x: T.Tensor((batch_size, features, seq_len), T.float32),
            out: T.Tensor((batch_size, features, seq_len), T.float32),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_t), batch_size * features, threads=block_t) as (bx, by):
                b = by // features
                c = by % features

                # Stage the block and its left halo in shared memory once.
                tile = T.alloc_shared((span,), "float32")
                base = bx * block_t - (window - 1)

                for off in T.serial(0, span, block_t):
                    for ti in T.Parallel(block_t):
                        idx = off + ti
                        if idx < span:
                            g = base + idx
                            if g >= 0 and g < seq_len:
                                tile[idx] = x[b, c, g]
                            else:
                                tile[idx] = 0.0

                for ti in T.Parallel(block_t):
                    t = bx * block_t + ti
                    if t < seq_len:
                        sum_v = T.alloc_local((1,), "float32")
                        sum_sq = T.alloc_local((1,), "float32")
                        count = T.alloc_local((1,), "float32")
                        sum_v[0] = 0.0
                        sum_sq[0] = 0.0
                        count[0] = 0.0

                        for k in T.serial(window):
                            g = t - (window - 1 - k)
                            if g >= 0:
                                v = tile[ti + k]
                                sum_v[0] = sum_v[0] + v
                                sum_sq[0] = sum_sq[0] + v * v
                                count[0] = count[0] + 1.0

                        denom = T.max(count[0], 1.0)
                        mean = sum_v[0] / denom
                        var = T.max(sum_sq[0] / denom - mean * mean, 0.0)
                        out[b, c, t] = (x[b, c, t] - mean) / T.sqrt(var + eps)

        return kernel

    return kernel_builder()


def causal_rolling_zscore_tilelang(x: torch.Tensor, window: int, eps: float = 1e-5) -> torch.Tensor:
    available, reason = tilelang_status()
    if not available:
        raise TileLangUnavailable(reason)
    if x.ndim != 3:
        raise ValueError(f"expected [B, L, C], got shape={tuple(x.shape)}")
    if window < 1:
        raise ValueError("window must be >= 1")
    if not x.is_cuda:
        raise ValueError("tilelang backend requires CUDA tensor")

    x_contig = x.contiguous()
    x_work = x_contig.float().permute(0, 2, 1).contiguous()
    bsz, features, seq_len = x_work.shape

    block_t = 128
    target = _tilelang_target()
    kernel = _build_tilelang_kernel(
        batch_size=int(bsz),
        features=int(features),
        seq_len=int(seq_len),
        window=int(window),
        block_t=int(block_t),
        eps=float(eps),
        target=target,
    )

    out_work = kernel(x_work)
    out = out_work.permute(0, 2, 1).contiguous()
    return out.to(dtype=x.dtype)
