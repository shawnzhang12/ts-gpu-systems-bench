from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None and tl is not None


if TRITON_AVAILABLE:

    @triton.jit
    def _causal_rolling_zscore_kernel(
        x_ptr,
        out_ptr,
        b_stride,
        l_stride,
        c_stride,
        B,
        L,
        C,
        EPS,
        WINDOW: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_bc = tl.program_id(1)

        b = pid_bc // C
        c = pid_bc % C

        t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < L

        base = x_ptr + b * b_stride + c * c_stride
        out_base = out_ptr + b * b_stride + c * c_stride

        mean = tl.zeros([BLOCK_T], dtype=tl.float32)
        m2 = tl.zeros([BLOCK_T], dtype=tl.float32)
        count = tl.zeros([BLOCK_T], dtype=tl.float32)

        for i in range(WINDOW):
            idx = t_offsets - (WINDOW - 1 - i)
            valid = t_mask & (idx >= 0)
            vals = tl.load(base + idx * l_stride, mask=valid, other=0.0).to(tl.float32)

            count_new = count + valid.to(tl.float32)
            delta = vals - mean
            mean = tl.where(valid, mean + delta / tl.maximum(count_new, 1.0), mean)
            delta2 = vals - mean
            m2 = tl.where(valid, m2 + delta * delta2, m2)
            count = count_new

        var = m2 / tl.maximum(count, 1.0)
        std = tl.sqrt(var + EPS)
        x_now = tl.load(base + t_offsets * l_stride, mask=t_mask, other=0.0).to(tl.float32)
        z = (x_now - mean) / std

        tl.store(out_base + t_offsets * l_stride, z, mask=t_mask)


def causal_rolling_zscore_triton(
    x: torch.Tensor,
    window: int,
    eps: float = 1e-5,
    block_t: int = 128,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise ValueError("Triton backend requires CUDA tensor")
    if x.ndim != 3:
        raise ValueError(f"expected [B, L, C], got shape={tuple(x.shape)}")
    if window < 1:
        raise ValueError("window must be >= 1")

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig, dtype=torch.float32)

    B, L, C = x_contig.shape
    grid = (triton.cdiv(L, block_t), B * C)

    _causal_rolling_zscore_kernel[grid](
        x_contig,
        out,
        x_contig.stride(0),
        x_contig.stride(1),
        x_contig.stride(2),
        B,
        L,
        C,
        eps,
        WINDOW=window,
        BLOCK_T=block_t,
    )

    return out.to(dtype=x.dtype)
