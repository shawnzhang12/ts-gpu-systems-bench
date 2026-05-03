from __future__ import annotations

import torch
import torch.nn as nn

_FLASH_IMPL = None
_FlashMHA = None
_flash_attn_func = None

try:
    from flash_attn.modules.mha import FlashMHA as _FlashMHA  # type: ignore

    _FLASH_IMPL = "flash_attn.modules.mha.FlashMHA"
except Exception:
    try:
        from flash_attn.modules.mha import MHA as _FlashMHA  # type: ignore

        _FLASH_IMPL = "flash_attn.modules.mha.MHA"
    except Exception:
        try:
            from flash_attn.flash_attention import FlashMHA as _FlashMHA  # type: ignore

            _FLASH_IMPL = "flash_attn.flash_attention.FlashMHA"
        except Exception:
            _FlashMHA = None

if _FlashMHA is None:
    try:
        from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore

        _FLASH_IMPL = "flash_attn.flash_attn_func"
    except Exception:
        try:
            from flash_attn.cute import flash_attn_func as _flash_attn_func  # type: ignore

            _FLASH_IMPL = "flash_attn.cute.flash_attn_func"
        except Exception:
            _flash_attn_func = None


class _AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.use_flash_mha = _FlashMHA is not None
        self.use_flash_func = _flash_attn_func is not None and d_model % n_heads == 0
        self.flash_attn = None
        self.flash_qkv = None
        self.flash_out = None
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if self.use_flash_mha:
            self.flash_attn = _FlashMHA(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                causal=True,
            )
        if self.use_flash_func:
            self.flash_qkv = nn.Linear(d_model, 3 * d_model)
            self.flash_out = nn.Linear(d_model, d_model)

        self.torch_mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def _flash_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.flash_attn(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _flash_func_forward(self, x: torch.Tensor) -> torch.Tensor:
        if _flash_attn_func is None or self.flash_qkv is None or self.flash_out is None:
            return self._torch_mha_forward(x)
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            return self._torch_mha_forward(x)

        batch, seqlen, _ = x.shape
        qkv = self.flash_qkv(x).view(batch, seqlen, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        out = _flash_attn_func(q, k, v, causal=True)
        out = out.reshape(batch, seqlen, self.n_heads * self.head_dim)
        return self.flash_out(out)

    def _torch_mha_forward(self, x: torch.Tensor) -> torch.Tensor:
        seqlen = x.shape[1]
        causal_mask = torch.triu(
            torch.ones((seqlen, seqlen), dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        out, _ = self.torch_mha(x, x, x, attn_mask=causal_mask, need_weights=False)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_flash_mha and x.is_cuda:
            attn_out = self._flash_forward(x)
        elif self.use_flash_func:
            attn_out = self._flash_func_forward(x)
        else:
            attn_out = self._torch_mha_forward(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class FlashTransformerForecaster(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.layers = nn.ModuleList([_AttentionBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        y = self.out_proj(h[:, -1, :])
        return y.squeeze(-1)


def flash_impl_name() -> str:
    return _FLASH_IMPL or "torch.nn.MultiheadAttention fallback"
