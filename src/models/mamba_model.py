from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba as _Mamba
except Exception:
    _Mamba = None


class _FallbackMambaBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


class _AdaptiveMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.mamba = _Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) if _Mamba is not None else None
        self.fallback = _FallbackMambaBlock(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mamba is not None and x.is_cuda:
            return self.mamba(x)
        return self.fallback(x)


class MambaForecaster(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)

        blocks: list[nn.Module] = []
        for _ in range(n_layers):
            blocks.append(_AdaptiveMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        y = self.out_proj(h[:, -1, :])
        return y.squeeze(-1)
