from __future__ import annotations

import pandas as pd
import torch

from src.data.etth1 import ETTh1WindowDataset, split_etth1


def _fake_frame(n: int = 200) -> pd.DataFrame:
    values = {
        "date": pd.date_range("2020-01-01", periods=n, freq="h"),
        "HUFL": torch.arange(n).numpy(),
        "HULL": torch.arange(n).numpy(),
        "MUFL": torch.arange(n).numpy(),
        "MULL": torch.arange(n).numpy(),
        "LUFL": torch.arange(n).numpy(),
        "LULL": torch.arange(n).numpy(),
        "OT": torch.arange(n).numpy(),
    }
    return pd.DataFrame(values)


def test_split_shapes_sum_to_len() -> None:
    df = _fake_frame(200)
    split = split_etth1(df)
    assert len(split.train) + len(split.val) + len(split.test) == len(df)


def test_window_dataset_shapes() -> None:
    df = _fake_frame(64)
    ds = ETTh1WindowDataset(df, lookback=16, horizon=1, target_col="OT")
    x, y = ds[0]
    assert x.shape == (16, 7)
    assert y.ndim == 0
