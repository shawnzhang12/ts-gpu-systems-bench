from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.data.etth1 import ETTh1WindowDataset, expand_etth1, split_etth1


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


def test_expand_repeats_rows() -> None:
    df = _fake_frame(128)
    out = expand_etth1(df, repeat_factor=4, drift_per_repeat=0.1, noise_std=0.0, seed=42)
    assert len(out) == 4 * len(df)


def test_standard_split_for_large_frame() -> None:
    df = _fake_frame(20000)
    split = split_etth1(df, split_mode="standard")
    assert len(split.train) == 12 * 30 * 24
    assert len(split.val) == 4 * 30 * 24
    assert len(split.test) == 4 * 30 * 24


def test_memmap_storage_shapes(tmp_path: Path) -> None:
    df = _fake_frame(96)
    ds = ETTh1WindowDataset(
        df,
        lookback=24,
        horizon=1,
        target_col="OT",
        storage="memmap",
        cache_dir=tmp_path,
        cache_tag="utest",
    )
    x, y = ds[0]
    assert x.shape == (24, 7)
    assert y.ndim == 0
