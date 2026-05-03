from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .benchmarks import (
    SplitFrames,
    TimeSeriesWindowDataset,
    build_dataloaders as _build_dataloaders,
    download_dataset,
    expand_dataset,
    load_dataset,
    split_dataset,
)

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
ETTH1_COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
STANDARD_TRAIN_END = 12 * 30 * 24
STANDARD_VAL_END = 16 * 30 * 24
STANDARD_TEST_END = 20 * 30 * 24

# Backward-compatibility alias
ETTh1WindowDataset = TimeSeriesWindowDataset


def download_etth1(path: str | Path = "data/ETTh1.csv", force: bool = False) -> Path:
    return download_dataset("etth1", path=path, force=force)


def load_etth1(path: str | Path = "data/ETTh1.csv") -> pd.DataFrame:
    df = load_dataset(path=path, name="etth1")
    missing = [c for c in ETTH1_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ETTh1 columns missing: {missing}")
    return df


def expand_etth1(
    df: pd.DataFrame,
    repeat_factor: int = 1,
    drift_per_repeat: float = 0.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    return expand_dataset(
        df=df,
        repeat_factor=repeat_factor,
        drift_per_repeat=drift_per_repeat,
        noise_std=noise_std,
        seed=seed,
    )


def split_etth1(df: pd.DataFrame, split_mode: str = "standard") -> SplitFrames:
    return split_dataset(df=df, name="etth1", split_mode=split_mode)


def build_dataloaders(
    split: SplitFrames,
    lookback: int,
    batch_size: int,
    horizon: int = 1,
    target_col: str = "OT",
    num_workers: int = 0,
    storage: Literal["torch", "numpy", "memmap"] = "torch",
    cache_dir: str | Path = "data/cache",
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    return _build_dataloaders(
        split=split,
        lookback=lookback,
        batch_size=batch_size,
        horizon=horizon,
        target_col=target_col,
        num_workers=num_workers,
        storage=storage,
        cache_dir=cache_dir,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def get_split_frame(split: SplitFrames, partition: Literal["train", "val", "test"]) -> pd.DataFrame:
    if partition == "train":
        return split.train
    if partition == "val":
        return split.val
    if partition == "test":
        return split.test
    raise ValueError(f"unknown partition: {partition}")

