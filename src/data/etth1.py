from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
ETTH1_COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
STANDARD_TRAIN_END = 12 * 30 * 24
STANDARD_VAL_END = 16 * 30 * 24
STANDARD_TEST_END = 20 * 30 * 24


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def download_etth1(path: str | Path = "data/ETTh1.csv", force: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return out

    df = pd.read_csv(ETTH1_URL)
    df.to_csv(out, index=False)
    return out


def load_etth1(path: str | Path = "data/ETTh1.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ETTH1_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ETTh1 columns missing: {missing}")
    return df


def split_etth1(df: pd.DataFrame) -> SplitFrames:
    if len(df) >= STANDARD_TEST_END:
        train = df.iloc[:STANDARD_TRAIN_END].reset_index(drop=True)
        val = df.iloc[STANDARD_TRAIN_END:STANDARD_VAL_END].reset_index(drop=True)
        test = df.iloc[STANDARD_VAL_END:STANDARD_TEST_END].reset_index(drop=True)
        return SplitFrames(train=train, val=val, test=test)

    train_end = int(0.7 * len(df))
    val_end = int(0.85 * len(df))
    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return SplitFrames(train=train, val=val, test=test)


class ETTh1WindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        lookback: int,
        horizon: int = 1,
        target_col: str = "OT",
        features: list[str] | None = None,
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        self.lookback = lookback
        self.horizon = horizon

        if features is None:
            features = [c for c in frame.columns if c != "date"]

        if target_col not in features:
            raise ValueError(f"target_col '{target_col}' must be in features")

        self.features = features
        self.target_col = target_col
        self.target_idx = self.features.index(target_col)

        values = torch.tensor(frame[self.features].values, dtype=torch.float32)
        n = values.shape[0] - lookback - horizon + 1
        if n <= 0:
            raise ValueError(
                f"insufficient rows={values.shape[0]} for lookback={lookback}, horizon={horizon}"
            )
        self.values = values
        self.num_windows = n

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.lookback
        x = self.values[start:end]
        y_index = end + self.horizon - 1
        y = self.values[y_index, self.target_idx]
        return x, y


def build_dataloaders(
    split: SplitFrames,
    lookback: int,
    batch_size: int,
    horizon: int = 1,
    target_col: str = "OT",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    train_ds = ETTh1WindowDataset(split.train, lookback=lookback, horizon=horizon, target_col=target_col)
    val_ds = ETTh1WindowDataset(split.val, lookback=lookback, horizon=horizon, target_col=target_col)
    test_ds = ETTh1WindowDataset(split.test, lookback=lookback, horizon=horizon, target_col=target_col)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    in_features = len(train_ds.features)
    return train_loader, val_loader, test_loader, in_features


def get_split_frame(split: SplitFrames, partition: Literal["train", "val", "test"]) -> pd.DataFrame:
    if partition == "train":
        return split.train
    if partition == "val":
        return split.val
    if partition == "test":
        return split.test
    raise ValueError(f"unknown partition: {partition}")
