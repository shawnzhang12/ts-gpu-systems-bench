from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

STANDARD_HOURLY_TRAIN_END = 12 * 30 * 24
STANDARD_HOURLY_VAL_END = 16 * 30 * 24
STANDARD_HOURLY_TEST_END = 20 * 30 * 24

STANDARD_MINUTE_TRAIN_END = 12 * 30 * 24 * 4
STANDARD_MINUTE_VAL_END = 16 * 30 * 24 * 4
STANDARD_MINUTE_TEST_END = 20 * 30 * 24 * 4


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    archive: Literal["none", "zip"] = "none"
    archive_member: str | None = None
    parser: Literal["csv", "semicolon_decimal_csv", "csv_no_header"] = "csv"
    date_col: str = "date"
    target_col: str = "OT"
    standard_split: tuple[int, int, int] | None = None
    synthetic_date_start: str | None = None
    synthetic_date_freq: str = "h"
    feature_prefix: str = "f"


DATASET_SPECS: dict[str, DatasetSpec] = {
    "etth1": DatasetSpec(
        name="etth1",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        target_col="OT",
        standard_split=(STANDARD_HOURLY_TRAIN_END, STANDARD_HOURLY_VAL_END, STANDARD_HOURLY_TEST_END),
    ),
    "etth2": DatasetSpec(
        name="etth2",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        target_col="OT",
        standard_split=(STANDARD_HOURLY_TRAIN_END, STANDARD_HOURLY_VAL_END, STANDARD_HOURLY_TEST_END),
    ),
    "ettm1": DatasetSpec(
        name="ettm1",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        target_col="OT",
        standard_split=(STANDARD_MINUTE_TRAIN_END, STANDARD_MINUTE_VAL_END, STANDARD_MINUTE_TEST_END),
    ),
    "ettm2": DatasetSpec(
        name="ettm2",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        target_col="OT",
        standard_split=(STANDARD_MINUTE_TRAIN_END, STANDARD_MINUTE_VAL_END, STANDARD_MINUTE_TEST_END),
    ),
    "jena_climate": DatasetSpec(
        name="jena_climate",
        url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        archive="zip",
        archive_member="jena_climate_2009_2016.csv",
        date_col="Date Time",
        target_col="T (degC)",
        standard_split=None,
    ),
    "electricity_uci": DatasetSpec(
        name="electricity_uci",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
        archive="zip",
        archive_member="LD2011_2014.txt",
        parser="semicolon_decimal_csv",
        date_col="Unnamed: 0",
        target_col="MT_001",
        standard_split=None,
    ),
    "electricity_ltsf": DatasetSpec(
        name="electricity_ltsf",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
        parser="csv_no_header",
        target_col="f000",
        standard_split=None,
        synthetic_date_start="2012-01-01 00:00:00",
        synthetic_date_freq="h",
    ),
    "traffic_ltsf": DatasetSpec(
        name="traffic_ltsf",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
        parser="csv_no_header",
        target_col="f000",
        standard_split=None,
        synthetic_date_start="2015-01-01 00:00:00",
        synthetic_date_freq="h",
    ),
    "solar_ltsf": DatasetSpec(
        name="solar_ltsf",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz",
        parser="csv_no_header",
        target_col="f000",
        standard_split=None,
        synthetic_date_start="2006-01-01 00:00:00",
        synthetic_date_freq="10min",
    ),
    "exchange_rate_ltsf": DatasetSpec(
        name="exchange_rate_ltsf",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
        parser="csv_no_header",
        target_col="f000",
        standard_split=None,
        synthetic_date_start="1990-01-01 00:00:00",
        synthetic_date_freq="D",
    ),
}


DATASET_ALIASES: dict[str, str] = {
    "ecl": "electricity_ltsf",
    "electricity": "electricity_ltsf",
    "traffic": "traffic_ltsf",
    "solar": "solar_ltsf",
    "exchange_rate": "exchange_rate_ltsf",
    "exchange": "exchange_rate_ltsf",
}


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def canonical_dataset_name(name: str) -> str:
    key = str(name).lower()
    key = DATASET_ALIASES.get(key, key)
    if key in DATASET_SPECS:
        return key
    if key.endswith("_xl"):
        base = key[:-3]
        if base in DATASET_SPECS:
            return base
    known = ", ".join(sorted(DATASET_SPECS.keys()))
    raise ValueError(f"unknown dataset '{name}', expected one of: {known}")


def get_dataset_spec(name: str) -> DatasetSpec:
    return DATASET_SPECS[canonical_dataset_name(name)]


def dataset_default_target(name: str) -> str:
    return get_dataset_spec(name).target_col


def list_dataset_names() -> list[str]:
    return sorted(DATASET_SPECS.keys())


def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def download_dataset(name: str, path: str | Path, force: bool = False) -> Path:
    spec = get_dataset_spec(name)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return out

    raw = _download_bytes(spec.url)
    if spec.archive == "none":
        out.write_bytes(raw)
        return out

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        member = spec.archive_member
        if member is None:
            candidates = [n for n in zf.namelist() if not n.endswith("/") and "__MACOSX" not in n]
            if not candidates:
                raise ValueError(f"no extractable files found for dataset={name}")
            member = candidates[0]
        if member not in zf.namelist():
            basename = Path(member).name
            matches = [n for n in zf.namelist() if Path(n).name == basename]
            if not matches:
                raise ValueError(f"archive member not found: {member}")
            member = matches[0]
        out.write_bytes(zf.read(member))
    return out


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def load_dataset(path: str | Path, name: str) -> pd.DataFrame:
    spec = get_dataset_spec(name)
    if spec.parser == "csv":
        df = pd.read_csv(path)
    elif spec.parser == "semicolon_decimal_csv":
        df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    elif spec.parser == "csv_no_header":
        df = pd.read_csv(path, header=None, compression="infer")
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df.columns = [f"{spec.feature_prefix}{i:03d}" for i in range(df.shape[1])]
        start = pd.Timestamp(spec.synthetic_date_start or "2000-01-01 00:00:00")
        dates = pd.date_range(start=start, periods=len(df), freq=spec.synthetic_date_freq)
        df.insert(0, "date", dates.strftime("%Y-%m-%d %H:%M:%S"))
        return df
    else:
        raise ValueError(f"unsupported parser: {spec.parser}")

    df = _standardize_columns(df)
    date_candidates = [spec.date_col, "date", "Date Time", "Unnamed: 0", ""]
    found_date_col = next((c for c in date_candidates if c in df.columns), None)
    if found_date_col is None:
        found_date_col = str(df.columns[0])
    df = df.rename(columns={found_date_col: "date"})

    numeric_cols = [c for c in df.columns if c != "date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    ordered_cols = ["date"] + numeric_cols
    return df[ordered_cols]


def expand_dataset(
    df: pd.DataFrame,
    repeat_factor: int = 1,
    drift_per_repeat: float = 0.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    if repeat_factor <= 1 and abs(drift_per_repeat) <= 0.0 and abs(noise_std) <= 0.0:
        return df
    if repeat_factor < 1:
        raise ValueError("repeat_factor must be >= 1")

    rng = np.random.default_rng(seed)
    numeric_cols = [c for c in df.columns if c != "date"]
    base_dates = pd.to_datetime(df["date"], errors="coerce")
    delta = _infer_time_delta(base_dates)
    n_rows = len(df)
    out_frames: list[pd.DataFrame] = []

    for rep in range(repeat_factor):
        chunk = df.copy(deep=True)
        chunk_dates = base_dates + (rep * n_rows * delta)
        chunk["date"] = chunk_dates.dt.strftime("%Y-%m-%d %H:%M:%S")

        vals = chunk[numeric_cols].to_numpy(dtype=np.float32, copy=True)
        if drift_per_repeat != 0.0:
            vals += float(rep) * float(drift_per_repeat)
        if noise_std > 0.0:
            vals += rng.normal(loc=0.0, scale=float(noise_std), size=vals.shape).astype(np.float32)
        chunk[numeric_cols] = vals
        out_frames.append(chunk)

    return pd.concat(out_frames, ignore_index=True)


def _infer_time_delta(dates: pd.Series) -> pd.Timedelta:
    valid = dates.dropna()
    if len(valid) < 2:
        return pd.Timedelta(hours=1)
    diffs = valid.diff().dropna()
    if len(diffs) == 0:
        return pd.Timedelta(hours=1)
    delta = diffs.mode().iloc[0]
    if delta <= pd.Timedelta(0):
        return pd.Timedelta(hours=1)
    return delta


def split_dataset(df: pd.DataFrame, name: str, split_mode: str = "standard") -> SplitFrames:
    spec = get_dataset_spec(name)
    split_mode = split_mode.lower()

    if split_mode == "standard" and spec.standard_split is not None:
        train_end, val_end, test_end = spec.standard_split
        if len(df) >= test_end:
            train = df.iloc[:train_end].reset_index(drop=True)
            val = df.iloc[train_end:val_end].reset_index(drop=True)
            test = df.iloc[val_end:test_end].reset_index(drop=True)
            return SplitFrames(train=train, val=val, test=test)

    if split_mode not in {"standard", "proportional", "ltsf"}:
        raise ValueError(f"split_mode must be 'standard', 'proportional', or 'ltsf', got {split_mode}")

    train_end = int(0.7 * len(df))
    if split_mode == "ltsf":
        val_end = int(0.8 * len(df))
    else:
        val_end = int(0.85 * len(df))
    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return SplitFrames(train=train, val=val, test=test)


class TimeSeriesWindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        lookback: int,
        horizon: int = 1,
        target_col: str = "OT",
        features: list[str] | None = None,
        storage: Literal["torch", "numpy", "memmap"] = "torch",
        cache_dir: str | Path = "data/cache",
        cache_tag: str = "train",
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
        self.storage = storage

        values_np = frame[self.features].to_numpy(dtype=np.float32, copy=True)
        if storage == "torch":
            values: torch.Tensor | np.ndarray = torch.from_numpy(values_np)
        elif storage == "numpy":
            values = values_np
        elif storage == "memmap":
            cache_root = Path(cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            key = _dataset_cache_key(values_np=values_np, target_col=target_col, cache_tag=cache_tag)
            cache_path = cache_root / f"{key}.npy"
            if not cache_path.exists():
                np.save(cache_path, values_np)
            values = np.load(cache_path, mmap_mode="r")
        else:
            raise ValueError(f"storage must be torch|numpy|memmap, got: {storage}")

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
        y_index = end + self.horizon - 1
        if isinstance(self.values, torch.Tensor):
            x = self.values[start:end]
            y = self.values[y_index, self.target_idx]
            return x, y

        x_np = np.ascontiguousarray(self.values[start:end], dtype=np.float32)
        y_val = float(self.values[y_index, self.target_idx])
        x = torch.from_numpy(x_np)
        y = torch.tensor(y_val, dtype=torch.float32)
        return x, y


def _dataset_cache_key(values_np: np.ndarray, target_col: str, cache_tag: str) -> str:
    h = hashlib.sha1()
    h.update(str(values_np.shape).encode("utf-8"))
    h.update(target_col.encode("utf-8"))
    h.update(cache_tag.encode("utf-8"))
    h.update(values_np[: min(1024, values_np.shape[0])].tobytes())
    return h.hexdigest()


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
    train_ds = TimeSeriesWindowDataset(
        split.train,
        lookback=lookback,
        horizon=horizon,
        target_col=target_col,
        storage=storage,
        cache_dir=cache_dir,
        cache_tag="train",
    )
    val_ds = TimeSeriesWindowDataset(
        split.val,
        lookback=lookback,
        horizon=horizon,
        target_col=target_col,
        storage=storage,
        cache_dir=cache_dir,
        cache_tag="val",
    )
    test_ds = TimeSeriesWindowDataset(
        split.test,
        lookback=lookback,
        horizon=horizon,
        target_col=target_col,
        storage=storage,
        cache_dir=cache_dir,
        cache_tag="test",
    )

    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

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
