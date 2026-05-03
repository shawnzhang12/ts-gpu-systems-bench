from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.data.benchmarks import (
    STANDARD_MINUTE_TEST_END,
    STANDARD_MINUTE_TRAIN_END,
    STANDARD_MINUTE_VAL_END,
    canonical_dataset_name,
    list_dataset_names,
    load_dataset,
    split_dataset,
)


def _fake_ett_frame(n: int = 70000) -> pd.DataFrame:
    values = {
        "date": pd.date_range("2020-01-01", periods=n, freq="15min"),
        "HUFL": torch.arange(n).numpy(),
        "HULL": torch.arange(n).numpy(),
        "MUFL": torch.arange(n).numpy(),
        "MULL": torch.arange(n).numpy(),
        "LUFL": torch.arange(n).numpy(),
        "LULL": torch.arange(n).numpy(),
        "OT": torch.arange(n).numpy(),
    }
    return pd.DataFrame(values)


def test_registry_contains_complex_datasets() -> None:
    names = list_dataset_names()
    assert "ettm1" in names
    assert "ettm2" in names
    assert "jena_climate" in names
    assert "electricity_uci" in names
    assert "electricity_ltsf" in names
    assert "traffic_ltsf" in names
    assert "solar_ltsf" in names
    assert "exchange_rate_ltsf" in names


def test_canonical_name_handles_xl_alias() -> None:
    assert canonical_dataset_name("etth1_xl") == "etth1"
    assert canonical_dataset_name("traffic") == "traffic_ltsf"
    assert canonical_dataset_name("electricity") == "electricity_ltsf"


def test_standard_split_for_ettm1() -> None:
    df = _fake_ett_frame()
    split = split_dataset(df, name="ettm1", split_mode="standard")
    assert len(split.train) == STANDARD_MINUTE_TRAIN_END
    assert len(split.val) == STANDARD_MINUTE_VAL_END - STANDARD_MINUTE_TRAIN_END
    assert len(split.test) == STANDARD_MINUTE_TEST_END - STANDARD_MINUTE_VAL_END


def test_load_jena_schema(tmp_path: Path) -> None:
    p = tmp_path / "jena.csv"
    p.write_text(
        '"Date Time","p (mbar)","T (degC)"\n'
        "01.01.2009 00:10:00,996.52,-8.02\n"
        "01.01.2009 00:20:00,996.57,-8.41\n",
        encoding="utf-8",
    )
    df = load_dataset(p, "jena_climate")
    assert "date" in df.columns
    assert "T (degC)" in df.columns
    assert df["T (degC)"].dtype.kind in {"f", "i"}


def test_load_electricity_semicolon_schema(tmp_path: Path) -> None:
    p = tmp_path / "LD2011_2014.txt"
    p.write_text(
        '"";"MT_001";"MT_002"\n'
        '"2011-01-01 00:15:00";71,7703349282297;0\n'
        '"2011-01-01 00:30:00";62,200956937799;1,25\n',
        encoding="utf-8",
    )
    df = load_dataset(p, "electricity_uci")
    assert "date" in df.columns
    assert "MT_001" in df.columns
    assert "MT_002" in df.columns
    assert df["MT_001"].dtype.kind in {"f", "i"}


def test_load_ltsf_noheader_schema(tmp_path: Path) -> None:
    p = tmp_path / "traffic.txt.gz"
    pd.DataFrame([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]).to_csv(
        p,
        index=False,
        header=False,
        compression="gzip",
    )
    df = load_dataset(p, "traffic_ltsf")
    assert "date" in df.columns
    assert "f000" in df.columns
    assert "f001" in df.columns
    assert df["f000"].dtype.kind in {"f", "i"}
