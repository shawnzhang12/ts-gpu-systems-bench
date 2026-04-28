# ts-gpu-systems-bench

End-to-end benchmark for the claim:

> Faster causal rolling z-score preprocessing enables longer lookback windows, which improves ETTh1 forecasting metrics.

The project compares preprocessing backends (`pytorch_eager`, `compile`, `triton`, `tilelang`) and forecasting models (`mamba`, `transformer`) on ETTh1.

## What this repository includes

- ETTh1 data pipeline with canonical long-sequence split logic.
- Causal rolling z-score preprocessing with shared backend interface.
- Backends:
  - `pytorch_eager`: unfold + reduction baseline.
  - `compile`: `torch.compile`-wrapped baseline.
  - `triton`: fused Triton kernel with in-kernel online variance update.
  - `tilelang`: scaffolded backend hook (runtime-gated, falls back to reference path after availability check).
- Two forecasting models:
  - `MambaForecaster` (uses `mamba-ssm` when available, RNN fallback otherwise).
  - `FlashTransformerForecaster` (uses FlashAttention modules when available, `nn.MultiheadAttention` fallback otherwise).
- Hydra config composition, MLflow tracking, Optuna grid sweeps, and profiler trace export.

## Repo structure

- `src/data`: ETTh1 download/loading/splitting/window datasets.
- `src/preprocess`: backend registry and kernel implementations.
- `src/models`: Mamba and Transformer forecasters.
- `src/train`: training engine, profiler helper, and reusable training runner.
- `scripts`: CLI entrypoints for download, benchmarking, training, and sweep.
- `configs`: Hydra config groups.
- `tests`: unit-style checks for data pipeline, preprocessing, and model output shapes.
- `notebooks/analysis.ipynb`: MLflow run analysis and plotting.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional model dependencies:

```bash
pip install -r requirements-models.txt
```

Optional TileLang dependency:

```bash
pip install -r requirements-tilelang.txt
```

## Quick start

Download ETTh1:

```bash
python scripts/download_data.py
```

Run preprocessing benchmark:

```bash
python scripts/benchmark_preprocess.py preprocess=triton data.lookback=720 data.batch_size=32
```

Train one run:

```bash
python scripts/train.py preprocess=triton model=mamba data.lookback=1440 data.batch_size=16
```

Run medium Optuna/Hydra sweep (nested MLflow runs):

```bash
python scripts/sweep.py
```

## Key CLI override examples

```bash
# Compile backend, Transformer model
python scripts/train.py preprocess=compile model=transformer data.lookback=720

# TileLang path (skips gracefully if runtime unavailable)
python scripts/benchmark_preprocess.py preprocess=tilelang data.lookback=1440

# Force larger lookback and smaller batch
python scripts/train.py data.lookback=2880 data.batch_size=16 preprocess=triton model=mamba
```

## Metrics logged to MLflow

- Preprocessing benchmark:
  - `pre_latency_mean_ms`, `pre_latency_p50_ms`, `pre_latency_p95_ms`
  - `pre_peak_mem_mb`
  - `pre_eff_bw_gbps`
  - `pre_mae_vs_eager`, `pre_max_abs_vs_eager`
- Training:
  - `train_loss`, `val_mse`, `test_mse`
  - `train_step_time_s`, `val_step_time_s`
  - `train_samples_per_s`, `test_samples_per_s`
  - `peak_memory_mb`
  - `fit_in_vram`

Profiler traces are exported to `results/traces/` and logged as artifacts.

## TileLang status in this implementation

The TileLang backend is scaffolded with runtime availability checks and full benchmark/training wiring. The execution path is intentionally a reference implementation placeholder until the full fused TileLang kernel is integrated.

This keeps the experiment interface stable while allowing Triton/PyTorch backend execution immediately.

## Analysis notebook

Use `notebooks/analysis.ipynb` to:

- Load MLflow runs into a DataFrame.
- Plot validation MSE vs lookback by backend/model.
- Plot step-time scaling by lookback.
- Compute max fit lookback by backend/model from `fit_in_vram`.

## Validation notes

- Syntax/import validation passed via:

```bash
python -m compileall src scripts tests
```

- `pytest` could not be run in this environment because `pytest` is not installed.
