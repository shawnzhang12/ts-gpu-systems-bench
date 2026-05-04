# ts-gpu-systems-bench

End-to-end benchmark for the claim:

> Faster causal rolling z-score preprocessing enables longer lookback windows, which improves long-context forecasting metrics.

The project compares preprocessing backends (`pytorch_eager`, `compile`, `triton`, `tilelang`) and forecasting models (`mamba`, `transformer`) across multiple time-series benchmarks.

## What this repository includes

- Multi-dataset pipeline with canonical long-sequence split logic:
  - ETT: `etth1`, `etth2`, `ettm1`, `ettm2`
  - Jena Climate: `jena_climate`
  - UCI Electricity Load Diagrams: `electricity_uci`
  - LTSF high-dimensional stress set:
    - `traffic_ltsf` (862 channels, hourly)
    - `electricity_ltsf` (321 channels, hourly)
    - `solar_ltsf` (137 channels, 10-min)
    - `exchange_rate_ltsf` (8 channels, daily)
- Configurable ETTh1 scaling (`repeat_factor`, drift/noise injection) for large-row experiments.
- Storage modes for window datasets (`torch`, `numpy`, `memmap`) to mimic larger production pipelines.
- Causal rolling z-score preprocessing with shared backend interface.
- Backends:
  - `pytorch_eager`: unfold + reduction baseline.
  - `compile`: `torch.compile`-wrapped baseline.
  - `triton`: fused Triton kernel with in-kernel online variance update.
  - `tilelang`: fused TileLang kernel with shared-memory staging for causal rolling z-score.
- Two forecasting models:
  - `MambaForecaster` (uses `mamba-ssm` when available, RNN fallback otherwise).
  - `FlashTransformerForecaster` (uses FlashAttention module or functional API when available, `nn.MultiheadAttention` fallback otherwise).
- Hydra config composition, MLflow tracking, Optuna grid sweeps, and profiler trace export.
- Constraint handling for real-world runs:
  - automatic batch-size backoff on OOM
  - AMP-to-FP32 precision fallback on non-finite loss
  - gradient accumulation for effective large batches

## Repo structure

- `src/data`: dataset specs, download/loading/splitting/window datasets.
- `src/preprocess`: backend registry and kernel implementations.
- `src/models`: Mamba and Transformer forecasters.
- `src/train`: training engine, profiler helper, and reusable training runner.
- `scripts`: CLI entrypoints for download, benchmarking, training, and sweep.
- `configs`: Hydra config groups.
- `tests`: unit-style checks for data pipeline, preprocessing, and model output shapes.
- `notebooks/analysis.ipynb`: MLflow run analysis and plotting.

## Environment setup

```bash
uv sync --extra models --extra tilelang --python 3.12
```

Use `pyproject.toml` + `uv.lock` as the source of truth. `requirements.txt` files are legacy convenience snapshots.

Validate the runtime stack:

```bash
uv run python scripts/check_runtime.py
uv run python scripts/check_runtime.py --require-cuda
```

`pyproject.toml` pins `torch`/`triton` to a tested-compatible range for the Mamba + Triton + TileLang stack.
Current default pin targets CUDA 13.x PyTorch wheels (`torch 2.11`, `cu130`).
The `models` extra also includes `nvidia-cuda-runtime-cu12` as a compatibility shim for current `mamba-ssm` binary linkage.

### System-level prerequisites

- NVIDIA driver with CUDA runtime support (this repo was validated on RTX 5080 + CUDA 13.x / toolkit 13.2).
- CUDA toolkit with `nvcc` available only if you install `flash-attn` from source.
- Linux build toolchain (`gcc`, `g++`, `ninja`).

Optional `flash-attn` install (kept separate because it is the most fragile dependency):

```bash
bash scripts/install_flash_attn.sh
```

`scripts/install_flash_attn.sh` enforces CUDA major-version parity (`13.x` with `13.x`) between `nvcc` and `torch.version.cuda`.
Minor mismatches (for example `torch=13.0` and `nvcc=13.2`) are allowed with a warning.

Optional CUDA-13-first path (FlashAttention-4 pre-release):

```bash
uv pip install --python .venv/bin/python --pre "flash-attn-4[cu13]"
```

### Distrobox note

Use an ML distrobox only if your host toolchain is mismatched or missing CUDA build dependencies. If your host already has working driver + CUDA toolkit + compiler toolchain, distrobox is optional.

### ONNX note

ONNX is not required for this benchmark. Add it only if you want export/deployment inference workflows.

Optional legacy installs:

```bash
uv add -r requirements.txt
uv add -r requirements-models.txt
uv add -r requirements-tilelang.txt
```

## Quick start

Download the default dataset from active Hydra config:

```bash
python scripts/download_data.py
```

Select a larger dataset:

```bash
python scripts/download_data.py data=ettm1
python scripts/download_data.py data=jena_climate
python scripts/download_data.py data=electricity_uci
python scripts/download_data.py data=traffic
python scripts/download_data.py data=electricity
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

Run real-world profile (larger data + larger models + longer contexts):

```bash
python scripts/sweep.py data=etth1_xl train=realworld sweep=realworld
```

Run high-dimensional stress tests (FlashAttention/Mamba pressure):

```bash
python scripts/sweep.py data=traffic train=realworld sweep=realworld
python scripts/sweep.py data=electricity train=realworld sweep=realworld
```

## Latest Traffic Sweeps (2026-05-02 UTC)

- Stress sweep parent run: `56a14e2a39d34ffd9a68e4fe8d5fc711`
- Coverage sweep parent run: `7758f59043f94aada0a1fee559c26425`
- Report: `reports/traffic_sweeps_2026-05-02.md`
- Stress table: `reports/traffic_stress_sweep_2026-05-02_table.md`
- Coverage table: `reports/traffic_coverage_sweep_2026-05-02_table.md`

## Latest Sweep Summary (2026-05-02 UTC)

Latest large-dataset runs were executed on `data=traffic` (17,544 rows, 862 features) to stress long-context forecasting.

- Stress sweep parent run id: `56a14e2a39d34ffd9a68e4fe8d5fc711`
- Coverage sweep parent run id: `7758f59043f94aada0a1fee559c26425`
- Consolidated report: `reports/traffic_sweeps_2026-05-02.md`
- Stress sweep table: `reports/traffic_stress_sweep_2026-05-02_table.md`
- Coverage sweep table: `reports/traffic_coverage_sweep_2026-05-02_table.md`

### Key outcomes

1. Best stress config: `tilelang + mamba_large + L720 + B8` with `val_mse=0.000940`.
2. Best coverage config: `triton + mamba + L192 + B8` with `val_mse=0.000538`.
3. At matched stress settings (`mamba_large, L720, B8`), TileLang preprocessing outperformed Triton:
   - `pre_latency_mean_ms`: `1.183` vs `4.218`
   - `pre_eff_bw_gbps`: `12121` vs `3399`
   - `infer_latency_p50_ms`: `10.02` vs `13.03`
4. `transformer_large` did not fit this 16GB-class GPU in the stress profile, while `mamba_large` reached `L=720`.
5. AMP caused instability in one coverage attempt; `amp=false` restored successful runs and should be the default for this traffic profile.

### Results Table (tabulate)

Representative rows across methods, datasets, and runtime profiles. Table generated with `tabulate`.

| dataset      | profile    | backend       | model       | L   | B   | val_mse   | test_mse   | pre_ms   | pre_bw_gbps   | infer_ms_batch   | train_sps   | test_sps   | peak_mem_mb   |
|--------------|------------|---------------|-------------|-----|-----|-----------|------------|----------|---------------|------------------|-------------|------------|---------------|
| traffic_ltsf | stress     | tilelang      | mamba_large | 720 | 8   | 0.0009    | 0.0062     | 1.1830   | 12121.0800    | 10.0190          | 229.7000    | 801.4000   | 599.2000      |
| traffic_ltsf | stress     | triton        | mamba_large | 720 | 8   | 0.0009    | 0.0062     | 4.2180   | 3399.4100     | 13.0350          | 200.2000    | 608.5000   | 599.9000      |
| traffic_ltsf | coverage   | triton        | mamba       | 192 | 8   | 0.0005    | 0.0011     | 0.3200   | 3211.4400     | 0.6770           | 4799.9000   | 11565.8000 | 45.5000       |
| traffic_ltsf | coverage   | triton        | transformer | 96  | 8   | 0.0008    | 0.0023     | 0.0990   | 2621.0700     | 0.4830           | 4298.5000   | 16060.1000 | 31.5000       |
| traffic_ltsf | coverage   | tilelang      | transformer | 96  | 4   | 0.0010    | 0.0013     | 0.0300   | 4378.2900     | 0.4440           | 2228.8000   | 8611.7000  | 25.5000       |
| etth1        | historical | triton        | mamba       | 720 | 32  | 11.3492   | 42.1133    | NA       | NA            | 1.0204           | 10660.3000  | 29088.8000 | 261.3000      |
| etth1        | historical | tilelang      | mamba       | 720 | 16  | 15.5948   | 34.5924    | NA       | NA            | 2.6918           | 4216.1000   | 5824.6000  | 685.1000      |
| etth1        | historical | pytorch_eager | transformer | 192 | 16  | 15.7595   | 66.9888    | NA       | NA            | 0.4344           | 9833.1000   | 35667.2000 | 80.1000       |

Metric notes:
- `pre_ms` and `pre_bw_gbps` are preprocessing kernel metrics and are only available in runs that logged preprocessing benchmarks.
- `infer_ms_batch` is median inference latency per batch (`infer_latency_p50_ms`).
- `train_sps` and `test_sps` are throughput in samples/sec.

Short summary:
- On `traffic_ltsf` stress runs, TileLang and Triton reached similar accuracy at `L=720`, with TileLang showing faster preprocessing and better effective bandwidth in matched Mamba-large configs.
- On fit-friendly coverage runs, `triton + mamba (L=192, B=8)` gave the best validation error, while Transformer variants delivered very high inference throughput at shorter contexts.
- On this 16GB-class setup, larger Transformer configs remain the main fit bottleneck at long lookbacks.

### Conclusions

1. Faster fused preprocessing is materially improving end-to-end throughput/latency on high-dimensional data.
2. On this hardware budget, model architecture (Mamba vs large Transformer) is currently the main long-context limiter.
3. Cold-start/JIT overhead is significant for some TileLang paths, so warm-run and cold-run metrics should be tracked separately.

### Forward Plan

See [`NEXT_STEPS.md`](NEXT_STEPS.md) for prioritized execution steps and exact commands.

## Historical Sweep Summary (2026-04-29 UTC)

Older ETTh1-focused sweep artifacts are kept for reference:

- Parent run id: `a7de8f1cee76457fae42fad776e995a5`
- Full table: `results/sweep_a7de8f1cee76457fae42fad776e995a5_staff_metrics.md`
- Backend summary: `results/sweep_a7de8f1cee76457fae42fad776e995a5_backend_summary.md`

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
  - `train_step_time_s`, `train_step_p50_ms`, `train_step_p95_ms`
  - `val_step_time_s`, `val_step_p50_ms`, `val_step_p95_ms`
  - `infer_step_time_s`, `infer_latency_p50_ms`, `infer_latency_p95_ms`
  - `cold_start_ms` (compile + first batch)
  - `train_samples_per_s`, `test_samples_per_s`
  - `train_tokens_per_s`, `val_tokens_per_s`, `infer_tokens_per_s`
  - `effective_batch_size`, `fit_retries`, `batch_backoff_success`, `precision_backoff_success`
  - `peak_memory_mb`
  - `fit_in_vram`

Profiler traces are exported to `results/traces/` and logged as artifacts.

## TileLang status in this implementation

The TileLang backend now executes a fused rolling z-score kernel (causal window mean/variance + normalization in a single kernel launch) with shared-memory staging for the block window and left halo.

Runtime checks still guard availability (package/CUDA), but execution no longer falls back to eager preprocessing.

## Analysis notebook

Use `notebooks/analysis.ipynb` to:

- Load MLflow runs into a DataFrame.
- Plot validation MSE vs lookback by backend/model.
- Plot step-time scaling by lookback.
- Compute max fit lookback by backend/model from `fit_in_vram`.

## Validation notes

- Syntax/import validation:

```bash
python -m compileall src scripts tests
```

- Test suite:

```bash
uv run pytest
```

Passed: `7 passed`.
