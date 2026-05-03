# NEXT STEPS

This file is the re-entry plan for the next session.

## Current status (as of 2026-05-02 UTC)

- Latest production-like dataset: `data=traffic` (862 features).
- Latest stress run parent: `56a14e2a39d34ffd9a68e4fe8d5fc711`.
- Latest coverage run parent: `7758f59043f94aada0a1fee559c26425`.
- Latest report: `reports/traffic_sweeps_2026-05-02.md`.
- Main branch includes all code + report updates from these runs.

## Priority 1: Stabilize fit/fail behavior

Goal: reduce NA/fail rows and make sweep outcomes reproducible.

1. Keep AMP off for traffic baseline sweeps:
   - `train.amp=false`
2. Add a deterministic stability pass:
   - same config across `seed=[42,43,44]`
   - track mean/std of `val_mse`, `fit_in_vram`, `cold_start_ms`
3. Add explicit failure-reason logging:
   - OOM
   - non-finite metrics
   - backend unavailable

## Priority 2: Build fit frontier (max context per model/backend)

Goal: publish a clear context-length capability chart.

1. Sweep lookbacks per profile, e.g. `[96,192,336,720,960,1440]`.
2. Keep batch-size backoff enabled to find true fit boundary.
3. Export per `(backend, model)`:
   - max fit lookback
   - best `val_mse` at each lookback
   - median `infer_latency_p50_ms`

## Priority 3: Deployment-style inference harness

Goal: make inference claims production-relevant on one GPU.

1. Add warmup phase and separate:
   - cold-start latency
   - warm steady-state latency (`p50/p95/p99`)
2. Add concurrency modes:
   - single-stream
   - multi-stream (2/4 streams)
3. Track:
   - tokens/sec
   - samples/sec
   - peak memory
   - jitter (`p95 - p50`)

## Priority 4: Cross-dataset validation

Goal: verify conclusions generalize beyond traffic.

Run the same benchmark harness on:

1. `data=electricity` (321 features)
2. `data=solar` (137 features)
3. `data=ettm1` (time-resolution stress case)

Compare:

1. backend delta (`triton` vs `tilelang`)
2. architecture delta (`mamba` vs `transformer`)
3. fit frontier shifts by dataset

## Suggested command set (safe baseline)

```bash
# 1) Runtime validation
uv run python scripts/check_runtime.py --require-cuda

# 2) Stress sweep (current baseline)
uv run python scripts/sweep.py \
  data=traffic train=realworld \
  train.epochs=4 train.grad_accum_steps=2 \
  sweep.lookbacks=[192,336,720] \
  sweep.batch_sizes=[8,16] \
  sweep.backends=[triton,tilelang] \
  sweep.models=[mamba_large,transformer_large] \
  sweep.max_trials=24

# 3) Coverage sweep (fit-friendly, lower NA)
uv run python scripts/sweep.py \
  data=traffic train=default \
  train.epochs=3 train.amp=false \
  sweep.lookbacks=[96,192] \
  sweep.batch_sizes=[4,8] \
  sweep.backends=[triton,tilelang] \
  sweep.models=[mamba,transformer] \
  sweep.max_trials=16
```

## When you come back

1. Open latest MLflow runs and confirm artifact integrity.
2. Re-run coverage sweep first (`amp=false`) to verify environment consistency.
3. Then run stress sweep and compare to previous parent run IDs.
4. Update `reports/` and README latest section with new run IDs and conclusions.
