# Traffic Dataset Sweeps (2026-05-02 UTC)

Dataset profile (`data=traffic`): 17,544 rows, 862 features (high-dimensional LTSF stress).

## Sweep A: Stress profile

- MLflow parent run: `56a14e2a39d34ffd9a68e4fe8d5fc711`
- Command class: `train=realworld`, `lookbacks=[192,336,720]`, `models=[mamba_large,transformer_large]`, `backends=[triton,tilelang]`, `batch_sizes=[8,16]`, `epochs=4`
- Trials: `24`
- Successful VRAM-fit trials: `12`
- Best objective (`val_mse`): `0.0009402344`
- Best config: `tilelang + mamba_large + lookback=720 + batch=8`
- Max fit lookback:
  - `triton:mamba_large = 720`
  - `tilelang:mamba_large = 720`
  - `triton:transformer_large = 0`
  - `tilelang:transformer_large = 0`
- Full table: `reports/traffic_stress_sweep_2026-05-02_table.md`

## Sweep B: Coverage profile

- MLflow parent run: `7758f59043f94aada0a1fee559c26425`
- Command class: `train=default`, `lookbacks=[96,192]`, `models=[mamba,transformer]`, `backends=[triton,tilelang]`, `batch_sizes=[4,8]`, `epochs=3`, `amp=false`
- Trials: `16`
- Successful VRAM-fit trials: `9`
- Best objective (`val_mse`): `0.0005379704`
- Best config: `triton + mamba + lookback=192 + batch=8`
- Max fit lookback:
  - `triton:transformer = 192`
  - `tilelang:transformer = 192`
  - `triton:mamba = 192`
  - `tilelang:mamba = 0` (current instability in this profile)
- Full table: `reports/traffic_coverage_sweep_2026-05-02_table.md`

## Notes

- The first attempted broad realworld sweep was intentionally stopped because it would take too long on a display-attached 16GB GPU.
- A separate failed coverage attempt with `amp=true` was diagnosed and corrected; `amp=false` restored successful training runs.
- MLflow contains all nested-run metrics, plus per-run artifacts (`config.yaml`, `device.json`, `backend_status.json`, per-run metrics JSON, and sweep summary files).
