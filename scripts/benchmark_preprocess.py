from __future__ import annotations

import json
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig

from src.data.benchmarks import (
    build_dataloaders,
    dataset_default_target,
    download_dataset,
    expand_dataset,
    load_dataset,
    split_dataset,
)
from src.preprocess.pytorch_backend import causal_rolling_zscore_eager
from src.preprocess.registry import BackendUnavailable, available_backends, get_preprocess_backend
from src.train.profiler import export_preprocess_trace
from src.utils.device import detect_device
from src.utils.metrics import benchmark_callable, peak_memory_mb, reset_peak_memory
from src.utils.seed import seed_everything


def _pick_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    dataset_name = str(getattr(cfg.data, "dataset", cfg.data.name))
    target_col = str(getattr(cfg.data, "target_col", dataset_default_target(dataset_name)))

    if cfg.data.download_if_missing:
        download_dataset(dataset_name, cfg.data.path)

    frame = load_dataset(cfg.data.path, dataset_name)
    frame = expand_dataset(
        frame,
        repeat_factor=int(cfg.data.repeat_factor),
        drift_per_repeat=float(cfg.data.drift_per_repeat),
        noise_std=float(cfg.data.noise_std),
        seed=int(cfg.seed),
    )
    split = split_dataset(frame, dataset_name, split_mode=str(cfg.data.split_mode))
    train_loader, _, _, _ = build_dataloaders(
        split,
        lookback=int(cfg.data.lookback),
        batch_size=int(cfg.data.batch_size),
        horizon=int(cfg.data.horizon),
        target_col=target_col,
        num_workers=int(cfg.data.num_workers),
        storage=str(cfg.data.storage),
        cache_dir=str(cfg.data.cache_dir),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=bool(cfg.data.persistent_workers),
        prefetch_factor=int(cfg.data.prefetch_factor),
    )

    x, _ = next(iter(train_loader))
    device = _pick_device(str(cfg.train.device))
    x = x.to(device)

    status = available_backends()
    backend_name = str(cfg.preprocess.backend)

    mlflow.set_tracking_uri(str(cfg.logging.tracking_uri))
    mlflow.set_experiment(str(cfg.experiment_name))

    with mlflow.start_run(run_name=str(cfg.logging.run_name) if cfg.logging.run_name else None):
        mlflow.log_params(
            {
                "backend": backend_name,
                "lookback": int(cfg.data.lookback),
                "batch_size": int(cfg.data.batch_size),
                "window": int(cfg.preprocess.window),
                "eps": float(cfg.preprocess.eps),
            }
        )
        mlflow.log_dict(detect_device().to_dict(), "device.json")
        mlflow.log_dict(status, "backend_status.json")

        if not status.get(backend_name, False):
            mlflow.log_metric("backend_available", 0.0)
            print(json.dumps({"status": "skipped", "reason": f"backend unavailable: {backend_name}"}, indent=2))
            return

        try:
            backend_fn = get_preprocess_backend(backend_name)
        except BackendUnavailable as exc:
            mlflow.log_metric("backend_available", 0.0)
            print(json.dumps({"status": "skipped", "reason": str(exc)}, indent=2))
            return

        window = int(cfg.preprocess.window)
        eps = float(cfg.preprocess.eps)

        reference = causal_rolling_zscore_eager(x, window=window, eps=eps)

        def run_once():
            return backend_fn(x, window=window, eps=eps)

        reset_peak_memory()
        timing = benchmark_callable(run_once, warmup=5, iters=30)
        out = run_once()
        mem_mb = peak_memory_mb()

        mae = torch.mean(torch.abs(out - reference)).item()
        max_abs = torch.max(torch.abs(out - reference)).item()

        approx_bytes = x.numel() * x.element_size() * (window + 2)
        eff_bw = approx_bytes / max(timing.mean_ms / 1000.0, 1e-12) / 1e9

        trace_path = Path("results/traces") / f"trace_{backend_name}_L{cfg.data.lookback}.json"
        export_preprocess_trace(trace_path, run_once, warmup=3, steps=10)

        metrics = {
            "backend_available": 1.0,
            "pre_latency_mean_ms": timing.mean_ms,
            "pre_latency_p50_ms": timing.p50_ms,
            "pre_latency_p95_ms": timing.p95_ms,
            "pre_peak_mem_mb": mem_mb,
            "pre_mae_vs_eager": mae,
            "pre_max_abs_vs_eager": max_abs,
            "pre_eff_bw_gbps": eff_bw,
        }
        mlflow.log_metrics(metrics)

        if bool(cfg.logging.log_artifacts):
            mlflow.log_artifact(str(trace_path))

        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
