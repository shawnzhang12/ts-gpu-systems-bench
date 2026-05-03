from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.benchmarks import (
    build_dataloaders,
    dataset_default_target,
    download_dataset,
    expand_dataset,
    load_dataset,
    split_dataset,
)
from src.models import build_model
from src.preprocess.registry import BackendUnavailable, available_backends, get_preprocess_backend
from src.train.engine import evaluate_model, train_model
from src.utils.device import detect_device
from src.utils.metrics import benchmark_callable, peak_memory_mb, reset_peak_memory
from src.utils.seed import seed_everything


def pick_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _is_non_finite(exc: RuntimeError) -> bool:
    return "non-finite" in str(exc).lower()


def _batch_candidates(
    requested_batch_size: int,
    min_batch_size: int,
    auto_batch_backoff: bool,
    max_batch_retries: int,
) -> list[int]:
    if requested_batch_size < 1:
        raise ValueError("requested_batch_size must be >= 1")
    if min_batch_size < 1:
        raise ValueError("min_batch_size must be >= 1")

    candidates = [requested_batch_size]
    if not auto_batch_backoff:
        return candidates

    bsz = requested_batch_size
    retries = 0
    while retries < max_batch_retries:
        nxt = bsz // 2
        if nxt < min_batch_size:
            break
        candidates.append(nxt)
        bsz = nxt
        retries += 1
    return candidates


def _benchmark_preprocess(
    preprocess_fn: Callable[[torch.Tensor, int, float], torch.Tensor],
    x: torch.Tensor,
    window: int,
    eps: float,
) -> dict[str, float]:
    def run_once() -> torch.Tensor:
        return preprocess_fn(x, window=window, eps=eps)

    reset_peak_memory()
    timing = benchmark_callable(run_once, warmup=3, iters=20)
    _ = run_once()
    mem_mb = peak_memory_mb()

    approx_bytes = x.numel() * x.element_size() * (window + 2)
    eff_bw = approx_bytes / max(timing.mean_ms / 1000.0, 1e-12) / 1e9

    return {
        "pre_latency_mean_ms": float(timing.mean_ms),
        "pre_latency_p50_ms": float(timing.p50_ms),
        "pre_latency_p95_ms": float(timing.p95_ms),
        "pre_peak_mem_mb": float(mem_mb),
        "pre_eff_bw_gbps": float(eff_bw),
    }


@torch.no_grad()
def _measure_cold_start_ms(
    model: torch.nn.Module,
    first_batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    preprocess_fn: Callable[[torch.Tensor, int, float], torch.Tensor],
    window: int,
    eps: float,
) -> float:
    model.eval()
    x, _ = first_batch
    x = x.to(device)

    _sync(device)
    start = time.perf_counter()
    x = preprocess_fn(x, window=window, eps=eps)
    _ = model(x)
    _sync(device)
    return (time.perf_counter() - start) * 1000.0


def _run_single_batch_size(
    cfg: DictConfig,
    split,
    preprocess_fn: Callable[[torch.Tensor, int, float], torch.Tensor],
    device: torch.device,
    model_type: str,
    model_kwargs: dict,
    target_col: str,
    batch_size: int,
    amp_enabled: bool,
) -> dict[str, float]:
    num_workers = int(cfg.data.num_workers)
    pin_memory = bool(cfg.data.pin_memory) and device.type == "cuda"
    persistent_workers = bool(cfg.data.persistent_workers) and num_workers > 0

    train_loader, val_loader, test_loader, in_features = build_dataloaders(
        split,
        lookback=int(cfg.data.lookback),
        batch_size=batch_size,
        horizon=int(cfg.data.horizon),
        target_col=target_col,
        num_workers=num_workers,
        storage=str(cfg.data.storage),
        cache_dir=str(cfg.data.cache_dir),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=int(cfg.data.prefetch_factor),
    )

    first_batch = next(iter(train_loader), None)
    if first_batch is None:
        return {
            "fit_in_vram": 0.0,
            "backend_available": 1.0,
            "val_mse": math.inf,
        }

    model = build_model(model_type=model_type, in_features=in_features, **model_kwargs)
    model = model.to(device)

    window = int(cfg.preprocess.window)
    eps = float(cfg.preprocess.eps)

    cold_start_ms = _measure_cold_start_ms(
        model=model,
        first_batch=first_batch,
        device=device,
        preprocess_fn=preprocess_fn,
        window=window,
        eps=eps,
    )

    x_pre, _ = first_batch
    x_pre = x_pre.to(device)
    preprocess_metrics = _benchmark_preprocess(
        preprocess_fn=preprocess_fn,
        x=x_pre,
        window=window,
        eps=eps,
    )

    reset_peak_memory()
    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        preprocess_fn=preprocess_fn,
        window=window,
        eps=eps,
        epochs=int(cfg.train.epochs),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        clip_grad_norm=float(cfg.train.clip_grad_norm),
        amp=amp_enabled,
        grad_accum_steps=int(cfg.train.grad_accum_steps),
    )
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        preprocess_fn=preprocess_fn,
        window=window,
        eps=eps,
    )
    max_mem = peak_memory_mb()

    lookback = int(cfg.data.lookback)
    train_tokens_per_s = float(train_metrics["train_samples_per_s"]) * lookback
    val_tokens_per_s = float(train_metrics["val_samples_per_s"]) * lookback
    infer_tokens_per_s = float(test_metrics["samples_per_s"]) * lookback

    for val in (float(train_metrics["train_loss"]), float(train_metrics["val_mse"]), float(test_metrics["mse"])):
        if not math.isfinite(val):
            raise RuntimeError("non-finite metrics encountered")

    return {
        **preprocess_metrics,
        **train_metrics,
        "cold_start_ms": float(cold_start_ms),
        "test_mse": test_metrics["mse"],
        "infer_step_time_s": test_metrics["step_time_s"],
        "infer_latency_p50_ms": test_metrics["latency_p50_ms"],
        "infer_latency_p95_ms": test_metrics["latency_p95_ms"],
        "test_samples_per_s": test_metrics["samples_per_s"],
        "train_tokens_per_s": train_tokens_per_s,
        "val_tokens_per_s": val_tokens_per_s,
        "infer_tokens_per_s": infer_tokens_per_s,
        "peak_memory_mb": max_mem,
        "fit_in_vram": 1.0,
        "effective_batch_size": float(batch_size),
        "amp_enabled": 1.0 if amp_enabled else 0.0,
    }


def train_once(cfg: DictConfig, nested: bool = False, run_name: str | None = None) -> dict[str, float]:
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

    device = pick_device(str(cfg.train.device))
    status = available_backends()

    backend_name = str(cfg.preprocess.backend)
    if not status.get(backend_name, False):
        return {
            "fit_in_vram": 0.0,
            "backend_available": 0.0,
            "val_mse": math.inf,
        }

    try:
        preprocess_fn = get_preprocess_backend(backend_name)
    except BackendUnavailable:
        return {
            "fit_in_vram": 0.0,
            "backend_available": 0.0,
            "val_mse": math.inf,
        }

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_type = str(model_cfg.pop("type"))

    requested_batch_size = int(cfg.data.batch_size)
    requested_amp = bool(cfg.train.amp)
    amp_modes = [requested_amp] + ([] if not requested_amp else [False])
    batch_candidates = _batch_candidates(
        requested_batch_size=requested_batch_size,
        min_batch_size=int(cfg.train.min_batch_size),
        auto_batch_backoff=bool(cfg.train.auto_batch_backoff),
        max_batch_retries=int(cfg.train.max_batch_retries),
    )

    mlflow.log_params(
        {
            "dataset": dataset_name,
            "target_col": target_col,
            "model_type": model_type,
            "backend": backend_name,
            "lookback": int(cfg.data.lookback),
            "requested_batch_size": requested_batch_size,
            "window": int(cfg.preprocess.window),
            "epochs": int(cfg.train.epochs),
            "lr": float(cfg.train.lr),
            "weight_decay": float(cfg.train.weight_decay),
            "amp": bool(cfg.train.amp),
            "requested_amp": requested_amp,
            "grad_accum_steps": int(cfg.train.grad_accum_steps),
            "auto_batch_backoff": bool(cfg.train.auto_batch_backoff),
            "min_batch_size": int(cfg.train.min_batch_size),
            "max_batch_retries": int(cfg.train.max_batch_retries),
            "data_split_mode": str(cfg.data.split_mode),
            "data_repeat_factor": int(cfg.data.repeat_factor),
            "data_noise_std": float(cfg.data.noise_std),
            "data_storage": str(cfg.data.storage),
            "nested": nested,
        }
    )
    mlflow.log_dict(detect_device().to_dict(), "device.json")
    mlflow.log_dict(status, "backend_status.json")
    mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")
    mlflow.log_metric("backend_available", 1.0)
    mlflow.log_metric("dataset_total_rows", float(len(frame)))
    mlflow.log_metric("dataset_train_rows", float(len(split.train)))
    mlflow.log_metric("dataset_val_rows", float(len(split.val)))
    mlflow.log_metric("dataset_test_rows", float(len(split.test)))
    in_features = len([c for c in frame.columns if c != "date"])
    trainable_params = float(
        sum(
            p.numel()
            for p in build_model(model_type=model_type, in_features=in_features, **model_cfg).parameters()
            if p.requires_grad
        )
    )
    mlflow.log_metric("trainable_params", trainable_params)

    last_oom: RuntimeError | None = None
    for amp_mode_idx, amp_enabled in enumerate(amp_modes):
        for retry_idx, batch_size in enumerate(batch_candidates):
            try:
                metrics = _run_single_batch_size(
                    cfg=cfg,
                    split=split,
                    preprocess_fn=preprocess_fn,
                    device=device,
                    model_type=model_type,
                    model_kwargs=model_cfg,
                    target_col=target_col,
                    batch_size=batch_size,
                    amp_enabled=amp_enabled,
                )
                metrics["fit_retries"] = float(retry_idx)
                metrics["effective_batch_size"] = float(batch_size)
                metrics["batch_backoff_success"] = 1.0 if batch_size != requested_batch_size else 0.0
                metrics["precision_backoff_success"] = 1.0 if amp_mode_idx > 0 else 0.0
                mlflow.log_metrics(metrics)

                if run_name:
                    out_file = Path("results") / f"{run_name}.json"
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    mlflow.log_artifact(str(out_file))

                return metrics
            except RuntimeError as exc:
                if _is_oom(exc):
                    last_oom = exc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                if _is_non_finite(exc):
                    break
                raise

    if last_oom is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = {
        "fit_in_vram": 0.0,
        "val_mse": math.inf,
        "train_loss": math.inf,
        "train_step_time_s": math.inf,
        "infer_latency_p50_ms": math.inf,
        "infer_latency_p95_ms": math.inf,
        "cold_start_ms": math.inf,
        "train_samples_per_s": 0.0,
        "test_samples_per_s": 0.0,
        "train_tokens_per_s": 0.0,
        "val_tokens_per_s": 0.0,
        "infer_tokens_per_s": 0.0,
        "peak_memory_mb": peak_memory_mb(),
        "pre_latency_mean_ms": math.inf,
        "pre_eff_bw_gbps": 0.0,
        "pre_peak_mem_mb": peak_memory_mb(),
        "effective_batch_size": float(batch_candidates[-1]),
        "fit_retries": float(max(len(batch_candidates) - 1, 0)),
        "batch_backoff_success": 0.0,
        "precision_backoff_success": 0.0,
        "amp_enabled": 1.0 if requested_amp else 0.0,
    }
    mlflow.log_metrics(metrics)

    if run_name:
        out_file = Path("results") / f"{run_name}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out_file))

    return metrics
