from __future__ import annotations

import json
import math
from pathlib import Path

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.etth1 import build_dataloaders, download_etth1, load_etth1, split_etth1
from src.models import build_model
from src.preprocess.registry import BackendUnavailable, available_backends, get_preprocess_backend
from src.train.engine import evaluate_model, train_model
from src.utils.device import detect_device
from src.utils.metrics import peak_memory_mb, reset_peak_memory
from src.utils.seed import seed_everything


def pick_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def train_once(cfg: DictConfig, nested: bool = False, run_name: str | None = None) -> dict[str, float]:
    seed_everything(int(cfg.seed))

    if cfg.data.download_if_missing:
        download_etth1(cfg.data.path)

    frame = load_etth1(cfg.data.path)
    split = split_etth1(frame)
    train_loader, val_loader, test_loader, in_features = build_dataloaders(
        split,
        lookback=int(cfg.data.lookback),
        batch_size=int(cfg.data.batch_size),
        horizon=int(cfg.data.horizon),
        target_col=str(cfg.data.target_col),
        num_workers=int(cfg.data.num_workers),
    )

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
    model = build_model(model_type=model_type, in_features=in_features, **model_cfg)

    mlflow.log_params(
        {
            "model_type": model_type,
            "backend": backend_name,
            "lookback": int(cfg.data.lookback),
            "batch_size": int(cfg.data.batch_size),
            "window": int(cfg.preprocess.window),
            "epochs": int(cfg.train.epochs),
            "lr": float(cfg.train.lr),
            "weight_decay": float(cfg.train.weight_decay),
            "amp": bool(cfg.train.amp),
            "nested": nested,
        }
    )
    mlflow.log_dict(detect_device().to_dict(), "device.json")
    mlflow.log_dict(status, "backend_status.json")
    mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")
    mlflow.log_metric("backend_available", 1.0)
    mlflow.log_metric("trainable_params", float(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    try:
        reset_peak_memory()
        train_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            preprocess_fn=preprocess_fn,
            window=int(cfg.preprocess.window),
            eps=float(cfg.preprocess.eps),
            epochs=int(cfg.train.epochs),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
            clip_grad_norm=float(cfg.train.clip_grad_norm),
            amp=bool(cfg.train.amp),
        )
        test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            preprocess_fn=preprocess_fn,
            window=int(cfg.preprocess.window),
            eps=float(cfg.preprocess.eps),
        )
        max_mem = peak_memory_mb()

        metrics = {
            **train_metrics,
            "test_mse": test_metrics["mse"],
            "test_samples_per_s": test_metrics["samples_per_s"],
            "peak_memory_mb": max_mem,
            "fit_in_vram": 1.0,
        }

    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metrics = {
            "fit_in_vram": 0.0,
            "val_mse": math.inf,
            "train_loss": math.inf,
            "train_step_time_s": math.inf,
            "train_samples_per_s": 0.0,
            "peak_memory_mb": peak_memory_mb(),
        }

    mlflow.log_metrics(metrics)

    if run_name:
        out_file = Path("results") / f"{run_name}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out_file))

    return metrics
