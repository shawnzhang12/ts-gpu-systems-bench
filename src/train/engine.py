from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochStats:
    loss: float
    step_time_s: float
    samples_per_s: float


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    preprocess_fn: Callable[[torch.Tensor, int, float], torch.Tensor],
    window: int,
    eps: float,
    optimizer: torch.optim.Optimizer | None,
    clip_grad_norm: float | None,
    amp: bool,
) -> EpochStats:
    criterion = nn.MSELoss()
    losses: list[float] = []
    step_times: list[float] = []

    total_samples = 0
    total_time = 0.0

    is_train = optimizer is not None
    model.train(is_train)

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        start = time.perf_counter()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        x = preprocess_fn(x, window=window, eps=eps)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=autocast_device, enabled=amp):
                pred = model(x)
                loss = criterion(pred, y)

            if is_train:
                loss.backward()
                if clip_grad_norm is not None and clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        dt = time.perf_counter() - start
        losses.append(loss.detach().item())
        step_times.append(dt)
        total_samples += x.shape[0]
        total_time += dt

    mean_loss = sum(losses) / max(len(losses), 1)
    mean_step = sum(step_times) / max(len(step_times), 1)
    sps = total_samples / max(total_time, 1e-12)
    return EpochStats(loss=mean_loss, step_time_s=mean_step, samples_per_s=sps)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    preprocess_fn: Callable[[torch.Tensor, int, float], torch.Tensor],
    window: int,
    eps: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    clip_grad_norm: float | None,
    amp: bool = False,
) -> dict[str, float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    final_train = EpochStats(0.0, 0.0, 0.0)
    final_val = EpochStats(0.0, 0.0, 0.0)

    for _ in range(epochs):
        final_train = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            preprocess_fn=preprocess_fn,
            window=window,
            eps=eps,
            optimizer=optimizer,
            clip_grad_norm=clip_grad_norm,
            amp=amp,
        )
        final_val = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            preprocess_fn=preprocess_fn,
            window=window,
            eps=eps,
            optimizer=None,
            clip_grad_norm=None,
            amp=amp,
        )

    return {
        "train_loss": final_train.loss,
        "val_mse": final_val.loss,
        "train_step_time_s": final_train.step_time_s,
        "val_step_time_s": final_val.step_time_s,
        "train_samples_per_s": final_train.samples_per_s,
        "val_samples_per_s": final_val.samples_per_s,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    preprocess_fn: Callable[[torch.Tensor, int, float], torch.Tensor],
    window: int,
    eps: float,
) -> dict[str, float]:
    model.eval()
    criterion = nn.MSELoss()

    losses: list[float] = []
    total_samples = 0
    total_time = 0.0

    for x, y in loader:
        start = time.perf_counter()
        x = x.to(device)
        y = y.to(device)
        x = preprocess_fn(x, window=window, eps=eps)
        pred = model(x)
        loss = criterion(pred, y)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        losses.append(loss.item())
        total_samples += x.shape[0]
        total_time += time.perf_counter() - start

    return {
        "mse": sum(losses) / max(len(losses), 1),
        "samples_per_s": total_samples / max(total_time, 1e-12),
    }
