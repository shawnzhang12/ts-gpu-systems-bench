from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
import mlflow
import optuna
from omegaconf import DictConfig, OmegaConf

from src.data.etth1 import download_etth1, load_etth1, split_etth1
from src.train.runner import train_once

MODEL_CONFIGS = {
    "mamba": {
        "type": "mamba",
        "d_model": 64,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "n_layers": 2,
    },
    "transformer": {
        "type": "transformer",
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.0,
    },
}


def build_trial_cfg(
    base_cfg: DictConfig,
    lookback: int,
    backend: str,
    model_type: str,
    batch_size: int,
) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.data.lookback = int(lookback)
    cfg.data.batch_size = int(batch_size)
    cfg.preprocess.backend = str(backend)
    cfg.preprocess.window = int(lookback)
    cfg.model = OmegaConf.create(MODEL_CONFIGS[model_type])
    return cfg


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(str(cfg.logging.tracking_uri))
    mlflow.set_experiment(str(cfg.experiment_name))

    if cfg.data.download_if_missing:
        download_etth1(cfg.data.path)
    frame = load_etth1(cfg.data.path)
    split = split_etth1(frame)
    max_lookback = min(len(split.train), len(split.val), len(split.test)) - int(cfg.data.horizon)
    requested_lookbacks = [int(v) for v in cfg.sweep.lookbacks]
    valid_lookbacks = [lb for lb in requested_lookbacks if lb <= max_lookback]
    dropped_lookbacks = [lb for lb in requested_lookbacks if lb > max_lookback]
    if not valid_lookbacks:
        raise ValueError(
            f"no valid lookbacks in sweep; requested={requested_lookbacks}, max_valid={max_lookback}"
        )

    search_space = {
        "lookback": valid_lookbacks,
        "backend": [str(v) for v in cfg.sweep.backends],
        "model": [str(v) for v in cfg.sweep.models],
        "batch_size": [int(v) for v in cfg.sweep.batch_sizes],
    }

    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    with mlflow.start_run(run_name=str(cfg.logging.run_name) if cfg.logging.run_name else "sweep") as parent_run:
        mlflow.log_dict(search_space, "sweep_space.json")
        mlflow.log_param("max_valid_lookback", int(max_lookback))
        if dropped_lookbacks:
            mlflow.log_dict({"dropped_lookbacks": dropped_lookbacks}, "dropped_lookbacks.json")

        def objective(trial: optuna.Trial) -> float:
            lookback = trial.suggest_categorical("lookback", search_space["lookback"])
            backend = trial.suggest_categorical("backend", search_space["backend"])
            model_type = trial.suggest_categorical("model", search_space["model"])
            batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])

            trial_cfg = build_trial_cfg(
                base_cfg=cfg,
                lookback=lookback,
                backend=backend,
                model_type=model_type,
                batch_size=batch_size,
            )

            run_name = f"{backend}_{model_type}_L{lookback}_B{batch_size}"
            trial.set_user_attr("backend", backend)
            trial.set_user_attr("model", model_type)
            trial.set_user_attr("lookback", lookback)
            trial.set_user_attr("batch_size", batch_size)

            try:
                with mlflow.start_run(run_name=run_name, nested=True):
                    metrics = train_once(trial_cfg, nested=True, run_name=run_name)
            except Exception as exc:
                trial.set_user_attr("fit_in_vram", 0.0)
                trial.set_user_attr("val_mse", math.inf)
                trial.set_user_attr("error", str(exc))
                return 1.0e12

            fit = float(metrics.get("fit_in_vram", 0.0))
            val_mse = float(metrics.get("val_mse", math.inf))
            objective_value = val_mse if fit > 0.5 and math.isfinite(val_mse) else 1.0e12

            trial.set_user_attr("fit_in_vram", fit)
            trial.set_user_attr("val_mse", val_mse)

            return objective_value

        study.optimize(objective, n_trials=int(cfg.sweep.max_trials))

        max_fit: dict[str, int] = {}
        for t in study.trials:
            backend = str(t.user_attrs.get("backend"))
            model = str(t.user_attrs.get("model"))
            fit = float(t.user_attrs.get("fit_in_vram", 0.0))
            lookback = int(t.user_attrs.get("lookback", 0))
            key = f"{backend}:{model}"
            if fit > 0.5:
                max_fit[key] = max(lookback, max_fit.get(key, 0))
            else:
                max_fit.setdefault(key, 0)

        best = {
            "value": study.best_value,
            "params": study.best_params,
            "trial": study.best_trial.number,
            "max_fit_lookback": max_fit,
            "parent_run_id": parent_run.info.run_id,
        }

        out = Path("results") / "sweep_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(best, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out))

        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
