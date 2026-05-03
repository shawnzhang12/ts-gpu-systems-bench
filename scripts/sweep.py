from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
import mlflow
import optuna
from omegaconf import DictConfig, OmegaConf
from tabulate import tabulate

from src.data.benchmarks import download_dataset, expand_dataset, load_dataset, split_dataset
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
    "mamba_large": {
        "type": "mamba",
        "d_model": 256,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        "n_layers": 8,
    },
    "transformer_large": {
        "type": "transformer",
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 6,
        "dropout": 0.1,
    },
}

TABLE_COLUMNS = [
    ("backend", "backend", "text"),
    ("model", "model", "text"),
    ("lookback", "L", "int"),
    ("batch_size", "B", "int"),
    ("fit_in_vram", "fit", "float"),
    ("val_mse", "val_mse", "float"),
    ("test_mse", "test_mse", "float"),
    ("pre_latency_mean_ms", "pre_ms", "float"),
    ("pre_eff_bw_gbps", "pre_bw_gbps", "float"),
    ("pre_peak_mem_mb", "pre_mem_mb", "float"),
    ("infer_latency_p50_ms", "infer_p50_ms", "float"),
    ("infer_latency_p95_ms", "infer_p95_ms", "float"),
    ("cold_start_ms", "cold_start_ms", "float"),
    ("train_samples_per_s", "train_sps", "float"),
    ("test_samples_per_s", "test_sps", "float"),
    ("train_tokens_per_s", "train_tok_s", "float"),
    ("infer_tokens_per_s", "infer_tok_s", "float"),
    ("effective_batch_size", "eff_B", "float"),
    ("fit_retries", "retries", "float"),
    ("peak_memory_mb", "peak_mem_mb", "float"),
]

METRIC_DIRECTIONS = {
    "val_mse": "min",
    "test_mse": "min",
    "pre_latency_mean_ms": "min",
    "pre_eff_bw_gbps": "max",
    "pre_peak_mem_mb": "min",
    "infer_latency_p50_ms": "min",
    "infer_latency_p95_ms": "min",
    "cold_start_ms": "min",
    "train_samples_per_s": "max",
    "test_samples_per_s": "max",
    "train_tokens_per_s": "max",
    "infer_tokens_per_s": "max",
    "effective_batch_size": "max",
    "fit_retries": "min",
    "peak_memory_mb": "min",
}


def build_trial_cfg(
    base_cfg: DictConfig,
    lookback: int,
    backend: str,
    model_type: str,
    batch_size: int,
) -> DictConfig:
    if model_type not in MODEL_CONFIGS:
        known = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise ValueError(f"unknown model preset '{model_type}', expected one of: {known}")

    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.data.lookback = int(lookback)
    cfg.data.batch_size = int(batch_size)
    cfg.preprocess.backend = str(backend)
    cfg.preprocess.window = int(lookback)
    cfg.model = OmegaConf.create(MODEL_CONFIGS[model_type])
    return cfg


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _is_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _fmt_metric(value, kind: str) -> str:
    if kind == "text":
        return str(value) if value not in (None, "") else "NA"
    if not _is_finite(value):
        return "NA"
    if kind == "int":
        return f"{int(value)}"
    if kind == "float":
        return f"{float(value):.4f}"
    return str(value)


def _select_best_values(rows: list[dict]) -> dict[str, float]:
    finite_rows = [r for r in rows if _safe_float(r.get("fit_in_vram", 0.0)) > 0.5]
    out: dict[str, float] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        vals = [_safe_float(r.get(metric, math.nan)) for r in finite_rows]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        out[metric] = min(vals) if direction == "min" else max(vals)
    return out


def _is_best(metric: str, value, best_values: dict[str, float]) -> bool:
    if metric not in best_values or not _is_finite(value):
        return False
    best = best_values[metric]
    val = float(value)
    return abs(val - best) <= max(1e-8, 1e-5 * max(abs(best), 1.0))


def _trial_rows(study: optuna.Study) -> list[dict]:
    rows: list[dict] = []
    metric_keys = {c[0] for c in TABLE_COLUMNS}

    for trial in study.trials:
        row: dict[str, object] = {
            "backend": str(trial.user_attrs.get("backend", "")),
            "model": str(trial.user_attrs.get("model", "")),
            "lookback": int(trial.user_attrs.get("lookback", 0)),
            "batch_size": int(trial.user_attrs.get("batch_size", 0)),
        }
        for key in metric_keys:
            if key in row:
                continue
            row[key] = trial.user_attrs.get(key, math.nan)
        rows.append(row)

    rows.sort(
        key=lambda r: (
            -_safe_float(r.get("fit_in_vram", 0.0)),
            _safe_float(r.get("val_mse", math.inf)),
            -_safe_float(r.get("pre_eff_bw_gbps", -math.inf)),
            _safe_float(r.get("infer_latency_p50_ms", math.inf)),
        )
    )
    return rows


def _render_table(rows: list[dict]) -> str:
    best_values = _select_best_values(rows)
    headers = [col[1] for col in TABLE_COLUMNS]
    rendered_rows: list[list[str]] = []

    for row in rows:
        rendered: list[str] = []
        for key, _, kind in TABLE_COLUMNS:
            val = row.get(key, math.nan)
            cell = _fmt_metric(val, kind)
            if _is_best(key, val, best_values):
                cell = f"**{cell}**"
            rendered.append(cell)
        rendered_rows.append(rendered)

    return tabulate(rendered_rows, headers=headers, tablefmt="github", disable_numparse=True)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(str(cfg.logging.tracking_uri))
    mlflow.set_experiment(str(cfg.experiment_name))
    dataset_name = str(getattr(cfg.data, "dataset", cfg.data.name))

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
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("max_valid_lookback", int(max_lookback))
        mlflow.log_metric("dataset_total_rows", float(len(frame)))
        mlflow.log_metric("dataset_train_rows", float(len(split.train)))
        mlflow.log_metric("dataset_val_rows", float(len(split.val)))
        mlflow.log_metric("dataset_test_rows", float(len(split.test)))
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
                trial.set_user_attr("val_mse", None)
                trial.set_user_attr("error", str(exc))
                return 1.0e12

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    val = float(value)
                    trial.set_user_attr(key, val if math.isfinite(val) else None)

            fit = float(metrics.get("fit_in_vram", 0.0))
            val_mse = float(metrics.get("val_mse", math.inf))
            objective_value = val_mse if fit > 0.5 and math.isfinite(val_mse) else 1.0e12

            trial.set_user_attr("fit_in_vram", fit)
            trial.set_user_attr("val_mse", val_mse if math.isfinite(val_mse) else None)

            return objective_value

        study.optimize(objective, n_trials=int(cfg.sweep.max_trials))

        max_fit: dict[str, int] = {}
        for trial in study.trials:
            backend = str(trial.user_attrs.get("backend"))
            model = str(trial.user_attrs.get("model"))
            fit = _safe_float(trial.user_attrs.get("fit_in_vram", 0.0))
            lookback = int(trial.user_attrs.get("lookback", 0))
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

        table_rows = _trial_rows(study)
        table_md = _render_table(table_rows)
        table_path = Path("results") / f"sweep_{parent_run.info.run_id}_table.md"
        table_path.write_text(table_md + "\n", encoding="utf-8")
        mlflow.log_artifact(str(table_path))

        fit_rows = [r for r in table_rows if _safe_float(r.get("fit_in_vram", 0.0)) > 0.5]
        commentary_lines = [
            f"- Trials completed: {len(table_rows)}",
            f"- Successful VRAM-fit trials: {len(fit_rows)}",
            "- Table now includes preprocessing latency/bandwidth/memory plus inference p50/p95 and cold-start.",
            "- Bold cells mark best values among successful trials.",
        ]
        commentary_path = Path("results") / f"sweep_{parent_run.info.run_id}_commentary.md"
        commentary_path.write_text("\n".join(commentary_lines) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(commentary_path))

        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
