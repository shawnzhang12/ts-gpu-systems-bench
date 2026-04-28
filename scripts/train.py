from __future__ import annotations

import json

import hydra
import mlflow
from omegaconf import DictConfig

from src.train.runner import train_once


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(str(cfg.logging.tracking_uri))
    mlflow.set_experiment(str(cfg.experiment_name))

    with mlflow.start_run(run_name=str(cfg.logging.run_name) if cfg.logging.run_name else None):
        metrics = train_once(cfg, nested=False, run_name="train_latest")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
