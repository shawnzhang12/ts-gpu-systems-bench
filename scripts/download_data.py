from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.data.benchmarks import download_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_name = str(getattr(cfg.data, "dataset", cfg.data.name))
    path = download_dataset(dataset_name, cfg.data.path, force=False)
    print(f"downloaded_or_cached={path}")


if __name__ == "__main__":
    main()
