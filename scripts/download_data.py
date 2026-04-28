from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.data.etth1 import download_etth1


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    path = download_etth1(cfg.data.path, force=False)
    print(f"downloaded_or_cached={path}")


if __name__ == "__main__":
    main()
