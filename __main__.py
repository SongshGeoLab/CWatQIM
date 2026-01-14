#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/


import hydra
from abses import Experiment
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cwatqim.model.main import CWatQIModel


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run_abm(cfg: DictConfig | None = None) -> None:
    """Run batch experiments for the water quota model.

    This function serves as the main entry point for running batch simulations
    of the CWatQIM model. It uses Hydra for configuration management and
    supports parallel execution of multiple simulation runs.

    The function will:
        1. Load configuration from `config/config.yaml` (relative to repository root)
        2. Create an Experiment instance with the CWatQIModel
        3. Run multiple simulation repeats (can be parallelized)
        4. Save summary statistics to CSV

    Note:
        This model should be run from the repository root directory where the
        `config/` folder is located. Configuration files are not included in
        the PyPI package - users should clone the repository from GitHub to
        access the full configuration.

    Args:
        cfg: Optional Hydra configuration dictionary. If None, Hydra will
            automatically load from the default config file. The configuration
            should include:
            - `exp.repeats`: Number of simulation repeats (default: 1)
            - `exp.num_process`: Number of parallel processes (default: 1)
            - Model parameters and data paths

    Example:
        Run from command line:
        ```bash
        python -m cwatqim
        ```

        Or with custom config:
        ```bash
        python -m cwatqim exp.repeats=10 exp.num_process=4
        ```

    Note:
        The function disables OmegaConf struct mode to allow the Experiment
        class to pass additional parameters dynamically.

    See Also:
        - `cwatqim.model.main.CWatQIModel`: The main model class
        - `abses.Experiment`: The experiment runner class
    """
    # Disable struct mode to allow Experiment to pass additional parameters
    OmegaConf.set_struct(cfg, False)

    exp = Experiment(CWatQIModel, cfg=cfg)
    exp.batch_run(
        repeats=cfg.exp.get("repeats", 1),
        parallels=cfg.exp.get("num_process", 1),
    )

    # Save summary to experiment folder
    summary_path = exp.folder / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    exp.summary().to_csv(summary_path)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    run_abm()
