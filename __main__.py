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
    """批量运行一次实验。

    使用 `config/config.yaml` 配置文件。
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
