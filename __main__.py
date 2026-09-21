#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import sys

# —— 这段必须在任何会拉起 aquacrop 的 import 之前跑（见 #103）——
#
# `aquacrop/__init__.py` 开头是 `if not '-m' in sys.argv:` 才导出自己的符号，
# 而那是对整个 argv 的**逐元素**匹配。Hydra 的 multirun 短选项恰好就是 `-m`，
# 于是 `python -m cwatqim -m scenario=a,b` 里用户那个 `-m` 被 aquacrop 当成
# "我是被 python -m aquacrop 调起来的"，直接跳过全部导出，最后崩在一个与
# 命令行毫不相干的 `ImportError: cannot import name 'Crop'`。
# （`--multirun` 不会触发：它不等于 `-m`，粘在一起的 `-mn` 之类也不会。）
#
# `__name__` 那个条件是必需的：本模块会被 `tests/model/test_main.py` 直接
# import，而 `pytest -m "not slow"` 这类命令的 argv 里同样有裸 `-m`——少了它，
# 光是 import 就会 SystemExit。只有真的 `python -m cwatqim` 时才拦。
#
# `argv[0]` 不必排除：runpy 在 `__main__.py` 执行前就把它换成文件路径了
# （`tests/model/test_main.py` 里记着这件事）。切掉 `[0]` 只是稳妥，不是必需。
if __name__ == "__main__" and "-m" in sys.argv[1:]:
    raise SystemExit(
        "cwatqim: 请把 `-m` 换成 `--multirun`。\n\n"
        "  python -m cwatqim --multirun scenario=baseline,strict ...\n\n"
        "aquacrop 会把命令行里任何位置的裸 `-m` 当作自己被 `python -m aquacrop` "
        "调起，从而跳过全部符号导出；继续跑下去只会得到一个看不出原因的 "
        "`ImportError: cannot import name 'Crop'`（见 issue #103）。"
    )

import hydra  # noqa: E402
from abses import Experiment  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from cwatqim.model.main import CWatQIModel  # noqa: E402


class SeededExperiment(Experiment):
    """`Experiment`，但把 hydra 的 job id 当成整数用。

    ABSESpy 的 `_get_seed` 算的是 `base_seed + job_id * 1000 + run_id`，而
    `Experiment.job_id` 在 hydra 环境下返回的是 `hydra.job.id` —— Hydra 把它
    声明成 `id: str`（`hydra/conf/__init__.py`，对照 `num: int`），所以那是个
    **字符串**。于是 `int + str` 直接 TypeError，multirun 一跑就崩。

    这个组合以前碰不到：产出论文结果的那次运行，`exp` 配置里根本没有 `seed`
    键，`_get_seed` 在 `if self._base_seed is None` 就返回了。后来种子被加进
    两个 exp profile，"multirun + 有种子"第一次相遇就炸（见 issue #82）。

    Note:
        本该在上游修，但 `abses` 被钉死在 0.11.7、理由正是可复现性，返修期不
        动那颗钉子。上游修好后删掉这个子类即可，是纯减法。
    """

    @property
    def job_id(self) -> int:
        """Hydra 的 job id，强制成整数。

        Returns:
            当前 job 的整数编号；非 hydra 环境下沿用基类的计数器。
        """
        return int(super().job_id)


@hydra.main(version_base=None, config_path="config", config_name="demo")
def run_abm(cfg: DictConfig | None = None) -> None:
    """Run batch experiments for the water quota model.

    This function serves as the main entry point for running batch simulations
    of the CWatQIM model. It uses Hydra for configuration management and
    supports parallel execution of multiple simulation runs.

    The function will:
        1. Load configuration from `config/demo.yaml` (relative to cwatqim package root)
        2. Create an Experiment instance with the CWatQIModel
        3. Run multiple simulation repeats (can be parallelized)
        4. Save summary statistics to CSV

    Note:
        This model should be run from the cwatqim package root directory where the
        `config/` folder is located. The default configuration uses sample data
        from `data/sample/` directory.

    Args:
        cfg: Optional Hydra configuration dictionary. If None, Hydra will
            automatically load from the default config file. The configuration
            should include:
            - `exp.repeats`: Number of simulation repeats (default: 1)
            - `exp.num_process`: Number of parallel processes (default: 1)
            - `exp.seed`: Base random seed (default: None). When set, each
              repeat gets a seed derived from it, so the whole batch is
              reproducible; when None, every run is seeded from entropy.
            - Model parameters and data paths

    Example:
        Run from command line (from cwatqim directory):
        ```bash
        python -m cwatqim
        ```

        Or with custom parameters:
        ```bash
        python -m cwatqim exp.repeats=10 exp.num_process=4
        python -m cwatqim config_name=demo time.start=1985 time.end=1990
        ```

        Multi-run sweeps must use the **long** flag `--multirun`; the short
        `-m` makes aquacrop skip its exports and the run dies at import time
        with a misleading `ImportError` (see issue #103):
        ```bash
        python -m cwatqim --multirun scenario=baseline,never,strict
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

    exp = SeededExperiment(CWatQIModel, cfg=cfg, seed=cfg.exp.get("seed", None))
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
