#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Per-city calibration of both dimensions of Grid-Group cultural theory.

The social term in `payoff.social_standing` takes the source's two symbols
directly: `grid` (`City.s_grid`) and `group` (`City.s_group`). Both were
single national scalars taken from the World Values Survey, shared by all 59
city agents and constant across all simulated years.

This module lets **both** parameters vary by city, each driven by its own
external dataset and its own switch:

    group_i = clip(group_bar + kappa_group * z_i, eps, 1 - eps)
    grid_i  = clip(grid_bar  + kappa_grid  * t_i, eps, 1 - eps)

`z_i` is the prefecture-level Collectivism Index; `t_i` is the province-level
Cultural Tightness index. The two axes have **different native resolutions**
(356 prefectures vs 31 provinces), which is why the Grid side carries a level
switch (`City.s_grid_level`) that the Group side does not need:

* `national`   —— the WVS scalar, shared by all 59 cities. Bit-identical to the
  pre-tightness model, because the branch returns before any file I/O.
* `province`   —— every city gets its own province's `z_province`; cities in the
  same province are identical.
* `prefecture` —— `z_province` plus a per-city draw from the **uncertainty of
  that province's mean**. It invents no data: it propagates the survey's own
  standard error down to the city grain.

Both indices are precomputed and stored under `data/processed/`
(`city_collectivism.csv` and `city_tightness.csv`, built by
`scripts/build_city_collectivism.py` / `scripts/build_city_tightness.py`).
`group_bar` / `grid_bar` are the national scalars and `kappa` the spread.
**`kappa == 0` reproduces the national-scalar model exactly** on either axis,
which is what keeps the golden fingerprint valid; the caller is responsible for
short-circuiting on that case before touching this module.

The prefecture-level draw is made **here, at load time**, from a private
`numpy` generator seeded by `City.s_grid_draw_seed` — never from the model's
own RNG. Two reasons: a sweep over draws must not shift every other random
number in the run (nothing else would stay comparable), and the product on disk
must not have one seed burnt into it (multirun could not sweep it).

Note:
    `payoff.cobb_douglas` guards its argument with `if parameter > 1 or
    parameter < 0: raise`. That guard is **NaN-permissive** — both
    comparisons are `False` for NaN, so a NaN `group` would sail through and
    silently poison `payoff = e * s` with no exception. Every entry point
    here therefore rejects non-finite input explicitly.

Google-style docs are used throughout.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .algorithms import require_finite, require_one_of

#: group 被夹在 [EPS, 1-EPS] 内。开区间是必须的：group=1 会让社会名望在第一次
#: 被批评时直接归零，group=0 则等于把这条通道悄悄关掉。
EPS: float = 1e-3

#: 集体主义产物里 z 列的命名。
Z_COLUMN_TEMPLATE: str = "z_{wave}"

#: Grid 的三档分辨率，`City.s_grid_level` 的取值域。
GRID_LEVELS: tuple[str, ...] = ("national", "province", "prefecture")

#: 逐市抽样用哪一种离散度，`City.s_grid_spread` 的取值域。
#: `se` 是省均值的不确定性（默认），`sd` 是省内受访者之间的差异——后者比前者
#: 大 sqrt(n) ≈ 19 倍，会把省际信号整个淹掉，见
#: `water_quota_analysis.analysis.cultural_tightness.city_spreads`。
GRID_SPREADS: tuple[str, ...] = ("se", "sd")

#: 紧密度产物里的列名。
Z_PROVINCE_COLUMN: str = "z_province"
SPREAD_COLUMN_TEMPLATE: str = "{spread}_z"

#: `City_ID` 列名，与 shapefile 一致。
CITY_ID_COLUMN: str = "City_ID"


def scale_wvs_column(scores: Iterable[float]) -> list[float]:
    """把 WVS 的一列分数 min-max 缩放到 [0, 1] —— **模型标定值的唯一换算**。

    `WVS_results.csv` 的 `Grid score` / `Group score` 两列都不是 [0, 1] 上的量：
    Grid 落在 0.40–0.81，Group 落在 0.46–0.61。而式(3) 的两个基底必须是份额，
    所以要先在 60 国这个样本内 min-max 缩放，再取某一国的值。中国因此是

        grid  = (0.56 − 0.40) / (0.81 − 0.40) = 0.390
        group = (0.53 − 0.46) / (0.61 − 0.46) = 0.467

    这两个数就是 `config.yaml` 里的 `City.s_grid` / `City.s_group`，由
    `tests/test_configs.py` 钉住。

    Note:
        这条换算此前**只存在于 `notebooks/culture_comparison.ipynb`**（一行
        `minmax_scale`），配置里却写着未缩放的 0.56 / 0.53，于是图和模型长期不在
        同一把尺子上。更糟的是 notebook 把 `china` 抽在缩放**之前**，`.iloc[0]`
        是拷贝，所以连它自己画的中国点也用的是原值。把换算收进这里就是为了让
        「配置值从哪来」只有一处可执行的答案（#211）。

    Note:
        min-max 是**单调递增**的，所以不存在"要不要取 1 −"的问题——中国在两列上
        的分位不变（Grid 16.7%、Group 48.3%）。此前的 `WVS_GRID_IS_INVERTED`
        常量问的是一个错的问题，已删除。

    Args:
        scores: 一列 WVS 分数（整列，不能只传一个国家——缩放要用样本的极值）。

    Returns:
        缩放到 [0, 1] 的同长列表。

    Raises:
        ValueError: 列为空、含非有限值，或整列相等（极差为 0，缩放无定义）。

    Example:
        ```python
        scale_wvs_column([0.40, 0.56, 0.81])  # -> [0.0, 0.39..., 1.0]

        # 取某一国：先缩放整列，再按**标签**选行。别反过来"按分数查行"——
        # 分数相同的国家会静默取到第一个，dtype/精度一变还会查不到。
        frame["grid"] = scale_wvs_column(frame["Grid score"])
        china = frame.loc[frame["Country"] == "China", "grid"].iloc[0]
        ```
    """
    values = [float(v) for v in scores]
    if not values:
        raise ValueError("WVS 列为空，无法缩放")
    for value in values:
        require_finite("WVS score", value)
    lo, hi = min(values), max(values)
    if hi == lo:
        raise ValueError(f"WVS 列极差为 0（全为 {lo}），min-max 缩放无定义")
    return [(v - lo) / (hi - lo) for v in values]


def _mapped(base: float, kappa: float, z: float) -> float:
    """两条轴共用的仿射映射，**夹取之前**的值。

    单独抽出来是因为 `clipped_city_ids` 判定的正是它——「什么算被夹取」与「夹取后
    是多少」必须共享同一个表达式，否则两者会各自漂移。
    """
    return base + kappa * z


def _axis_from_index(base: float, kappa: float, z: float, eps: float) -> float:
    """Grid 与 Group 共用的那一条映射：仿射之后夹进 `[eps, 1 - eps]`。

    两条轴同属社会项 `s = grid^m · (1 - group)^n`，夹取约定必须一致——一边夹一边
    不夹，结果照样跑得出来、照样「合理」，只是两轴不再是同一把尺子。所以只写一遍。
    """
    for name, value in (("base", base), ("kappa", kappa), ("z", z)):
        require_finite(name, value)
    return min(max(_mapped(base, kappa, z), eps), 1.0 - eps)


def group_from_index(base: float, kappa: float, z: float, eps: float = EPS) -> float:
    """Per-city Group parameter from a standardised collectivism index.

    Args:
        base: The national scalar `group_bar` (`City.s_group`).
        kappa: Spread. 0 makes the result exactly `base`.
        z: The city's standardised collectivism index.
        eps: Half-open margin keeping the result inside (0, 1).

    Returns:
        The per-city Group parameter, always within `[eps, 1 - eps]`.

    Raises:
        ValueError: If any argument is not finite. NaN must never reach
            `cobb_douglas`, whose range guard does not catch it.

    Example:
        ```python
        group_from_index(base=0.55, kappa=0.0, z=2.4)   # 0.55
        group_from_index(base=0.55, kappa=0.1, z=2.4)   # 0.79
        ```
    """
    return _axis_from_index(base, kappa, z, eps)


@lru_cache(maxsize=None)
def load_city_z(path: str, wave: int) -> Mapping[int, float]:
    """Read the per-city standardised collectivism index, once per run.

    Args:
        path: Path to `city_collectivism.csv`.
        wave: Census wave, one of 2000 / 2010 / 2020.

    Returns:
        Mapping from `City_ID` to `z`.

    Raises:
        FileNotFoundError: If the table is missing.
        KeyError: If the requested wave's column is absent.
        ValueError: If any `z` is not finite, or an id repeats.

    Note:
        `lru_cache` keeps this to one file read per process rather than one
        per agent. The key is `(path, wave)`, so different runs in the same
        process still get their own table.
    """
    table = Path(path)
    if not table.exists():
        raise FileNotFoundError(
            f"集体主义指数表不存在：{table}。"
            "先跑 `python scripts/build_city_collectivism.py` 生成。"
        )
    frame = pd.read_csv(table)
    column = Z_COLUMN_TEMPLATE.format(wave=wave)
    if column not in frame.columns:
        raise KeyError(
            f"{table} 里没有 {column} 列；可用波次：{_available_waves(frame)}"
        )
    if frame[CITY_ID_COLUMN].duplicated().any():
        raise ValueError(f"{table} 里 {CITY_ID_COLUMN} 有重复")
    values = frame.set_index(CITY_ID_COLUMN)[column]
    # 与 `load_city_grid_z` 共用同一道守卫：这是整个 culture 模块的安全前提——
    # `cobb_douglas` / `social_standing` 对 NaN 放行，全靠这里拦。两份实现将来
    # 一份收紧另一份不会跟，而漏过去的 NaN 会静默污染 `payoff = e·s`。
    _require_all_finite(
        table, column, values, pd.Series(values.index, index=values.index)
    )
    return {int(city_id): float(z) for city_id, z in values.items()}


def _available_waves(frame: pd.DataFrame) -> list[int]:
    """从列名里反解出可用的波次，仅用于报错信息。"""
    prefix = Z_COLUMN_TEMPLATE.format(wave="")
    return sorted(
        int(col.removeprefix(prefix))
        for col in frame.columns
        if col.startswith(prefix) and col.removeprefix(prefix).isdigit()
    )


def max_kappa_without_clipping(
    z: Iterable[float], base: float, eps: float = EPS
) -> float:
    """Largest `kappa` at which no city's `group` hits the clip.

    Beyond this value `clip` saturates and the z ordering is destroyed at the
    tails — silently. Any sweep over `kappa` should assert against this.

    Args:
        z: The standardised indices of every city in the run.
        base: The national scalar.
        eps: The same margin passed to `group_from_index`.

    Returns:
        The clip-free ceiling, or `inf` if every `z` is zero.

    Raises:
        ValueError: If `z` is empty.
    """
    values = list(z)
    if not values:
        raise ValueError("z 为空，无法计算 kappa 上限")
    lowest, highest = min(values), max(values)
    limits = []
    if lowest < 0:
        limits.append((base - eps) / -lowest)
    if highest > 0:
        limits.append((1.0 - eps - base) / highest)
    return min(limits) if limits else math.inf


def clipped_city_ids(
    z: Mapping[int, float], base: float, kappa: float, eps: float = EPS
) -> list[int]:
    """Cities whose `group` would be clipped at this `kappa`.

    Args:
        z: Mapping from `City_ID` to standardised index.
        base: The national scalar.
        kappa: Spread.
        eps: The clip margin.

    Returns:
        Sorted `City_ID`s whose `group` `clip` actually moved.

    Note:
        Landing exactly on a bound is not clipping — `clip` is a no-op there.
        The comparison is therefore non-strict, which also makes
        `max_kappa_without_clipping` a consistent ceiling rather than one that
        reports itself as already clipping.
    """
    return sorted(
        city_id
        for city_id, value in z.items()
        if not eps <= _mapped(base, kappa, value) <= 1.0 - eps
    )


def grid_from_tightness(base: float, kappa: float, z: float, eps: float = EPS) -> float:
    """Per-city Grid parameter from a standardised cultural-tightness index.

    Same shape as `group_from_index` on purpose — one mapping, two axes.

    Args:
        base: The national scalar `grid_bar` (`City.s_grid`).
        kappa: Spread, **signed** (`City.s_grid_kappa`). 0 makes the result
            exactly `base`.
        z: The city's standardised tightness index.
        eps: Half-open margin keeping the result inside (0, 1).

    Returns:
        The per-city Grid parameter, always within `[eps, 1 - eps]`.

    Raises:
        ValueError: If any argument is not finite.

    Note:
        **The sign of `kappa` is settled and positive, and it is an identity
        rather than a convention.**

        `payoff.enforcement_share` is the closed form `share = grid`: with
        `v ~ U(0, 1)` and the rule `v > 1 - grid`, `grid` **is** the fraction of
        eligible neighbours that actually sanction a defector. The second half
        of the tightness definition (Chua et al. 2019, after Gelfand) is *"the
        extent to which people are punished or sanctioned when they deviate"* —
        the same quantity, not merely a correlate. Tighter province, larger
        `grid`, hence `kappa > 0`.

        The cost channel gives the same answer independently: filing a report
        burns `1 - grid` of goodwill, and where sanctioning deviance is expected
        the critic carries no stigma, so a tight province has the **low** cost
        and therefore the **high** `grid`.

        This replaces an earlier "instrument reading" that argued for `kappa <
        0` on the grounds that `grid_bar` comes from the WVS `Grid score`
        column, which runs roughly opposite to Gelfand tightness. That argument
        is void: `scale_wvs_column` is a min-max over 60 **countries**, so it
        carries a cross-national relative statement that says nothing within
        China. Once this axis is read as tightness rather than as Douglas's
        Grid, staying on the WVS instrument is not a virtue.

    Note:
        **`base` and `kappa * z` come from different instruments, on purpose.**

        Chua's index is standardised **within** China's 31 provinces, so its
        mean is 0 by construction and it carries no national level at all. The
        level has to come from somewhere else, and `grid_bar = 0.39` (WVS `Grid
        score`, min-max scaled — `scale_wvs_column`) is used as a **proxy** for
        it. What justifies the proxy is not that the two instruments agree —
        they do not, Gelfand et al. (2011) place China on the tight side while
        the WVS column puts it at the 16.7th percentile — but that the model is
        insensitive to the level over the whole plausible range:

        | `grid_bar` | sanctioning rate | deterrent | static compliance |
        | ---------- | ---------------- | --------- | ----------------- |
        | 0.39       | 39%              | 2.84x     | 0.973             |
        | 0.70       | 70%              | 6.50x     | 0.993             |

        (at `group = 0.47`, `n_friends = 6`, `breach = 0.2983`; the economic
        temptation has median 1.11 and p90 1.64, so **both** anchors sit far
        above it and the social term is saturated either way.)

        So the division of labour is: **WVS supplies the level, which barely
        matters; Chua supplies the ordering, which is what carries the result.**
        Every spatial conclusion rests on the ordering.

        The honest caveat: saturation is a statement about the static judgement
        in a single year. The simulated model is more sensitive than that table
        suggests, because a small per-year edge compounds through 30 years of
        imitation — moving one province from 0.39 to 0.52 shifted its breach
        rate by 7.3 pp. Treat the anchor as a sensitivity parameter, not as a
        measured quantity.

    Example:
        ```python
        grid_from_tightness(base=0.39, kappa=0.0, z=1.5)    # 0.39
        grid_from_tightness(base=0.39, kappa=0.1, z=1.5)    # 0.54
        grid_from_tightness(base=0.39, kappa=-0.1, z=1.5)   # 0.24
        ```
    """
    return _axis_from_index(base, kappa, z, eps)


@lru_cache(maxsize=None)
def load_city_grid_z(
    path: str, level: str, spread: str = "se", seed: int = 0
) -> Mapping[int, float]:
    """Read the per-city tightness index at the requested resolution.

    Args:
        path: Path to `city_tightness.csv`.
        level: `province` or `prefecture`. `national` never gets here — the
            caller short-circuits before any file I/O.
        spread: Which dispersion the prefecture draw uses, one of
            `GRID_SPREADS`. Ignored at `province` level.
        seed: Seed of the private generator used for the prefecture draw.
            Ignored at `province` level.

    Returns:
        Mapping from `City_ID` to the standardised tightness index.

    Raises:
        FileNotFoundError: If the table is missing.
        ValueError: If `level` / `spread` is unknown, an id repeats, or any
            value is not finite.
        KeyError: If a required column is absent.

    Note:
        The draw is `z_province + spread_z * N(0, 1)`, one standard normal per
        city taken in ascending `City_ID` order from a `default_rng(seed)`.
        Two consequences worth knowing before reading results:

        * It is reproducible from `(path, level, spread, seed)` alone and
          touches no other random number in the run.
        * It is **not** re-centred within province. That is deliberate: the
          spread being propagated is the uncertainty of the province's mean, so
          a draw that moves the whole province is a legitimate outcome, not a
          bug. Re-centring would silently turn it into within-province noise.

    Note:
        `lru_cache` keeps this to one file read per process. The key covers
        every argument, so two runs differing only in `seed` still get their
        own tables.
    """
    require_one_of("City.s_grid_level", level, GRID_LEVELS)
    require_one_of("City.s_grid_spread", spread, GRID_SPREADS)
    table = Path(path)
    if not table.exists():
        raise FileNotFoundError(
            f"文化紧密度表不存在：{table}。"
            "先跑 `python scripts/build_city_tightness.py` 生成。"
        )
    frame = pd.read_csv(table).sort_values(CITY_ID_COLUMN, ignore_index=True)
    spread_column = SPREAD_COLUMN_TEMPLATE.format(spread=spread)
    needed = [CITY_ID_COLUMN, Z_PROVINCE_COLUMN] + (
        [spread_column] if level == "prefecture" else []
    )
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise KeyError(f"{table} 里没有这些列：{missing}")
    if frame[CITY_ID_COLUMN].duplicated().any():
        raise ValueError(f"{table} 里 {CITY_ID_COLUMN} 有重复")

    values = frame[Z_PROVINCE_COLUMN].astype(float)
    if level == "prefecture":
        sigma = frame[spread_column].astype(float)
        _require_all_finite(table, spread_column, sigma, frame[CITY_ID_COLUMN])
        if (sigma < 0).any():
            raise ValueError(f"{table} 的 {spread_column} 列有负值，标准差无定义")
        draws = np.random.default_rng(seed).standard_normal(len(frame))
        values = values + sigma * draws
    _require_all_finite(table, Z_PROVINCE_COLUMN, values, frame[CITY_ID_COLUMN])
    return {
        int(city_id): float(value)
        for city_id, value in zip(frame[CITY_ID_COLUMN], values)
    }


def _require_all_finite(
    table: Path, column: str, values: pd.Series, ids: pd.Series
) -> None:
    """Reject non-finite values, naming the cities they belong to.

    Args:
        table: The file the values came from, for the message.
        column: Column name, for the message.
        values: The values to check.
        ids: `City_ID`s aligned with `values`.

    Raises:
        ValueError: If any value is not finite. NaN must never reach
            `cobb_douglas`, whose range guard does not catch it.
    """
    bad = ~np.isfinite(values.to_numpy())
    if bad.any():
        offenders = dict(zip(ids[bad], values[bad]))
        raise ValueError(f"{table} 的 {column} 列有非有限值：{offenders}")
