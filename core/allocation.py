#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""地表水/地下水配比的数值优化。

原本整段长在 `City.water_withdraw` 里（见 issue #27）：参数校验、目标函数闭包
与差分进化调用三件事混在一个 68 行的方法体中，只能通过构造一个完整的模型与
城市智能体才能测到。这里把它抽成不依赖智能体的纯函数，`City.water_withdraw`
只负责把自己的状态翻译成这些参数。
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

#: SciPy 1.15 把 `differential_evolution(seed=...)` 改名成 `rng=...`，旧名已
#: 弃用并计划移除。这里探测当前 SciPy 认哪个。
DE_RNG_KWARG = (
    "rng" if "rng" in inspect.signature(differential_evolution).parameters else "seed"
)

#: 差分进化的默认参数，针对本问题调过。
DE_DEFAULTS: dict[str, Any] = {
    "popsize": 15,  # 种群规模倍数
    "maxiter": 100,  # 最大迭代次数
    "polish": True,  # 末尾用 L-BFGS-B 精修
}


def validate_surface_boundaries(
    total_irrigation: float,
    surface_boundaries: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """校验并补全地表水取值区间。

    Args:
        total_irrigation: 总灌溉量（mm）。
        surface_boundaries: `(下界, 上界)`，None 时取 `(0, total_irrigation)`。

    Returns:
        校验过的 `(下界, 上界)`。

    Raises:
        ValueError: 下界为负，或任一界超过总灌溉量。
    """
    if surface_boundaries is None:
        return 0.0, total_irrigation
    surface_lb, surface_ub = surface_boundaries
    if surface_lb < 0.0 or max(surface_lb, surface_ub) > total_irrigation:
        raise ValueError(f"Invalid boundary values: {surface_boundaries}.")
    return surface_lb, surface_ub


def optimize_surface_share(
    payoff_func: Callable[..., float],
    total_irrigation: float,
    surface_boundaries: Optional[Tuple[float, float]] = None,
    *,
    rng: Optional[np.random.Generator] = None,
    ga_kwargs: Optional[dict[str, Any]] = None,
    **payoff_kwargs: Any,
) -> Tuple[float, float]:
    """用差分进化找出让收益最大的地表/地下配比。

    优化问题：

        max  payoff(q_surface, q_ground, ...)
        s.t. q_surface + q_ground = total_irrigation
             q_surface ∈ [下界, 上界]

    Note:
        差分进化是求**极小**，所以目标函数取 `payoff` 的相反数。随机性默认来自
        传入的 `rng`（模型自己的发生器），不传就退回 SciPy 的全局 RNG——那会
        让同一个种子的两次运行给出不同配比（见 issue #18）。

    Args:
        payoff_func: 收益函数，按关键字接收 `q_surface`、`q_ground` 及
            `payoff_kwargs` 里的其余参数。
        total_irrigation: 总灌溉量（mm）。为 0 时直接返回 `(0.0, 0.0)`，不做优化。
        surface_boundaries: 地表水取值区间，None 时取 `(0, total_irrigation)`。
        rng: 随机数发生器。调用方已在 `ga_kwargs` 里指定 `seed`/`rng` 时忽略。
        ga_kwargs: 覆盖 `DE_DEFAULTS` 的差分进化参数。
        **payoff_kwargs: 透传给 `payoff_func` 的其余参数。

    Returns:
        `(q_surface, q_ground)`，单位 mm。

    Raises:
        ValueError: 区间不合法。
    """
    if total_irrigation == 0.0:
        return 0.0, 0.0
    surface_lb, surface_ub = validate_surface_boundaries(
        total_irrigation, surface_boundaries
    )

    def fitness(q_surface: np.ndarray | float) -> float:
        """差分进化的目标函数（取负号是因为它求极小）。

        Args:
            q_surface: 候选的地表水用量。

        Returns:
            收益的相反数。
        """
        q_surface_val = q_surface[0] if isinstance(q_surface, np.ndarray) else q_surface
        return -payoff_func(
            q_surface=q_surface_val,
            q_ground=total_irrigation - q_surface_val,
            **payoff_kwargs,
        )

    de_params = dict(DE_DEFAULTS)
    de_params.update(ga_kwargs or {})
    if rng is not None and "seed" not in de_params and "rng" not in de_params:
        de_params[DE_RNG_KWARG] = rng

    result = differential_evolution(
        func=fitness,
        bounds=[(surface_lb, surface_ub)],
        **de_params,
    )
    q_surface_opt = float(result.x[0])
    return q_surface_opt, total_irrigation - q_surface_opt
