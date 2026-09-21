#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""地表水/地下水配比的求解。

原本整段长在 `City.water_withdraw` 里（见 issue #27）：参数校验、目标函数闭包与
差分进化调用三件事混在一个 68 行的方法体中，只能构造一个完整的模型与城市智能体
才能测到。这里把它抽成不依赖智能体的纯函数。

**为什么不再用差分进化（issue #94）**：目标函数对 `q_surface` 是**分段凸**的。
作物收入与配比无关（`crop_yield` 在进求解之前就绑死），水费是线性的，社会项只在
配额处跳一次。凸函数在闭区间上的极大值必在**端点**，所以正确的解法是枚举端点，
而不是让全域随机搜索去逼近它——后者给本应精确的解注入了收敛残差（实测非角点解
到最近角点的相对距离中位数 0.0009），还平白消耗模型的随机数流。

速度不是主要理由，别照抄成"快了很多"：实测目标函数求值少了 26×，但整模型墙钟只
快 1.03×——耗时由 AquaCrop 占着。

**为什么检的是凸性而不是仿射性**：端点枚举要的只是"极大值落在端点"，而凸性正是
它的充分条件；仿射是凸的特例，比需要的更强。真正危险的是**凹**——凹函数的极大值
可以落在内点，端点枚举会静默给出错的解。所以守卫是**单边**的：凹（中点高于两端
平均）抛错，凸（中点低于平均）放行。

这条区分不是学究：曾经的 `City.payoff_floor` 让效用变成 `max(e, floor)·s`，在下限那一侧
是平的、另一侧是仿射的——**凸的折线**，端点枚举照旧精确，但双边的仿射检验会把它
误判成非线性（2026-08-31 实测被拦下）。

分段性仍是**前提**而不是巧合，所以 `solve_surface_share` 默认会验证它：谁将来给
目标函数加了凹的非线性项（例如随取水量递减的边际收益），这里会立刻抛错，而不是
静默返回一个错的角点。
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple

#: 凸性检验的容差。`economic_payoff` 把结果 `round(..., 2)`，所以绝对项不能小于
#: 半个分；相对项按收益量级（约 1e8 元）给出约 0.1 元的余量。真正的非线性项会比这
#: 大好几个数量级。
CONVEXITY_ATOL: float = 0.05
CONVEXITY_RTOL: float = 1e-9


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


def _kink_inside(lower: float, upper: float, kink: Optional[float]) -> bool:
    """拐点是否真的落在可行域**内部**。

    只有这时它才既是分段的切点、又是一个候选解。两个用途此前各写一遍同样的
    判断（issue #136），改一处漏一处就会让分段与候选集对不上。

    Args:
        lower: 下界。
        upper: 上界。
        kink: 拐点位置，None 表示目标在整段上仿射。

    Returns:
        落在开区间 `(lower, upper)` 内时为 True。
    """
    return kink is not None and lower < kink < upper


def _pieces(
    lower: float, upper: float, kink: Optional[float]
) -> list[Tuple[float, float]]:
    """把可行域按拐点切成若干段。

    Args:
        lower: 下界。
        upper: 上界。
        kink: 拐点位置，None 或落在区间外时不切。

    Returns:
        每段的 `(左端, 右端)`。
    """
    if not _kink_inside(lower, upper, kink):
        return [(lower, upper)]
    return [(lower, kink), (kink, upper)]


def assert_convex_on_pieces(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    kink: Optional[float] = None,
) -> None:
    """验证目标函数在每一段上都**不是凹的** —— 端点枚举成立的充分条件。

    在段的四分位处取三点，比较中点与两侧的平均：

        凹（mid > 平均）-> 极大值可能落在内点 -> **抛错**
        凸（mid < 平均）-> 极大值必在端点     -> 放行
        仿射（相等）    -> 两者的边界情形     -> 放行

    只在**段内部**取点，因此对左闭右开的那一段同样成立——拐点本身属于左边那段
    （`City.decide` 用严格大于判定违规），在端点取值会混两支。

    Args:
        objective: 单变量目标函数。
        lower: 下界。
        upper: 上界。
        kink: 拐点位置，None 表示整段只有一段。

    Raises:
        ValueError: 某一段上中点**高于**两侧取值的平均超过容差，即该段是凹的。
            这说明极大值可能落在内点，端点枚举给出的解不再可信。

    Note:
        2026-08-31 之前这里检的是**仿射性**（双边）。改成单边是因为
        曾经的 `City.payoff_floor` 把效用变成 `max(e, floor)·s`：在下限那一侧是平的，
        另一侧仿射，合起来是**凸**的折线。端点枚举对它照旧精确，而双边检验会把它
        误判成非线性并抛错。放宽到凸性没有削弱保护——危险的是凹，那一侧仍然拦。
    """
    for left, right in _pieces(lower, upper, kink):
        if right <= left:
            continue
        span = right - left
        low = objective(left + 0.25 * span)
        mid = objective(left + 0.50 * span)
        high = objective(left + 0.75 * span)
        expected = (low + high) / 2
        tolerance = CONVEXITY_ATOL + CONVEXITY_RTOL * max(abs(low), abs(high))
        if mid - expected > tolerance:
            raise ValueError(
                f"目标函数在 [{left}, {right}] 上是凹的："
                f"中点 {mid} 高于两端平均 {expected}（容差 {tolerance}）。"
                "凹函数的极大值可能落在内点，端点枚举不再成立，见 issue #94。"
            )


def solve_surface_share(
    payoff_func: Callable[..., float],
    total_irrigation: float,
    surface_boundaries: Optional[Tuple[float, float]] = None,
    *,
    kink: Optional[float] = None,
    **payoff_kwargs,
) -> Tuple[float, float]:
    """枚举端点，求出让效用最大的地表/地下配比。

    优化问题：

        max  payoff(q_surface, q_ground, ...)
        s.t. q_surface + q_ground = total_irrigation
             q_surface ∈ [下界, 上界]

    目标分段仿射（见模块 docstring），所以候选解只有 `{下界, 拐点, 上界}` 三个。
    结果是**精确**且确定的：不消耗随机数，同一输入永远给同一输出。

    Args:
        payoff_func: 收益函数，按关键字接收 `q_surface`、`q_ground` 及
            `payoff_kwargs` 里的其余参数。
        total_irrigation: 总灌溉量（mm）。为 0 时直接返回 `(0.0, 0.0)`。
        surface_boundaries: 地表水取值区间，None 时取 `(0, total_irrigation)`。
        kink: 目标函数的拐点（模型里是配额）。落在区间内时它是第三个候选解，
            也是"守约角点"。None 表示目标在整个区间上仿射。
        **payoff_kwargs: 透传给 `payoff_func` 的其余参数。

    Returns:
        `(q_surface, q_ground)`，单位与 `total_irrigation` 相同。

    Raises:
        ValueError: 区间不合法，或目标函数不是分段仿射的。

    Note:
        取值相同时选**较小**的 `q_surface`。这既让结果确定，又与 `City.decide`
        的严格大于判定一致：打平时算守约。

    Example:
        ```python
        # 地表水更便宜，所以最优解就是上界
        solve_surface_share(
            lambda q_surface, q_ground: -(q_surface * 0.2 + q_ground * 0.68),
            total_irrigation=10.0,
        )
        # -> (10.0, 0.0)
        ```

    See Also:
        - `cwatqim.agents.city.City.water_withdraw`: 唯一的调用点
        - `cwatqim.core.payoff.aggregate_utility`: 被最大化的量
    """
    # 旧的差分进化调参关键字会被 `**payoff_kwargs` 静默吞掉再原样转给目标函数——
    # 调用方以为在调参，实际什么都没发生。摘掉并出声（见 issue #130）。
    payoff_kwargs = _warn_legacy_de_kwargs(payoff_kwargs)
    if total_irrigation == 0.0:
        return 0.0, 0.0
    surface_lb, surface_ub = validate_surface_boundaries(
        total_irrigation, surface_boundaries
    )

    def objective(q_surface: float) -> float:
        """把候选配比翻成效用。"""
        return payoff_func(
            q_surface=q_surface,
            q_ground=total_irrigation - q_surface,
            **payoff_kwargs,
        )

    assert_convex_on_pieces(objective, surface_lb, surface_ub, kink)

    candidates = [surface_lb, surface_ub]
    if _kink_inside(surface_lb, surface_ub, kink):
        candidates.append(kink)
    # 升序 + 严格大于 ⇒ 打平时留住较小的 q_surface。从最小的那个起步，
    # 剩下的才进循环——否则它的目标函数会被多求一次值（#136）。
    ordered = sorted(candidates)
    best, best_value = ordered[0], objective(ordered[0])
    for candidate in ordered[1:]:
        value = objective(candidate)
        if value > best_value:
            best, best_value = candidate, value
    return best, total_irrigation - best


#: 旧调用方可能仍在传的调参关键字。`ga_kwargs` 是 `City.water_withdraw` 在
#: **v0.1.6 及更早**真正接受过的参数，而 `solve_surface_share` 的 `**payoff_kwargs`
#: 会把它静默转给目标函数——调用方以为在调参，实际什么都没发生。这是必须出声的一格。
#:
#: 只列这一个：差分进化时代的其它名字（`de_kwargs` / `rng` / `seed` / `popsize` /
#: `maxiter`）从未随包发布过，把它们也摘掉反而会吞掉将来合法的同名关键字。
_LEGACY_DE_KWARGS = ("ga_kwargs",)


def _warn_legacy_de_kwargs(payoff_kwargs: dict) -> dict:
    """摘掉旧的调参关键字并发告警。

    Args:
        payoff_kwargs: 传给目标函数的关键字，可能混进旧的调参键。

    Returns:
        去掉旧调参键之后的副本；目标函数拿到的仍然只有它自己的参数。
    """
    stale = [key for key in _LEGACY_DE_KWARGS if key in payoff_kwargs]
    if not stale:
        return payoff_kwargs
    warnings.warn(
        f"{stale} 是差分进化时代的调参关键字，`solve_surface_share` 已改成枚举"
        "端点的 closed form，它们**不再有任何作用**。此处已忽略。以前它们会被 "
        "`**payoff_kwargs` 原样转给目标函数，既不报错也无提示（见 issue #130）。",
        DeprecationWarning,
        stacklevel=3,
    )
    return {k: v for k, v in payoff_kwargs.items() if k not in _LEGACY_DE_KWARGS}
