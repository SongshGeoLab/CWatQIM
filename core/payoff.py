#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Functions for calculating economic and social payoffs.

This module provides functions for calculating various components of agent
payoffs, including:
    - Economic benefits from crop production
    - Water costs
    - Social standing retained under peer criticism
    - Combined economic and social payoffs
    - The peer-enforcement rule (`reports_defector`) and the population share
      it implies (`enforcement_share`)

These functions are used by City agents to evaluate different water use
strategies and make optimal decisions. The last pair lives here rather than
in the agent because `water_quota_analysis` draws the enforcement curve from
the very same definition -- two copies of a decision rule drift silently
(issue #129).

Note:
    The social term is a **multiplier on the payoff**, not a cost to
    subtract: 1.0 is the untouched case and 0.0 the fully eroded one. It was
    named and documented the other way round until issue #60.
"""

import warnings
from typing import Any, Optional, Tuple

import pandas as pd

from .algorithms import DictLikeType, require_finite, require_unit_interval, squeeze
from .data_loaders import WaterUnitType, convert_mm_to_m3


def cobb_douglas(parameter: float, times: int) -> float:
    """Multiplicative decay of what an agent keeps after `times` hits.

    A simplified Cobb-Douglas form: each occurrence multiplies what is left by
    `(1 - parameter)`, so the return value is what **survives**, not what is
    lost. Nothing having happened yet (`times = 0`) leaves everything intact
    and returns 1.0.

    Formula:
        f(parameter, times) = (1 - parameter) ** times

    This is the `(1 - group)^n` half of the social term (`social_standing`) --
    the only complement **inside the social term**. The other half, `grid^m`, is
    a plain power: `grid` is already a surviving share, so it needs no `1 -`
    (issue #210). Outside the social term one more complement is legitimate and
    unavoidable: `reports_defector` compares vengefulness against the *cost* of
    a report, `1 - grid`.

    Args:
        parameter: Per-occurrence loss rate in range [0, 1]. Higher values
            decay faster.
        times: Number of occurrences (violations caught, or reports filed).
            Must be non-negative.

    Returns:
        The surviving share, in range [0, 1]: 1.0 when `times` is 0, falling
        towards 0 as `times` grows.

    Raises:
        ValueError: If parameter is outside [0, 1].

    Example:
        What is left of an agent's standing after being caught twice:

        ```python
        # Loses 80% of what remains each time it is caught
        cobb_douglas(0.8, 2)  # (1-0.8)^2 = 0.04 -> almost nothing left

        # Loses only 20% each time
        cobb_douglas(0.2, 2)  # (1-0.2)^2 = 0.64 -> most of it survives
        ```

    Note:
        Read the value as a multiplier on the payoff, never as a cost to
        subtract — the sign was documented backwards until issue #60.
    """
    # 刻意**不**走 `algorithms.require_unit_interval`：这个写法对 NaN 放行，而
    # `core.culture` 的整套设计就建立在"NaN 在入口被拦下、到不了这里"之上
    # （见那边的 Note 与 `test_nan_would_otherwise_slip_past_cobb_douglas`）。
    if parameter > 1 or parameter < 0:
        raise ValueError("Parameter should be between 0 and 1.")
    return (1 - parameter) ** times


def social_standing(
    grid: float, group: float, caught_times: int, punish_times: int
) -> float:
    """Social standing an agent **retains**, as a multiplier on its payoff.

    This is equation (3) of the Supplementary Methods of Castilla-Rho et al.
    (2017, *Nature Human Behaviour*), p. 26:

        S = grid^m * (1 - group)^n

    with `m` the number of times the agent reports a non-compliant neighbour
    (`punish_times`) and `n` the number of times it is caught extracting water
    illegally (`caught_times`). Both counts are per season -- the source assumes
    "agents have no memory of past decisions" (SI II.ii.g).

    **The config values are the source's symbols, used as they are.** `grid` is
    `City.s_grid` and `group` is `City.s_group`, both read straight off the
    World Values Survey columns (issue #102). The `1 -` written above, on
    `group`, is the **only** complement anywhere in the social term: no other
    function in the model or in `water_quota_analysis` complements either
    parameter, and none should be added (issue #210).

    The two bases run in opposite directions, which is the source's design.
    Grid-Group cultural theory gives each of them a reading, and the reading is
    what fixes the direction -- not the arithmetic:

    * **Grid = how far rules are externally imposed.** Under low Grid the rules
      are nobody's business but your own ("who are you to police me?"), so
      reporting a neighbour is a personal betrayal that earns you the informer's
      name: it costs a lot, and `grid^m` decays fast. Under high Grid roles and
      rules come from outside the individual, so reporting is merely doing your
      part -- no stigma, `grid^m` stays near 1. A high-Grid society follows
      social norms strictly and is *more* willing to punish breaches of them
      even with no direct benefit to the punisher. `grid` is therefore the share
      of goodwill that **survives** each report filed, and one report costs
      `1 - grid` -- which is exactly the threshold `reports_defector` compares
      vengefulness against.
    * **Group = how far the individual is embedded in the collective.** Under
      low Group everyone minds their own business and what others think does not
      bite, so being caught barely hurts. Under high Group individual and
      collective interests overlap heavily, so being *seen* by the collective to
      breach is enormously costly. `group` is therefore the share of standing
      **lost** to each criticism, and `n` criticisms leave `(1 - group)^n`.

    At the calibrated `grid = 0.39`, `group = 0.47` (the WVS columns min-max
    scaled across 60 countries, China's entry -- see
    `core.culture.scale_wvs_column` and issue #211) the term is
    `S = 0.39^m * 0.53^n`, so a single critic multiplies the deterrent by
    `1 / (1 - group) = 1.89`.

    Args:
        grid: `City.s_grid`, in [0, 1]. Goodwill surviving one report.
        group: `City.s_group`, in [0, 1]. Standing lost to one criticism.
        caught_times: `n`, neighbours criticising this agent.
        punish_times: `m`, neighbours this agent criticises.

    Returns:
        Retained standing in [0, 1]. 1.0 when both counts are 0.

    Raises:
        ValueError: `grid` or `group` falls outside [0, 1].

    See Also:
        - `water_quota_analysis.analysis.social_cost`: the closed forms for the
          two branches, and the deterrent ratio built on them.
    """
    # 与 `cobb_douglas` 同一条约定：对 NaN 放行，由 `core.culture` 的入口负责拦。
    if grid > 1 or grid < 0:
        raise ValueError("Parameter should be between 0 and 1.")
    goodwill_left = grid**punish_times
    reputation_left = cobb_douglas(group, caught_times)
    return goodwill_left * reputation_left


def reports_defector(vengefulness: float, grid: float) -> bool:
    """Whether an eligible agent criticises a defecting neighbour.

    The single definition of the reporting rule. Filing a report burns
    `1 - grid` of the goodwill an agent still holds -- `grid` being the share
    that survives it -- so an agent files only when the norm matters to it more
    than the report costs:

        report  iff  v > 1 - grid

    The threshold is flat in the number of reports already filed, so
    enforcement is all-or-nothing per agent; the derivation is spelled out in
    `cwatqim.agents.city.City.will_report`, which is the only caller that
    supplies eligibility (an agent that defected last year cannot criticise).

    Args:
        vengefulness: How much the agent cares about the norm, in [0, 1].
        grid: Goodwill surviving one report, in [0, 1] (the `City` parameter
            `s_grid`). The report's cost is its complement.

    Returns:
        True if the agent files a report.

    Note:
        Validates nothing on purpose: this runs once per agent per neighbour
        per year, and both inputs are already guarded upstream --
        `grid` by the `City.s_grid` property and `vengefulness` by its
        `U(0, 1)` initialisation. Be aware that a NaN slipping through would
        return False silently, since `nan > x` is False; that is why the guard
        sits on the property rather than here.

    Example:
        ```python
        reports_defector(0.9, 0.56)   # -> True
        reports_defector(0.3, 0.56)   # -> False
        ```

    See Also:
        - `cwatqim.core.payoff.enforcement_share`: the population share this
          rule implies
        - `cwatqim.agents.city.City.will_report`: the eligibility wrapper
    """
    return vengefulness > 1.0 - grid


def enforcement_share(grid: float) -> float:
    """Share of eligible agents that enforce, under `v ~ U(0, 1)`.

    The measure of `{v : reports_defector(v, grid)}` when vengefulness is
    uniform on the unit interval, which is how `City` initialises it.

    Be honest about what this is: `grid` is the **closed form** of that
    measure, not something computed from the rule, so the two could in
    principle drift apart. What stops them is a test rather than the code --
    `tests/model/test_payoff.py::TestEnforcementRule` evaluates
    `reports_defector` on a dense grid of the unit interval and compares the
    empirical share against this function. Change the rule without changing
    this and the suite goes red, instead of a figure quietly disagreeing with
    the model (issue #129).

    Direction: the cheaper enforcement is -- the more goodwill survives a
    report, i.e. the higher `grid` -- the more of it happens, and the stronger
    the deterrent (issue #109). Under the pre-#210 reading this share was
    `1 - beta_1`; the parameter is now the source's `grid` itself, so the
    complement is gone from both the rule and this closed form.

    Args:
        grid: `City.s_grid`, in [0, 1].

    Returns:
        The fraction of eligible agents that file a report: `grid`.

    Raises:
        ValueError: `grid` is not finite or falls outside [0, 1].

    Example:
        ```python
        enforcement_share(0.39)   # the calibrated value for China
        # -> 0.39
        ```

    See Also:
        - `cwatqim.core.payoff.reports_defector`: the rule itself
    """
    return require_unit_interval("grid", grid)


_DEPRECATED_NAMES = {
    # 旧名不是"过时"，是**反的**：下游拿到 0.95 会读成"损失了 95%"，然后写出
    # `payoff = e * (1 - s)`。`cwatqim` 是带 DOI 的公开包（见 .zenodo.json、
    # sync-public-repo.yml），删名字会打断外部引用，所以留垫片——但必须出声，
    # 而且告警里要写明方向，否则会被当成纯改名而不去复核符号（见 #60）。
    "lost_reputation": "social_standing",
}


def __getattr__(name: str) -> Any:
    """Forward deprecated names, warning about the direction they got wrong.

    Args:
        name: Attribute requested from this module.

    Returns:
        The replacement object, when `name` is a known deprecated alias.

    Raises:
        AttributeError: For any other name, as usual.
    """
    if name in _DEPRECATED_NAMES:
        replacement = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"`{name}` is deprecated; use `{replacement}`. Mind the direction: "
            "it returns the social standing **retained** (1.0 = intact, 0.0 = "
            "fully eroded), not a loss to subtract. The old name said the "
            "opposite — check the sign of anything built on it (see issue #60).",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[replacement]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def sell_crop(
    yield_: float,
    price: float = 1.0,
    area: float = 1.0,
) -> float:
    """Calculate revenue from selling a crop.

    This function calculates the total revenue from crop sales by multiplying
    yield per hectare, price per tonne, and total area.

    Formula:
        revenue = yield (t/ha) * price (RMB/t) * area (ha)

    Args:
        yield_: Crop yield per hectare in tonnes/ha. Must be non-negative.
        price: Crop price per tonne in RMB/t. Default 1.0. Must be positive.
        area: Irrigated area for this crop in hectares. Default 1.0.
            Must be non-negative.

    Returns:
        Total revenue in RMB (Chinese Yuan). The result is a float value
        representing the monetary value of the crop production.

    Example:
        Calculate revenue for maize:

        ```python
        # Maize: 5 t/ha yield, 2000 RMB/t price, 100 ha area
        revenue = sell_crop(yield_=5.0, price=2000.0, area=100.0)
        # Returns: 1,000,000 RMB
        ```

    Note:
        This is a simple linear calculation. For multiple crops, use
        `crops_reward` which handles dictionaries of crops.
    """
    return yield_ * price * area


def crops_reward(
    crop_yields: DictLikeType,
    prices: DictLikeType,
    areas: DictLikeType,
) -> float:
    """Calculate total revenue from multiple crops.

    This function calculates the combined revenue from all crops grown by
    an agent. It handles various input formats:
        - Single crop: Single numeric values for yield, price, area
        - Multiple crops: Dictionaries or Series with crop names as keys

    The function iterates through all crops and sums their individual revenues.

    Args:
        crop_yields: Crop yields per hectare. Can be:
            - float: Single crop yield (t/ha)
            - dict: Dictionary mapping crop names to yields (t/ha)
            - pd.Series: Series with crop names as index, yields as values
        prices: Crop prices per tonne. Can be:
            - float: Single price (RMB/t) used for all crops
            - dict: Dictionary mapping crop names to prices (RMB/t)
            - pd.Series: Series with crop names as index, prices as values
        areas: Irrigated areas. Can be:
            - float: Single area (ha) used for all crops
            - dict: Dictionary mapping crop names to areas (ha)
            - pd.Series: Series with crop names as index, areas as values

    Returns:
        Total revenue in RMB from all crops. The value is the sum of
        individual crop revenues calculated using `sell_crop`.

    Raises:
        TypeError: If crop_yields is not a supported type (float, int, dict,
            or pd.Series).

    Example:
        Calculate revenue for multiple crops:

        ```python
        yields = {"Maize": 5.0, "Wheat": 4.0, "Rice": 6.0}  # t/ha
        prices = {"Maize": 2000, "Wheat": 2500, "Rice": 3000}  # RMB/t
        areas = {"Maize": 100, "Wheat": 80, "Rice": 50}  # ha

        total_revenue = crops_reward(yields, prices, areas)
        # Returns sum of: 1,000,000 + 800,000 + 900,000 = 2,700,000 RMB
        ```

        Single crop (scalar inputs):

        ```python
        revenue = crops_reward(5.0, 2000.0, 100.0)
        # Returns: 1,000,000 RMB
        ```

    See Also:
        - `cwatqim.core.payoff.sell_crop`: Function for single crop revenue
    """
    if isinstance(crop_yields, (float, int)):
        price = squeeze(prices, raise_not_num=True)
        area = squeeze(areas, raise_not_num=True)
        return sell_crop(crop_yields, price=price, area=area)
    if isinstance(crop_yields, pd.Series):
        crop_yields = crop_yields.to_dict()
    if not isinstance(crop_yields, dict):
        raise TypeError(f"{type(crop_yields)} is not allowed.")
    # 对字典进行迭代，每一种作物都进行计算
    reward = 0
    for crop, yield_ in crop_yields.items():
        price = squeeze(prices, get_by=crop)
        area = squeeze(areas, get_by=crop)
        reward += sell_crop(yield_, price=price, area=area)
    return reward


def water_costs(
    q_surface: float,
    q_ground: float,
    price: DictLikeType = 1.0,
    flags: Tuple[str, str] = ("surface", "ground"),
    area: Optional[float] = None,
    unit: WaterUnitType = "m3",
) -> float:
    """Calculate total water cost from surface and groundwater use.

    This function calculates the monetary cost of water use by multiplying
    water volumes by their respective prices. It handles different units
    and can apply different prices for surface water and groundwater.

    Unit conversions:
        - "mm": Converts from mm depth to m³ using area
        - "m3": Uses volumes directly in m³
        - "1e8m3": Converts from 1e8 m³ to m³ for calculation

    Args:
        q_surface: Surface water volume. Units depend on `unit` parameter.
        q_ground: Groundwater volume. Units depend on `unit` parameter.
        price: Water price(s). Can be:
            - float: Single price (RMB/m³) applied to both sources
            - dict: Dictionary with "surface" and "ground" keys (RMB/m³)
            - pd.Series: Series with flags as index, prices as values
        flags: Tuple of (surface_key, ground_key) for dictionary/Series
            price lookups. Default ("surface", "ground").
        area: Irrigated area in hectares. Required when unit="mm" for
            conversion. Optional otherwise.
        unit: Unit of input volumes. Options:
            - "mm": Millimeters (water depth), requires area for conversion
            - "m3": Cubic meters
            - "1e8m3": 100 million cubic meters (converted to m³ internally)

    Returns:
        Total water cost in RMB. Calculated as:
            cost = q_surface_m3 * price_surface + q_ground_m3 * price_ground

    Raises:
        ValueError: If unit is not one of the supported options.
        TypeError: If price type is not supported (must be dict, Series, or
            numeric).

    Example:
        Calculate cost with different prices:

        ```python
        # Surface: 100 m³ at 0.5 RMB/m³, Ground: 50 m³ at 0.8 RMB/m³
        prices = {"surface": 0.5, "ground": 0.8}
        cost = water_costs(100, 50, price=prices, unit="m3")
        # Returns: 100*0.5 + 50*0.8 = 90 RMB
        ```

        Calculate from mm depth:

        ```python
        # 200 mm depth on 100 ha
        cost = water_costs(200, 0, price=0.5, area=100, unit="mm")
        # Converts 200 mm * 100 ha = 200,000 m³, then * 0.5 = 100,000 RMB
        ```

    Note:
        The function automatically handles unit conversions. For mm inputs,
        the conversion factor is 10 (1 ha * 1 mm = 10 m³).
    """
    if unit == "mm":
        q_surface = convert_mm_to_m3(q_surface, area)
        q_ground = convert_mm_to_m3(q_ground, area)
    elif unit == "m3":
        pass
    elif unit == "1e8m3":
        q_ground *= 1e8
        q_surface *= 1e8
    else:
        raise ValueError(f"Unknown water volume unit {unit}.")

    if isinstance(price, (dict, pd.Series)):
        sw, gw = flags
        return q_surface * price[sw] + q_ground * price[gw]
    if isinstance(price, (float, int)):
        return q_surface * price + q_ground * price
    raise TypeError(f"prices should be a dict or a float, got {type(price)}.")


def economic_payoff(
    q_surface: float,  # mm
    q_ground: float,  # mm
    water_prices: DictLikeType,  # RMB/m3
    crop_yield: Optional[float] = None,  # t/ha
    crop_prices: Optional[DictLikeType] = 1.0,  # RMB/t
    area: float = 1.0,  # ha
    unit: WaterUnitType = "mm",
) -> float:
    """Calculate net economic payoff from irrigation.

    This function calculates the net economic benefit from crop production
    and water use. The payoff is the difference between crop revenue and
    water costs:

        payoff = crop_revenue - water_costs

    If crop yield is not provided (None), the function returns the negative
    water cost, representing a pure cost scenario.

    Args:
        q_surface: Surface water use. Units depend on `unit` (default: mm).
        q_ground: Groundwater use. Units depend on `unit` (default: mm).
        water_prices: Water prices in RMB/m³. Can be a single value or
            dictionary with "surface" and "ground" keys for different prices.
        crop_yield: Optional crop yield in tonnes/ha. If None, only water
            costs are considered (negative payoff).
        crop_prices: Crop price in RMB/t. Default 1.0. Can be a single value
            or dictionary for multiple crops. Must not be None when
            `crop_yield` is given (raises ValueError).
        area: Irrigated area in hectares. Default 1.0. Used for converting
            mm to m³ and calculating total crop revenue.
        unit: Unit of water volumes. Default "mm". Options: "mm", "m3", "1e8m3".

    Returns:
        Net economic payoff in RMB, rounded to 2 decimal places. The value
        can be:
            - Positive: Revenue exceeds costs (profitable)
            - Zero: Revenue equals costs (break-even)
            - Negative: Costs exceed revenue (loss)

    Raises:
        ValueError: If `crop_yield` is given but `crop_prices` is None.

    Example:
        Calculate payoff with crop production:

        ```python
        # 500 mm surface water, 200 mm groundwater
        # Yield: 5 t/ha, Price: 2000 RMB/t, Area: 100 ha
        # Water prices: 0.5 RMB/m³ (surface), 0.8 RMB/m³ (ground)
        water_prices = {"surface": 0.5, "ground": 0.8}

        payoff = economic_payoff(
            q_surface=500,
            q_ground=200,
            water_prices=water_prices,
            crop_yield=5.0,
            crop_prices=2000.0,
            area=100.0,
            unit="mm"
        )
        # Revenue: 5 * 2000 * 100 = 1,000,000 RMB
        # Costs: (500*100*10*0.5) + (200*100*10*0.8) = 410,000 RMB
        # Payoff: 590,000 RMB
        ```

        Calculate cost-only (no crop):

        ```python
        # Only water costs, no crop revenue
        cost = economic_payoff(
            q_surface=500,
            q_ground=200,
            water_prices=0.5,
            crop_yield=None,  # No crop
            area=100.0,
            unit="mm"
        )
        # Returns: -410,000 RMB (negative cost)
        ```

    Note:
        This function is used during water source optimization to evaluate
        different allocation strategies. The optimizer seeks to maximize
        this payoff value.

    See Also:
        - `cwatqim.core.payoff.crops_reward`: Crop revenue calculation
        - `cwatqim.core.payoff.water_costs`: Water cost calculation
        - `cwatqim.agents.city.water_withdraw`: Optimization using this function
    """
    costs = water_costs(q_surface, q_ground, water_prices, unit=unit, area=area)
    reward = gross_revenue(crop_yield, crop_prices, area)
    # 没有作物产量时 `gross_revenue` 返回 0.0，于是这里退回"纯成本情景"
    return round(reward - costs, 2)


def gross_revenue(
    crop_yield: Optional[float] = None,  # t/ha
    crop_prices: Optional[DictLikeType] = 1.0,  # RMB/t
    area: float = 1.0,  # ha
) -> float:
    """卖掉全部收成能拿到的钱，不扣任何成本。

    `economic_payoff` 的被减数。它**不进效用**（乘性形式 `U = e·s` 没有标尺），
    单独拿出来是因为分析侧要用它把经济诱惑与威慑放到同一根轴上；2026-08-20 到
    08-30 之间的加性效用曾拿它当社会项的标尺，那段已撤销（见 issue #121）。

    只有一处实现，`economic_payoff` 与 `City.calc_payoff` 都调它——两边各算一遍
    迟早会漂移。

    Args:
        crop_yield: 单产（t/ha）。None 表示还没有收成（例如第一个模拟年，
            `_results` 还是空表），此时返回 0.0。
        crop_prices: 作物价（RMB/t）。
        area: 灌溉面积（ha）。

    Returns:
        毛收入（RMB），恒 ≥ 0。没有收成时为 0.0。

    Raises:
        ValueError: 给了产量却没给价格。静默降级成 0 会让优化目标悄悄丢掉作物
            收益（见 issue #15）。

    Example:
        ```python
        gross_revenue(5.0, 2000.0, area=100.0)  # 5 t/ha x 2000 RMB/t x 100 ha
        # -> 1_000_000.0
        ```

    See Also:
        - `cwatqim.core.payoff.crops_reward`: 逐作物求和的实现
        - `cwatqim.core.payoff.aggregate_utility`: 用它当社会项的标尺
    """
    if crop_yield is None:
        return 0.0
    if crop_prices is None:
        raise ValueError(
            "`crop_prices` is None while `crop_yield` is given: "
            "cannot value the harvest. Pass crop prices explicitly, "
            "or set `crop_yield=None` for a water-cost-only payoff."
        )
    return crops_reward(crop_yield, crop_prices, area)


def aggregate_utility(economic: float, standing: float) -> float:
    """主体真正最大化的量：经济收益按保留下来的社会地位打折。

    Formula:
        U = e · s

    其中 `s ∈ [0,1]` 是**保留下来**的社会地位（1 = 无人批评，方向见 issue #60）。
    社会项是**乘数**而不是减项：被批评把收益整体打折，折扣的深浅由 group 与批评
    人数决定。

    ## 亏损年社会项会反号，而这是**有意保留**的（2026-09-01，作者决定）

    `e < 0` 时 `e·s` 反号：被批评（`s` 变小）反而让 `U` 变大（更接近 0），威慑在
    那一格变成奖励（issue #121）。实测 `e < 0` 占约 12.5% 的城市-年，是真实的亏损
    年（水费盖过作物收入），不是脏数据。

    读法是**经济上本来就是负激励的那一格，社会项转成奖励是可接受的**：亏损年里
    继续种、继续抽水本身已经被经济惩罚了，社会评价不必在那里再叠一层同向的压力。

    ### 为什么不用 `payoff_floor` 去掰正

    2026-08-30 至 09-01 之间这里写的是 `U = max(e, floor) · s`，`floor` 是一个很小
    的正下限，为的就是让方向处处为正。它有一个**致命的副作用**，直到换 canonical
    才暴露（issue #213）：`e ≤ floor` 时 `max(e, floor)` 恒为 `floor`，于是

        U = floor · s

    **与配水完全无关**——同一支内部 `s` 是常数、`floor` 也是常数，目标函数在整个
    可行域上是一条水平线，argmax 退化，解由求解器任取。实测 12.5% 的城市-年因此
    落在 `surface = 0`，既污染一切以地表水为分母的指标（#214），又让 #94 那条
    「配水解是角点解」在 13% 的样本上不成立（1.0000 → 0.8742）。

    那 12.5% 不是"方向反了的结果"，是"根本没有结果"。**反号是个可以讨论的读法，
    退化是个没有解的洞** —— 两害相权，取反号。

    Args:
        economic: 经济收益 e（RMB），可正可负。
        standing: 保留下来的社会地位 s，必须在 [0, 1]；`social_standing` 的产物。

    Returns:
        效用 U（RMB）。`e < 0` 时它同样为负，且随 `s` 变小而**上升** —— 见上。

    Raises:
        ValueError: 任一入参非有限（NaN / ±inf），或 `standing` 越界。
            非有限值必须**显式**拒绝：`nan < 0` 是 False，NaN 会原样穿过区间守卫，
            效用变成 NaN，而 `change_mind` 里 `nan > x` 恒为 False —— 主体从此
            静默地再也学不到东西，全程无异常（同一教训见 `core.culture`）。

    Example:
        ```python
        # 正常年份：被批评越多，效用越低
        aggregate_utility(economic=4e6, standing=1.0)   # -> 4e6
        aggregate_utility(economic=4e6, standing=0.5)   # -> 2e6

        # 亏损年：方向反过来，被批评反而"更好" —— 有意保留，见上
        aggregate_utility(economic=-4e6, standing=1.0)  # -> -4e6
        aggregate_utility(economic=-4e6, standing=0.5)  # -> -2e6
        ```

    Note:
        `include_s` 为假时 `City.agg_payoff` **不**走这里，直接返回 `e` —— 没有乘法
        就不该动经济收益，否则 `never` 情景会跟着动，而它逐位不变正是一条有用的
        一致性检验。

    See Also:
        - `cwatqim.core.payoff.social_standing`: 产出 `standing`
        - `cwatqim.agents.city.City.agg_payoff`: 调用点
    """
    require_finite("经济收益", economic)
    require_unit_interval("社会地位", standing)
    return economic * standing
