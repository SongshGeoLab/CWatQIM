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

These functions are used by City agents to evaluate different water use
strategies and make optimal decisions.

Note:
    The social term is a **multiplier on the payoff**, not a cost to
    subtract: 1.0 is the untouched case and 0.0 the fully eroded one. It was
    named and documented the other way round until issue #60.
"""

import warnings
from typing import Any, Optional, Tuple

import pandas as pd

from .algorithms import DictLikeType, squeeze
from .data_loaders import WaterUnitType, convert_mm_to_m3


def cobb_douglas(parameter: float, times: int) -> float:
    """Multiplicative decay of what an agent keeps after `times` hits.

    A simplified Cobb-Douglas form: each occurrence multiplies what is left by
    `(1 - parameter)`, so the return value is what **survives**, not what is
    lost. Nothing having happened yet (`times = 0`) leaves everything intact
    and returns 1.0.

    Formula:
        f(parameter, times) = (1 - parameter) ** times

    Used twice in the social term of the payoff — once for standing lost to
    neighbours' criticism, once for the goodwill spent criticising them.

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
    if parameter > 1 or parameter < 0:
        raise ValueError("Parameter should be between 0 and 1.")
    return (1 - parameter) ** times


def social_standing(
    cost: float, reputation: float, caught_times: int, punish_times: int
) -> float:
    """Social standing an agent **retains**, as a multiplier on its payoff.

    Two mechanisms erode standing, each decaying multiplicatively via
    `cobb_douglas`:
        1. **Reputation**: eroded by every neighbour who criticises this
           agent's over-withdrawal.
        2. **Enforcement**: eroded by every neighbour this agent criticises --
           reporting a peer is not free.

    The result is the average of the two surviving shares.

    Formula:
        s = [ (1 - cost)^punish_times + (1 - reputation)^caught_times ] / 2

    Args:
        cost: Per-report goodwill lost when criticising a neighbour, in
            [0, 1] (the `City` parameter `s_enforcement_cost`).
        reputation: Per-criticism standing lost when caught, in [0, 1]
            (the `City` parameter `s_reputation`).
        caught_times: Number of neighbours criticising this agent.
        punish_times: Number of neighbours this agent criticises.

    Returns:
        Retained standing in range [0, 1], where:
            - 1.0: nobody criticised, and nobody was criticised (best case)
            - 0.0: standing entirely eroded (worst case)

    Example:
        ```python
        # Nothing has happened yet: standing is intact
        social_standing(cost=0.5, reputation=0.8, caught_times=0, punish_times=0)
        # -> 1.0

        # Criticised by three neighbours, criticised one in turn
        social_standing(cost=0.5, reputation=0.8, caught_times=3, punish_times=1)
        # -> (0.5 ** 1 + 0.2 ** 3) / 2 = 0.254
        ```

    Note:
        Both the old name (`lost_reputation`) and its docstring described the
        **complement** of what the arithmetic returns (see issue #60). The
        direction matters: `City.agg_payoff` computes `payoff = e * s`, so a
        value near 0 is the punishment and a value near 1 is the intact case.
        Writing it up as a cost to subtract would invert the mechanism.

    See Also:
        - `cwatqim.core.payoff.cobb_douglas`: Underlying decay function
        - `cwatqim.agents.city.City.calc_social_standing`: Method using it
    """
    # what survives of this agent's reputation after neighbours criticised it
    reputation_left = cobb_douglas(reputation, caught_times)
    # what survives of its goodwill after it criticised neighbours in turn
    goodwill_left = cobb_douglas(cost, punish_times)
    return (goodwill_left + reputation_left) / 2


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
    # 如果没有作物产量（纯成本情景），直接返回负的水费
    if crop_yield is None:
        return -round(costs, 2)
    # 有产量却没有价格，说明调用方漏传了参数：静默降级成"只算水费"会让
    # 优化目标悄悄丢掉作物收益（见 issue #15），因此这里必须报错。
    if crop_prices is None:
        raise ValueError(
            "`crop_prices` is None while `crop_yield` is given: "
            "cannot value the harvest. Pass crop prices explicitly, "
            "or set `crop_yield=None` for a water-cost-only payoff."
        )
    # 否则计算作物收益，减去水费
    reward = crops_reward(crop_yield, crop_prices, area)
    return round(reward - costs, 2)
