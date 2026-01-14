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
    - Social costs (reputation loss)
    - Combined economic and social payoffs

These functions are used by City agents to evaluate different water use
strategies and make optimal decisions.
"""

from typing import Dict, Literal, Optional, Tuple, Union

import pandas as pd
from loguru import logger

from .algorithms import DictLikeType, squeeze
from .data_loaders import WaterUnitType, convert_mm_to_m3


def check_boundary(
    low_boundary: float,
    up_boundary: float,
) -> None:
    """检查边界值是否符合条件"""
    if low_boundary < 0.0:
        raise ValueError(f"Invalid lower boundary: {low_boundary}.")
    if up_boundary < 0.0:
        raise ValueError(f"Invalid up boundary: {up_boundary}.")
    if low_boundary == up_boundary:
        raise ValueError(f"Invalid boundary values: {low_boundary} == {up_boundary}.")


def cobb_douglas(parameter: float, times: int) -> float:
    """Calculate Cobb-Douglas function value.

    This function implements a simplified Cobb-Douglas form used for modeling
    non-linear relationships, particularly in social cost calculations. The
    function creates a decreasing exponential relationship where the output
    decreases as `times` increases.

    Formula:
        f(parameter, times) = (1 - parameter) ** times

    This is used to model:
        - Reputation loss: Decreases exponentially with number of violations
        - Social enforcement cost: Decreases with number of punishments

    Args:
        parameter: Base parameter in range [0, 1]. Higher values lead to
            faster decay. Typically represents a cost or loss rate.
        times: Exponent representing the number of occurrences (e.g., number
            of violations, number of punishments). Must be non-negative.

    Returns:
        Function value in range [0, 1]. When times=0, returns 1.0. As times
        increases, the value approaches 0.

    Raises:
        ValueError: If parameter is outside [0, 1].

    Example:
        Calculate reputation loss after multiple violations:

        ```python
        # High reputation sensitivity (0.8), 2 violations
        loss = cobb_douglas(0.8, 2)  # (1-0.8)^2 = 0.04

        # Low reputation sensitivity (0.2), 2 violations
        loss = cobb_douglas(0.2, 2)  # (1-0.2)^2 = 0.64
        ```

    Note:
        The function ensures that social costs decrease non-linearly with
        repeated violations, modeling the idea that people become desensitized
        to violations over time.
    """
    if parameter > 1 or parameter < 0:
        raise ValueError("Parameter should be between 0 and 1.")
    return (1 - parameter) ** times


def lost_reputation(
    cost: float, reputation: float, caught_times: int, punish_times: int
) -> float:
    """Calculate reputation loss from rule violations and social enforcement.

    This function models the social cost of violating water quota rules,
    combining two mechanisms:
        1. **Reputation Loss**: Cost from being caught and criticized by
           neighbors. Decreases exponentially with number of violations
           (people become desensitized).
        2. **Enforcement Cost**: Cost from actively reporting others'
           violations. Decreases with number of reports (reluctance to
           repeatedly report).

    Both components use the Cobb-Douglas function to model non-linear decay.
    The final cost is the average of both components.

    Args:
        cost: Base enforcement cost parameter [0, 1]. Higher values indicate
            greater initial cost of reporting others.
        reputation: Base reputation sensitivity [0, 1]. Higher values indicate
            greater initial reputation loss from being caught.
        caught_times: Number of times the agent has been caught violating
            rules (non-negative integer).
        punish_times: Number of times the agent has reported others' violations
            (non-negative integer).

    Returns:
        Combined reputation loss in range [0, 1], where:
            - 0.0: No reputation loss (best case)
            - 1.0: Maximum reputation loss (worst case)
        The value is the average of enforcement cost and reputation loss.

    Example:
        Calculate reputation loss for an agent:

        ```python
        # Agent with high sensitivity, caught 3 times, reported others 1 time
        loss = lost_reputation(
            cost=0.5,
            reputation=0.8,
            caught_times=3,
            punish_times=1
        )
        # Returns average of enforcement and reputation components
        ```

    Note:
        This function is used in the social cost calculation to determine
        how much an agent's social satisfaction decreases due to rule
        violations and enforcement actions.

    See Also:
        - `cwatqim.core.payoff.cobb_douglas`: Underlying decay function
        - `cwatqim.agents.city.calc_social_costs`: Method using this function
    """
    # lost reputation because of others' report
    lost = cobb_douglas(reputation, caught_times)
    # not willing to offensively report others
    cost = cobb_douglas(cost, punish_times)
    return (cost + lost) / 2


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
    crop_prices: float = 1.0,  # RMB/t
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
            or dictionary for multiple crops.
        area: Irrigated area in hectares. Default 1.0. Used for converting
            mm to m³ and calculating total crop revenue.
        unit: Unit of water volumes. Default "mm". Options: "mm", "m3", "1e8m3".

    Returns:
        Net economic payoff in RMB, rounded to 2 decimal places. The value
        can be:
            - Positive: Revenue exceeds costs (profitable)
            - Zero: Revenue equals costs (break-even)
            - Negative: Costs exceed revenue (loss)

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
    # 如果没有作物产量，直接返回负的水费
    if crop_yield is None or crop_prices is None:
        return -round(costs, 2)
    # 否则计算作物收益，减去水费
    reward = crops_reward(crop_yield, crop_prices, area)
    return round(reward - costs, 2)
