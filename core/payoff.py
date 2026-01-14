#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Functions for calculating water use payoffs."""

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
    """Cobb-Douglas函数"""
    if parameter > 1 or parameter < 0:
        raise ValueError("Parameter should be between 0 and 1.")
    return (1 - parameter) ** times


def lost_reputation(
    cost: float, reputation: float, caught_times: int, punish_times: int
) -> float:
    """损失声誉"""
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
    """计算某种作物赚的钱

    Parameters:
        yield_:
            作物单产，单位为吨/公顷。
        price:
            作物价格，单位为元/吨。
        area:
            种植面积，单位为公顷，默认为1公顷。

    Returns:
        作物的收益，单位为元。
    """
    return yield_ * price * area


def crops_reward(
    crop_yields: DictLikeType,
    prices: DictLikeType,
    areas: DictLikeType,
) -> float:
    """计算所有作物的收益

    Parameters:
        crop_yields:
            作物产量，单位为吨/公顷。
        crop_prices:
            作物价格，单位为元/吨。

    Returns:
        所有作物的收益，单位为元。
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
    """计算农民的水费

    Parameters:
        q_surface:
            地表水用量，单位为立方米或毫米。
        q_ground:
            地下水用量，单位为立方米。
        price:
            水价，单位为元/立方米。
        flags:
            地表水和地下水的标识。

    Returns:
        水费，单位为元。
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
    """计算农民的经济收益，作物收益减去水费。"""
    costs = water_costs(q_surface, q_ground, water_prices, unit=unit, area=area)
    # 如果没有作物产量，直接返回负的水费
    if crop_yield is None or crop_prices is None:
        return -round(costs, 2)
    # 否则计算作物收益，减去水费
    reward = crops_reward(crop_yield, crop_prices, area)
    return round(reward - costs, 2)
