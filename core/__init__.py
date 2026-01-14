#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Core utilities and algorithms for the CWatQIM model.

This package provides fundamental utilities used throughout the model:
    - Algorithms: Allocation, aggregation, and data manipulation functions
    - Data loaders: Functions for loading time-varying data from CSV files
    - Payoff calculations: Economic and social payoff functions

These utilities are designed to be independent and reusable across different
parts of the model.
"""

from .algorithms import ceil_divide, squeeze
from .data_loaders import (
    CROPS,
    convert_ha_mm_to_1e8m3,
    convert_mm_to_m3,
    update_city_csv,
    update_province_csv,
    ureg,
)
from .payoff import (
    cobb_douglas,
    economic_payoff,
    lost_reputation,
    sell_crop,
    water_costs,
)

__all__ = [
    "ceil_divide",
    "squeeze",
    "update_city_csv",
    "update_province_csv",
    "ureg",
    "CROPS",
    "convert_mm_to_m3",
    "convert_ha_mm_to_1e8m3",
    "cobb_douglas",
    "economic_payoff",
    "lost_reputation",
    "sell_crop",
    "water_costs",
]
