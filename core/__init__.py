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

from typing import Any

from .algorithms import ceil_divide, squeeze
from .data_loaders import (
    CROPS,
    convert_ha_mm_to_1e8m3,
    convert_mm_to_m3,
    update_city_csv,
    update_province_csv,
)
from .payoff import (
    cobb_douglas,
    economic_payoff,
    sell_crop,
    social_standing,
    water_costs,
)

__all__ = [
    "ceil_divide",
    "squeeze",
    "update_city_csv",
    "update_province_csv",
    "CROPS",
    "convert_mm_to_m3",
    "convert_ha_mm_to_1e8m3",
    "cobb_douglas",
    "economic_payoff",
    "sell_crop",
    "social_standing",
    "water_costs",
]


def __getattr__(name: str) -> Any:
    """Forward deprecated names to `cwatqim.core.payoff`, which warns.

    `lost_reputation` is deliberately absent from the eager imports and from
    `__all__`: importing it here would fire its `DeprecationWarning` on every
    `import cwatqim`, and listing it would advertise a name whose meaning is
    the reverse of its value (see issue #60).

    Args:
        name: Attribute requested from this package.

    Returns:
        The attribute resolved through `cwatqim.core.payoff`.

    Raises:
        AttributeError: If `payoff` does not define it either.
    """
    from . import payoff

    return getattr(payoff, name)
