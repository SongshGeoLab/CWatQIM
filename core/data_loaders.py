#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Data loading utilities for the model.

1. 各省的用水配额上限
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pint import UnitRegistry

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from abses.time import TimeDriver

    from ..agents.city import City, Province

WaterUnitType: TypeAlias = Literal["m3", "mm", "1e8m3"]

ureg = UnitRegistry()  # 注册单位
CROPS = ("Rice", "Wheat", "Maize")


@lru_cache
def load_quotas(path: str) -> pd.DataFrame:
    """加载配额数据集。"""
    return pd.read_csv(path, index_col=0)


def update_city_csv(
    data: pd.DataFrame,
    obj: City,
    time: TimeDriver,
) -> pd.Series:
    """从csv文件中动态读取城市用水强度。

    Parameters:
        data:
            数据来源，一个 `pd.DataFrame` 数据框。
        obj:
            应该读取这个数据的实例是一个 `City`。
        time:
            模型的时间驱动器，存储着有关当前模型运行时间的信息。

    Returns:
        该城市不同农作物的时间序列。例如：
        ```
        Maize: xxx,
        Wheat: xxx,
        Rice: xxx,
        ```
    """
    # If City_ID is not set yet (during initialization), return empty series
    if obj.city_id is None:
        return pd.Series({crop: 0.0 for crop in CROPS})

    index = data["Year"] == time.year
    data_tmp = data.loc[index].set_index("City_ID")
    return data_tmp.loc[f"C{obj.city_id}", list(CROPS)]


def update_province_csv(
    data: pd.DataFrame, obj: Province, time: TimeDriver, **kwargs
) -> float:
    """从csv文件中动态读取省份数据。

    Parameters:
        data:
            数据来源，一个 `pd.DataFrame` 数据框。
        obj:
            应该读取这个数据的实例是一个 `Province`。
        time:
            模型的时间驱动器，存储着有关当前模型运行时间的信息。

    Returns:
        某省份在当年的数据。
    """
    return data.loc[time.year, obj.name_en]


def convert_mm_to_m3(num: float, area: float = 1.0) -> float:
    """将单位从毫米转换为立方米"""
    # if not area:
    #     raise ValueError("Area should be provided when unit is 'mm'.")
    return num * area * 10


def convert_ha_mm_to_1e8m3(num: float) -> float:
    """将单位从公顷乘毫米转化成亿立方米"""
    return num * 10 / 1e8
