#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Core algorithms used in the model."""

from typing import Callable, Dict, List, Optional, Sized, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

Number: TypeAlias = Union[float, int]

DictLikeType: TypeAlias = Union[Dict[str, float], pd.Series, Number]


def ceil_divide(num: int, weights: np.ndarray) -> np.ndarray:
    """根据某个比例，按照期待的权重，将整数进行分配。

    Parameters:
        num:
            需要分配的整数。
        weights:
            权重

    Returns:
        包括各个作物的土地数量。
    """
    if not isinstance(num, int):
        raise TypeError(f"Number {num} (type '{type(num)}') should be an integer.")
    weights = np.array(weights)
    if not weights.any() or weights.ndim != 1:
        raise ValueError(f"Invalid weights: '{weights}'.")
    ratio = weights / weights.sum()
    allocation = np.ceil(ratio * num).astype(int)
    allocation[np.argmax(allocation)] += num - allocation.sum()
    return allocation


def squeeze(
    item: DictLikeType,
    get_by: Optional[str] = None,
    raise_not_num: bool = False,
    aggfunc: str | Callable = "mean",
) -> float:
    """将一个可能是数字或者序列的值进行压缩。"""
    if isinstance(item, (int, float)):
        return item
    if raise_not_num:
        raise TypeError(f"{item} is expected as a number.")
    if get_by:
        return item.get(get_by)
    aggfunc = getattr(np, aggfunc) if isinstance(aggfunc, str) else aggfunc
    # if isinstance(item, (tuple, list, np.ndarray)):
    #     return aggfunc(item)
    if isinstance(item, dict):
        return aggfunc(list(item.values()))
    if isinstance(item, pd.Series):
        return aggfunc(item.values)
    raise TypeError(f"Unknown type {type(item)}.")
