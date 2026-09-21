#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Core algorithms and utility functions for the model.

This module provides fundamental algorithms used throughout the model,
including allocation algorithms, data type conversions, aggregation
functions, and the shared numeric guards.
"""

import math
from typing import Callable, Dict, Iterable, Optional, TypeAlias, Union

import numpy as np
import pandas as pd

Number: TypeAlias = Union[float, int]
"""Type alias for numeric values."""

DictLikeType: TypeAlias = Union[Dict[str, float], pd.Series, Number]
"""Type alias for values that can be represented as dict, Series, or number."""


def ceil_divide(num: int, weights: np.ndarray) -> np.ndarray:
    """Allocate an integer value proportionally using ceiling division.

    This function distributes a total integer value among multiple categories
    based on proportional weights. It uses ceiling division to ensure that:
        1. Each category receives at least its proportional share (rounded up)
        2. The total allocation exactly equals the input value
        3. The largest category absorbs any rounding differences

    The algorithm:
        1. Calculate proportional ratios: ratio_i = weight_i / sum(weights)
        2. Apply ceiling: allocation_i = ceil(ratio_i * num)
        3. Adjust for exact total: Add/subtract difference from largest category

    Args:
        num: Total integer value to allocate (must be non-negative).
        weights: Array of weights for proportional allocation. Must be
            non-negative and at least one weight must be positive. Can be
            any 1D array-like structure.

    Returns:
        NumPy array of integers with the same length as weights, where each
        element represents the allocated value for that category. The sum
        of all elements equals `num`.

    Raises:
        TypeError: If `num` is not an integer.
        ValueError: If `weights` is empty, has wrong dimensions, or all
            weights are zero.

    Example:
        Allocate 10 units among 3 categories:

        ```python
        weights = np.array([1, 2, 3])
        allocation = ceil_divide(10, weights)
        # Result: [2, 3, 5] (proportional to [1, 2, 3], sums to 10)
        ```

        Allocate land among crops:

        ```python
        crop_weights = np.array([0.3, 0.5, 0.2])  # Maize, Wheat, Rice
        total_land = 100
        land_allocation = ceil_divide(total_land, crop_weights)
        # Result: [30, 50, 20] (or similar, depending on rounding)
        ```

    Note:
        This function is useful when you need to allocate discrete resources
        (like land parcels, agents, etc.) proportionally while ensuring
        the total is exact.
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
    """Extract a single numeric value from various data structures.

    This utility function provides a flexible way to extract numeric values
    from different data types commonly used in the model:
        - Single numbers: Returns as-is
        - Dictionaries: Extracts by key or aggregates all values
        - Pandas Series: Extracts by index or aggregates all values
        - Lists/arrays: Aggregates all values

    This is particularly useful when dealing with crop data that might be
    represented as a single value (one crop) or a dictionary/Series (multiple
    crops).

    Args:
        item: The value to extract from. Can be:
            - int or float: Returned directly
            - dict: Extracted by key or aggregated
            - pd.Series: Extracted by index or aggregated
        get_by: If provided, extract a specific key/index from dict/Series.
            If None, aggregates all values.
        raise_not_num: If True, raises TypeError if item is not a number
            and get_by is not provided. If False, attempts aggregation.
        aggfunc: Aggregation function to use when get_by is None. Can be:
            - String: Name of numpy function (e.g., "mean", "sum", "max")
            - Callable: Function that accepts an iterable and returns a number

    Returns:
        Single float value extracted or aggregated from the input.

    Raises:
        TypeError: If `raise_not_num=True` and item is not a number, or if
            item type is not supported.
        KeyError: If `get_by` is provided but key/index doesn't exist.

    Example:
        Extract from dictionary:

        ```python
        crop_yields = {"Maize": 5.0, "Wheat": 4.0, "Rice": 6.0}
        maize = squeeze(crop_yields, get_by="Maize")  # Returns 5.0
        avg = squeeze(crop_yields, aggfunc="mean")  # Returns 5.0
        ```

        Handle single values:

        ```python
        single_yield = 5.0
        value = squeeze(single_yield)  # Returns 5.0
        ```

        Extract from Series:

        ```python
        series = pd.Series([1, 2, 3], index=["a", "b", "c"])
        value_a = squeeze(series, get_by="a")  # Returns 1.0
        total = squeeze(series, aggfunc="sum")  # Returns 6.0
        ```
    """
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


def require_finite(name: str, value: float) -> float:
    """Return `value`, refusing NaN and the infinities.

    The single home for a guard that was written out five times over (issue
    #136): twice in `core.payoff` (`aggregate_utility` and
    `enforcement_share`), once in `core.culture`, and once in each of two
    `City` properties. Most of those docstrings pointed at `core.culture` as
    "the same lesson", which is how one rule ended up copied five ways.

    It has to be a **separate** check rather than folded into a range test,
    because every comparison against NaN is False: `nan < 0`, `nan > 1` and
    `0 <= nan <= 1` all evaluate False, so a bare range guard waves NaN
    through and the model runs on garbage (issue #131).

    Args:
        name: What to call the value in the error message. Use the name the
            reader would recognise -- a config key like
            `City.s_grid`, not a local variable.
        value: The number to check.

    Returns:
        `value` unchanged, so this can wrap an assignment.

    Raises:
        ValueError: `value` is NaN, `inf`, or `-inf`.

    Example:
        ```python
        grid = require_finite("City.s_grid", float(params["s_grid"]))
        ```
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限值，收到 {value!r}")
    return value


def require_one_of(name: str, value: str, allowed: Iterable[str]) -> str:
    """Return `value`, refusing anything outside a closed set of names.

    The third sibling of `require_finite` / `require_unit_interval`, for the
    other guard this repo had started copying: config keys whose domain is a
    handful of strings. Three of them now exist (`City.s_grid_level`,
    `City.s_grid_spread`, `model.network`) and each had written its own
    `if x not in ALLOWED: raise` — the same drift that produced #131 and #136
    on the numeric side, one shape at a time.

    Failing loudly matters more here than the tidiness: every one of these
    switches selects between **mechanisms**, so a typo that silently fell back
    to a default would produce a complete, plausible set of results for the
    branch nobody asked for.

    Args:
        name: What to call the value in the error message. Use the config key
            a reader would recognise -- `model.network`, not `mode`.
        value: The string to check.
        allowed: The permitted values.

    Returns:
        `value` unchanged, so this can wrap an assignment.

    Raises:
        ValueError: `value` is not in `allowed`. The message lists the
            permitted values, because a typo is the likely cause.

    Example:
        ```python
        mode = require_one_of("model.network", raw, NETWORK_MODES)
        ```
    """
    options = tuple(allowed)
    if value not in options:
        raise ValueError(f"{name} 必须是 {options} 之一，收到 {value!r}")
    return value


def require_unit_interval(name: str, value: float) -> float:
    """Return `value`, refusing anything outside a finite [0, 1].

    `require_finite`'s sibling for the other guard this repo kept copying: five
    places wrote `if not 0.0 <= x <= 1.0: raise ValueError(f"… 必须在 [0, 1] …")`
    by hand, and they had already drifted — some spelled it `x > 1 or x < 0`,
    which **passes NaN** (every comparison against NaN is False), while the
    `not 0.0 <= x <= 1.0` spelling **rejects** it. Same intent, opposite
    behaviour on the one input that matters (issues #131, #136).

    This one is strict on both counts: it calls `require_finite` first, so NaN
    and the infinities raise before the range test runs.

    Note:
        Two callers stay deliberately NaN-**permissive** and therefore do *not*
        use this: `payoff.cobb_douglas` and `payoff.social_standing`. That is
        the contract `core.culture` is built on — its entry points reject NaN
        so it never reaches the Cobb-Douglas, and
        `tests/model/test_culture.py::test_nan_would_otherwise_slip_past_cobb_douglas`
        pins it. Both sites carry a comment saying so; don't "fix" them here.

    Args:
        name: What to call the value in the error message. Use the name the
            reader would recognise -- a config key like `City.s_grid`.
        value: The number to check.

    Returns:
        `value` unchanged, so this can wrap an assignment.

    Raises:
        ValueError: `value` is non-finite, or falls outside [0, 1].

    Example:
        ```python
        grid = require_unit_interval("City.s_grid", float(params["s_grid"]))
        ```
    """
    require_finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} 必须在 [0, 1]，收到 {value!r}")
    return value
