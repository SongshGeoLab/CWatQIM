#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Data loading utilities for the model.

This module provides functions for loading and updating time-varying data
from CSV files. The functions are designed to work with the ABSESpy dynamic
variable system, which allows agent attributes to change over time based on
external data sources.

Key functions:
    - `update_city_csv`: Loads city-specific data (irrigation area, WUI)
    - `update_province_csv`: Loads province-specific data (water quotas)
    - Unit conversion functions for water volumes
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
    from abses.core.time_driver import TimeDriver

    from ..agents.city import City, Province

WaterUnitType: TypeAlias = Literal["m3", "mm", "1e8m3"]
"""Type alias for supported water volume units."""

ureg = UnitRegistry()
"""Pint unit registry for unit conversions and validation."""

CROPS = ("Rice", "Wheat", "Maize")
"""Tuple of crop names used throughout the model."""


@lru_cache
def load_quotas(path: str) -> pd.DataFrame:
    """Load water quota data from CSV file with caching.

    This function loads province-level water quota data from a CSV file.
    The data is cached using LRU cache to avoid repeated file I/O operations.

    Expected CSV format:
        - Index: Years (integer)
        - Columns: Province names (e.g., "Henan", "Shandong")
        - Values: Water quotas in 1e8 m³ (100 million cubic meters)

    Args:
        path: File path to the CSV file containing quota data.

    Returns:
        DataFrame with years as index and provinces as columns. Values are
        water quotas in 1e8 m³.

    Note:
        The function uses `@lru_cache` to cache the loaded data. If the file
        changes during execution, the cache will not reflect the changes
        unless the Python process is restarted.

    Example:
        Load quota data:

        ```python
        quotas = load_quotas("data/processed/quotas.csv")
        # Access quota for Henan in 2000
        henan_2000 = quotas.loc[2000, "Henan"]
        ```
    """
    return pd.read_csv(path, index_col=0)


def update_city_csv(
    data: pd.DataFrame,
    obj: City,
    time: TimeDriver,
    **kwargs,
) -> pd.Series:
    """Extract city-specific data from CSV for the current simulation year.

    This function is used as an update function for dynamic variables in
    the ABSESpy framework. It extracts year- and city-specific data from a
    DataFrame and returns it as a Series indexed by crop names.

    Expected DataFrame format:
        - Columns: "Year", "City_ID", "Maize", "Wheat", "Rice"
        - Rows: One row per city per year
        - City_ID format: "C{id}" (e.g., "C102" for city ID 102)

    Args:
        data: DataFrame containing city data with columns for Year, City_ID,
            and crop names (Maize, Wheat, Rice).
        obj: City agent instance for which to extract data. The function uses
            `obj.city_id` to identify the correct row.
        time: TimeDriver object from the model, providing `time.year` for
            filtering the correct year.

    Returns:
        Pandas Series with crop names as index ("Maize", "Wheat", "Rice")
        and data values as values. If the city ID is not yet set (during
        initialization), returns a Series of zeros.

    Example:
        Use as dynamic variable update function:

        ```python
        # In City.setup()
        city.add_dynamic_variable(
            name="wui",
            data=pd.read_csv("data/irr_wui.csv"),
            function=update_city_csv
        )

        # Later, during simulation
        wui_data = city.dynamic_var("wui")  # Returns Series for current year
        maize_wui = wui_data["Maize"]  # WUI for maize
        ```

    Note:
        This function handles the case where `city_id` is None (during
        initialization) by returning zeros. This prevents errors when the
        function is called before the city ID is assigned from the shapefile.
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
    """Extract province-specific data from CSV for the current simulation year.

    This function is used as an update function for dynamic variables in
    the ABSESpy framework. It extracts year- and province-specific data from
    a DataFrame.

    Expected DataFrame format:
        - Index: Years (integer)
        - Columns: Province names (e.g., "Henan", "Shandong")
        - Values: Province-specific data (e.g., water quotas in 1e8 m³)

    Args:
        data: DataFrame with years as index and provinces as columns.
        obj: Province agent instance for which to extract data. The function
            uses `obj.name_en` to identify the correct column.
        time: TimeDriver object from the model, providing `time.year` for
            filtering the correct year.
        **kwargs: Additional keyword arguments (not used, but accepted for
            compatibility with ABSESpy update function signature).

    Returns:
        Float value for the province in the current year. Typically represents
        water quota in 1e8 m³.

    Raises:
        KeyError: If the year or province name is not found in the DataFrame.

    Example:
        Use as dynamic variable update function:

        ```python
        # In Province.setup()
        province.add_dynamic_variable(
            name="quota",
            data=pd.read_csv("data/quotas.csv", index_col=0),
            function=update_province_csv
        )

        # Later, during simulation
        quota = province.dynamic_var("quota")  # Returns float for current year
        ```

    Note:
        The function assumes the DataFrame index contains years and columns
        contain province names matching `obj.name_en`.
    """
    return data.loc[time.year, obj.name_en]


def convert_mm_to_m3(num: float, area: float = 1.0) -> float:
    """Convert water depth in millimeters to volume in cubic meters.

    This function converts a water depth (in mm) applied over an area (in ha)
    to a volume (in m³). The conversion uses the relationship:
        1 ha = 10,000 m²
        1 mm = 0.001 m
        Volume = depth (m) * area (m²) = (depth_mm * 0.001) * (area_ha * 10000)
                = depth_mm * area_ha * 10

    Args:
        num: Water depth in millimeters (mm).
        area: Irrigated area in hectares (ha). Default 1.0. Must be positive.

    Returns:
        Water volume in cubic meters (m³).

    Example:
        Convert irrigation depth to volume:

        ```python
        # 200 mm depth on 100 ha
        volume = convert_mm_to_m3(200, area=100)
        # Returns: 200 * 100 * 10 = 200,000 m³
        ```

    Note:
        The conversion factor of 10 comes from:
        - 1 ha = 10,000 m²
        - 1 mm = 0.001 m
        - Factor = 10,000 * 0.001 = 10
    """
    # if not area:
    #     raise ValueError("Area should be provided when unit is 'mm'.")
    return num * area * 10


def convert_ha_mm_to_1e8m3(num: float) -> float:
    """Convert water volume from hectare-millimeters to 1e8 cubic meters.

    This function converts a water volume represented as the product of area
    (ha) and depth (mm) to units of 1e8 m³ (100 million cubic meters), which
    is a common unit for large-scale water management in China.

    The conversion:
        1 ha * 1 mm = 10 m³ (from convert_mm_to_m3)
        1e8 m³ = 100,000,000 m³
        Therefore: ha*mm / 1e8 = (ha*mm * 10) / 1e8 = ha*mm * 10 / 1e8

    Args:
        num: Water volume in hectare-millimeters (ha*mm). This is typically
            the product of irrigated area (ha) and water depth (mm).

    Returns:
        Water volume in units of 1e8 m³ (100 million cubic meters).

    Example:
        Convert irrigation volume:

        ```python
        # 100 ha * 500 mm = 50,000 ha*mm
        volume_1e8m3 = convert_ha_mm_to_1e8m3(50000)
        # Returns: 50000 * 10 / 1e8 = 0.005 1e8 m³
        ```

        Convert from Series (per-crop):

        ```python
        # Series: area (ha) * depth (mm) for each crop
        irr_volume = pd.Series([5000, 4000, 3000], index=["Maize", "Wheat", "Rice"])
        volume_1e8m3 = convert_ha_mm_to_1e8m3(irr_volume)
        # Returns Series with volumes in 1e8 m³
        ```

    Note:
        This unit (1e8 m³) is commonly used in Chinese water resource
        management because it provides a convenient scale for basin-level
        water allocations and quotas.
    """
    return num * 10 / 1e8
