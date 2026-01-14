#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Water Quota Incentive Model (CWatQIM) Package.

A agent-based model for simulating Yellow River water quota allocation and
analyzing policy incentives. This package implements an agent-based model (ABM)
that simulates the interactions between provinces, cities, and farmers in the
Yellow River Basin, focusing on water quota compliance and social learning
mechanisms.

The model is built on the ABSESpy framework and integrates with AquaCrop for
crop yield simulation. It can be published independently as it contains only
the core model components without analysis dependencies.

Main Components:
    - CWatQIModel: The main model class that orchestrates the simulation
    - City: City-level agents representing irrigation units
    - Province: Province-level agents managing water quota allocation
    - Core utilities: Algorithms, data loaders, and payoff calculations

Example:
    Basic usage of the model:

    ```python
    from cwatqim import CWatQIModel
    from hydra import compose, initialize

    with initialize(config_path="config"):
        cfg = compose(config_name="config")
        model = CWatQIModel(parameters=cfg)
        model.setup()
        for _ in range(10):
            model.step()
        model.end()
    ```

Note:
    This package is designed to be independent of analysis tools. For result
    analysis, use the separate `water_quota_analysis` package.
"""

from .agents import City, Farmer, Province
from .model import CWatQIModel

__all__ = [
    "City",
    "Farmer",
    "Province",
    "Experiment",
    "CWatQIModel",
]

__version__ = "0.1.1"  # x-release-please-version
