#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Agent classes for the CWatQIM model.

This package contains the agent classes that represent the decision-making
entities in the water quota allocation model:
    - City: City-level irrigation units that make water use decisions
    - Province: Province-level authorities that allocate water quotas

All agents inherit from ABSESpy base classes and integrate with AquaCrop
for crop simulation.
"""

from aquacrop_abses.farmer import Farmer

from .city import City
from .province import Province

__all__ = [
    "City",
    "Farmer",
    "Province",
]
