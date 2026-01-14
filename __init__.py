#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Water Quota Model Package

A multi-agent model for Yellow River water quota allocation.
This package contains the core model components and can be published independently.
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

__version__ = "0.1.0"
