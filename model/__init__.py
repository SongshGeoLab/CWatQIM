#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Model package for the CWatQIM framework.

This package contains the main model class that orchestrates the multi-agent
simulation of water quota allocation in the Yellow River Basin.
"""

from .main import CWatQIModel

__all__ = [
    "CWatQIModel",
]
