#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Agent classes for the model."""

from aquacrop_abses.farmer import Farmer

from .city import City
from .nature import Nature
from .province import Province

__all__ = [
    "City",
    "Farmer",
    "Province",
    "Nature",
]
