#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/


from typing import Dict, cast

import geopandas as gpd
import rioxarray
import xarray as xr
from abses import BaseNature
from aquacrop_abses import CropCell, CropLand
from loguru import logger


class Nature(BaseNature):
    """自然模块"""

    def setup(self):
        """初始化自然模块。

        1. 从矢量文件中导入城市数据。
        2. 从栅格文件中导入灌溉区域数据。
        3. 从栅格文件中导入土壤数据。
        """
        cities = gpd.read_file(self.ds.cities.shp)

        yr_basin: CropLand = self.create_module(
            vector_file=cities,
            resolution=0.1,  # 0.1 degree.
            major_layer=True,
            name="yr_basin",
            cell_cls=CropCell,
            module_cls=CropLand,
        )

        # 导入灌溉区域数据（如果配置了的话）
        if hasattr(self.ds, "irr_area") and self.ds.irr_area is not None:
            data = rioxarray.open_rasterio(self.ds.irr_area)
            yr_basin.apply_raster(data, attr_name="irr_area", resampling_method="sum")
        else:
            logger.warning(
                "No irr_area raster configured, skipping irrigation area setup"
            )
