#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Literal, Optional

import geopandas as gpd
from abses import ActorsList, MainModel
from loguru import logger

from ..agents.city import City
from ..agents.province import Province

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

ManagerType: TypeAlias = Literal["Province", "City"]


class YellowRiver(MainModel):
    """模拟黄河水资源分配的多主体模型。

    基于 ABSESpy 框架搭建。
    """

    def setup(self) -> None:
        """配置模型。

        从 `.shp` 文件里读取城市数据集，并用其创建 City 主体：
        - `City_ID` 属性将被用作该主体的唯一ID。
        - `Province_n` 属性将被读作为该城市的省份。
        """
        cities = gpd.read_file(self.ds.cities.shp)
        self.agents.new_from_gdf(
            gdf=cities,
            agent_cls=City,
            attrs={"Province_n": "province", "City_ID": "City_ID"},
        )

    @property
    def provinces(self) -> ActorsList[Province]:
        """选择当前的所有省份主体。"""
        return self.agents[Province]

    @property
    def cities(self) -> ActorsList[City]:
        """选择当前的所有城市主体。"""
        return self.agents[City]

    def sel_city(self, city_id: Optional[int] = None) -> City:
        """根据ID选择一个城市

        Parameters:
            city_id:
                城市主体的唯一ID（来自City_ID属性）。

        Returns:
            利用城市ID筛选的城市主体。

        Raises:
            ValueError: 如果找不到对应的城市。
        """
        if city_id is None:
            return self.cities.random.choice()
        return self.cities.select({"city_id": city_id}).item("only")

    def sel_prov(self, name_en: Optional[str] = None) -> Province:
        """根据英文名称选择一个省份。

        Parameters:
            name_en:
                省份英文名称。

        Returns:
            利用省份英文名称筛选的省份主体。

        Raises:
            ValueError: 如果找不到对应的省份，或者有多个省份匹配。
        """
        if name_en is None:
            return self.provinces.random.choice()
        return self.provinces.select({"name_en": name_en}).item("only")

    def step(self):
        """
        模型每一时间步（年）所触发的行动。

        - 模型的时间步更新（即本方法 model.step）
            1. 更新各省份数据
            2. 更新各省社会网络
        - 主体的时间步自动更新（由于 schedule.step）
        - 数据收集（datacollector.collect）
        - 日志记录
        """
        # preparing parameters
        logger.info(f"Starting a new year: {self.time.year}")
        self.provinces.shuffle_do("update_data")
        self.provinces.shuffle_do("update_graph", l_p=self.p["l_p"])
        self.cities.shuffle_do("step")

        # 收集数据
        self.datacollector.collect(self)

    def end(self) -> None:
        """结束模拟并存储数据。"""
        logger.info("Simulation ends.")
        # 确保输出目录存在
        self.outpath.mkdir(parents=True, exist_ok=True)
        df_cities = self.datacollector.get_agent_vars_dataframe("City")
        df_cities.to_csv(self.outpath / f"{self.run_id}_cities.csv")
        # Validation logic moved to analysis layer
