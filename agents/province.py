#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import pandas as pd
from abses import Actor, ActorsList, MainModel
from loguru import logger

from ..core import update_province_csv

if TYPE_CHECKING:
    from .city import City


class Province(Actor):
    """每个省的主体。

    在黄河的水分配模型中，省起到的作用主要是向下一级分配水配额。
    因为“八七”分水方案是省尺度进行配额的，但省一级单位通常不了解有多少用水需求。
    因此，本模型假设省份控制总配额量，根据灌溉面积将其分配给各个地级市。
    """

    _instances = {}

    def __init__(self, *args, name: str, **kwargs) -> None:
        self.name_en = name
        super().__init__(*args, **kwargs)
        if name not in self.p.names:
            raise ValueError(f"Province {name} not in parameters.")

    def __str__(self) -> str:
        """返回省份的英文名。"""
        return self.name_en

    def __repr__(self) -> str:
        """返回省份的英文名。"""
        return f"<Province: {str(self)}>"

    def __hash__(self) -> int:
        """Make Manager instances hashable for use as dictionary keys.

        Uses the unique_id from the parent Actor class which is guaranteed
        to be immutable and unique for each agent instance.
        """
        return hash(self.unique_id)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, str):
            is_eng_name = getattr(self, "name_en", None) == __value
            is_ch_name = getattr(self, "name_ch", None) == __value
            return is_ch_name or is_eng_name
        return super().__eq__(__value)

    @cached_property
    def sw_irr_eff(self) -> float:
        """地表水灌溉效率。"""
        return self.p.sw_irr_eff[self.name_en]

    @cached_property
    def gw_irr_eff(self) -> float:
        """地下水灌溉效率。"""
        return self.p.gw_irr_eff[self.name_en]

    @property
    def managed(self) -> ActorsList[Actor]:
        """该省所管辖的主体。
        - 对于城市：是该城市所有的农民主体。
        - 对于省：是该省所有的市水资源管理单位。
        """
        return self.link.get(self.breed, default=True)

    @property
    def quota(self) -> float:
        """计算当前的水资源配额，统计数据通常是亿立方米，需要转化单位为立方米。"""
        return self._quota

    @quota.setter
    def quota(self, value: float) -> None:
        self.assign(value, "quota", by="total_area")
        self._quota = value

    @classmethod
    def create(cls, model: MainModel, name_en: str, **kwargs) -> Province:
        """使用单例模式创造一个省主体。

        Parameters:
            model:
                当前模型。
            name_en:
                省份的英文名。

        Returns:
            一个省份的实例。
        """
        # 对当前模型的每个省份，只有一个实例
        if model not in cls._instances:
            cls._instances[model] = {}
        # 如果已经有了这个省份的实例，直接返回
        if instance := cls._instances[model].get(name_en):
            return instance
        # 否则，创建一个新的实例
        instance: Province = model.agents.new(
            cls, singleton=True, name=name_en, **kwargs
        )
        cls._instances[model][name_en] = instance
        return instance

    @cached_property
    def water_prices(self) -> Dict[str, float]:
        """水资源价格字典，分别指示地表水和地下水的价格。
        价格应该是一个正数，从数据中读取，单位是元/立方米。

        - 'ground': 地下水价格
        - 'surface': 地表水价格
        """
        data = pd.read_csv(self.ds.prices, index_col=0)
        return data.loc[self.name_en, ["surface", "ground"]].to_dict()

    @cached_property
    def crop_prices(self) -> Dict[str, float]:
        """作物价格字典，分别指示水稻、小麦和玉米的价格。
        价格应该是一个正数，从数据中读取，单位是元/千克。
        """
        data = pd.read_csv(self.ds.prices, index_col=0)
        prices = data.loc[self.name_en, ["rice", "wheat", "maize"]]
        prices.index = prices.index.str.capitalize()
        return (prices * 1000).to_dict()

    def setup(self):
        """初始化一个省份的数据。

        1. 水资源配额数据。来源于黄河水资源分配方案。
        2. 属于该省份的主体数量数据。来源于统计局各省的乡村数量数据。
        """
        self.add_dynamic_variable(
            name="quota",
            data=pd.read_csv(self.ds.quotas, index_col=0),
            function=update_province_csv,
        )

    def assign(self, value: float, attr: str, by: Optional[str] = None) -> np.ndarray:
        """将水资源额度分配给下辖的所有主体。

        Parameters:
            quota:
                需要分配的水资源额度。
            flag:
                分配的是最大水资源（flag = 'max'）还是最小水资源（flag = 'min'）
            weighted_by:
                权重的属性名。默认为空，为空则从配置文件中提取。

        Returns:
            一个数组，包括所有主体的水资源额度。
        """
        # 如果没有任何被管理者，则什么也不做，直接返回。
        if not self.managed:
            raise ValueError("No managed agents to assign quota.")
        # 如果没有设置用什么作为权重进行水资源分配，则获取。
        if by is None:
            weights = np.ones(len(self.managed))
        else:
            # 如果设置了权重，根据权重进行分配。
            weights = self.managed.array(attr=by)
            # 如果权重加起来是0（即大家都没需求的情况），所有人等权重。
            if not weights.sum():
                weights = np.ones(len(self.managed))
        values = value * weights / weights.sum()
        self.managed.update(attr, values)
        return values

    def update_data(
        self,
    ) -> None:
        """更新配额数据和主体数量数据，按需分配给每个其管辖的地级市。
        权重变量应该是一个字符串，指向一个已经存在于所辖城市主体的动态变量。

        Parameters:
            agents_by:
                用于分配农民数量的权重变量。
            quotas_by:
                用于分配水资源配额的权重变量。
        """
        self.quota = self.dynamic_var("quota") * 1e8
        # === logging ===
        logger.info(
            f"{self.name_en} assigned water quota to {len(self.managed)} Cities."
        )

    def update_graph(
        self,
        l_p: float,
        mutual: bool = True,
    ) -> int:
        """更新社会网络。
        社会网络代表着不同乡村之间的联系。
        当乡村之间有联系时，他们可以感知到对方的决策信息。
        如果乡村 A 知道了乡村 B 大量超用水，那么乡村 A 可能会感到不满。
        这种不满会同时降低两者的社会满意程度。

        Parameters:
            l_p:
                省之间，hub节点的概率。
            l_c:
                城市内部的联系概率。

        Returns:
            创建的链接数量。
        """
        links = self.managed.random.link("friend", p=l_p, mutual=mutual)
        # 为了方便测试，记录一下链接的数量
        logger.info(f"{self.name_en} has {len(links)} links.")
        return len(links)
