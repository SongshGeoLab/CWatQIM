#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

from functools import cached_property
from typing import Dict, Optional

import numpy as np
import pandas as pd
from abses import Actor, ActorsList, MainModel

from ..core import update_province_csv


class Province(Actor):
    """Province-level agent managing water quota allocation to cities.

    The Province agent represents a provincial-level water management authority
    in the Yellow River Basin. It implements the "87 Water Allocation Scheme"
    (八七分水方案), which allocates water quotas at the provincial scale.

    Key responsibilities:
        1. **Quota Management**: Receives annual water quota from the basin
           authority and allocates it to subordinate cities
        2. **Proportional Allocation**: Distributes quota based on irrigated
           area, ensuring cities with larger irrigation needs receive
           proportionally more quota
        3. **Social Network Management**: Creates and maintains social network
           links between cities within the province, enabling information
           sharing and peer influence

    The allocation mechanism assumes that provinces have limited information
    about actual water demand at the city level, so they use irrigated area
    as a proxy for water needs. This creates a realistic constraint where
    allocation may not perfectly match demand.

    Attributes:
        name_en (str): English name of the province (e.g., "Henan", "Shandong").
        _quota (float): Total water quota allocated to this province in m³.
            Note: The public `quota` property returns this in 1e8 m³.
        managed (ActorsList): List of City agents under this province's
            jurisdiction, linked through the "Province" relationship.

    Class Attributes:
        _instances (dict): Class-level dictionary tracking province instances
            per model to ensure singleton behavior (one province per name
            per model instance).

    Example:
        Create and use a province:

        ```python
        # Create province (singleton pattern)
        henan = Province.create(model, "Henan")

        # Access quota and allocation
        print(f"Total quota: {henan.quota} 1e8 m³")
        print(f"Managed cities: {len(henan.managed)}")

        # Update and allocate quota
        henan.update_data()  # Loads quota from data and allocates to cities
        ```

    Note:
        Provinces are created using a singleton pattern to ensure only one
        instance exists per province name per model. This prevents duplicate
        provinces and ensures consistent quota allocation.

    See Also:
        - `cwatqim.agents.city.City`: City agents that receive quota allocations
        - `cwatqim.model.main.CWatQIModel`: Main model that manages provinces
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
        """Create or retrieve a province agent using singleton pattern.

        This class method ensures that only one Province instance exists for
        each province name within a model. If a province with the given name
        already exists, it returns the existing instance. Otherwise, it
        creates a new one.

        The singleton pattern is important because:
            - Quota allocation should be managed by a single authority
            - Prevents duplicate provinces with conflicting allocations
            - Ensures consistent province-level data (prices, efficiency)

        Args:
            model: The main model instance in which to create the province.
            name_en: English name of the province. Must match a name in
                `model.p.names` (province names from configuration).
            **kwargs: Additional arguments passed to the Actor constructor.

        Returns:
            Province instance. If one already exists with this name in this
            model, returns the existing instance. Otherwise, creates and
            returns a new instance.

        Raises:
            ValueError: If the province name is not in `model.p.names`.

        Example:
            Create provinces for a model:

            ```python
            # First call creates the province
            henan1 = Province.create(model, "Henan")

            # Second call returns the same instance
            henan2 = Province.create(model, "Henan")
            assert henan1 is henan2  # True
            ```

        Note:
            The singleton is tracked per model instance, so the same province
            name can exist in different model runs without conflict.
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
        """Allocate a value to all managed cities proportionally.

        This method distributes a total value (e.g., water quota) among all
        cities under this province's jurisdiction. The allocation is
        proportional to a specified attribute (e.g., irrigated area).

        The allocation formula:
            allocation_i = value * (weight_i / sum(weights))

        If weights sum to zero (all cities have zero weight), equal weights
        are used instead.

        Args:
            value: Total value to allocate (e.g., total quota in m³).
            attr: Attribute name to set on managed cities (e.g., "quota").
            by: Attribute name used as weights for proportional allocation.
                If None, uses equal weights (1.0 for all cities). Common
                values include "total_area" (irrigated area) for quota
                allocation.

        Returns:
            NumPy array containing the allocated values for each managed
            city, in the same order as `self.managed`.

        Raises:
            ValueError: If there are no managed cities to assign to.

        Example:
            Allocate quota based on irrigated area:

            ```python
            # Allocate 1000 m³ total quota proportionally by area
            allocations = province.assign(
                value=1000.0,
                attr="quota",
                by="total_area"
            )

            # Each city now has quota set proportionally
            for city, quota in zip(province.managed, allocations):
                print(f"{city.city_name}: {quota} m³")
            ```

        Note:
            The method updates the attribute on all managed cities
            automatically. The returned array is for reference/verification.
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

    def update_data(self) -> None:
        """Update province quota data and allocate to cities.

        This method performs the annual update of water quota allocation:
            1. Loads the current year's quota from the dynamic variable
            2. Converts from 1e8 m³ to m³ for internal calculation
            3. Allocates the quota to all managed cities proportionally
               based on their total irrigated area

        The allocation uses `self.assign()` with `by="total_area"`, ensuring
        that cities with larger irrigated areas receive proportionally more
        quota. This reflects the assumption that larger irrigation areas
        indicate greater water needs.

        Note:
            This method is called automatically during the model's step()
            method. It should be called after cities have updated their
            irrigated area data for the current year.

        Raises:
            ValueError: If no cities are managed by this province, or if
                the quota dynamic variable cannot be accessed.

        Example:
            Update quota allocation for a year:

            ```python
            # During model step
            province.update_data()

            # Check allocations
            for city in province.managed:
                print(f"{city.city_name}: {city.quota} 1e8 m³")
            ```

        See Also:
            - `cwatqim.agents.province.assign`: Proportional allocation method
            - `cwatqim.core.data_loaders.update_province_csv`: Function for
                loading province quota data from CSV
        """
        self.quota = self.dynamic_var("quota") * 1e8

    def update_graph(
        self,
        l_p: float,
        mutual: bool = True,
    ) -> int:
        """Update the social network between cities within the province.

        This method creates friendship links between cities, enabling social
        learning and peer influence. The social network allows cities to:
            - Observe neighbors' water use decisions
            - Compare their own performance with neighbors
            - Learn successful strategies from better performers
            - Experience social costs when neighbors violate rules

        The network is created probabilistically, where each potential link
        between cities has probability `l_p` of being created. This creates
        a random graph structure that varies between simulation runs.

        Args:
            l_p: Probability of creating a link between any two cities
                within the province. Range [0, 1], where:
                - 0.0: No links created (isolated cities)
                - 1.0: All possible links created (fully connected)
            mutual: If True, links are bidirectional (if A links to B, then
                B also links to A). If False, links are directed.

        Returns:
            Number of links created in the social network. This can be used
            to verify network connectivity and for logging purposes.

        Example:
            Create a sparse social network:

            ```python
            # Create links with 20% probability
            num_links = province.update_graph(l_p=0.2, mutual=True)
            print(f"Created {num_links} friendship links")

            # Check a city's friends
            city = province.managed[0]
            friends = city.friends
            print(f"{city.city_name} has {len(friends)} friends")
            ```

        Note:
            The social network is updated annually, allowing the network
            structure to evolve over time. However, in the current
            implementation, links are recreated each year rather than
            persisting.

        See Also:
            - `cwatqim.agents.city.friends`: Property accessing city's
                social network connections
            - `cwatqim.agents.city.change_mind`: Social learning using
                the network
        """
        links = self.managed.random.link("friend", p=l_p, mutual=mutual)
        return len(links)
