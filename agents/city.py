#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from abses import ActorsList, PatchModule
from aquacrop import Crop, InitialWaterContent, IrrigationManagement
from aquacrop.core import AquaCropModel
from aquacrop.entities.soil import Soil
from aquacrop_abses import CropCell
from aquacrop_abses.cell import get_crop_datetime
from aquacrop_abses.farmer import Farmer
from aquacrop_abses.load_datasets import crop_name_to_crop
from loguru import logger
from scipy.optimize import differential_evolution

from ..core import economic_payoff, lost_reputation, update_city_csv
from ..core.data_loaders import convert_ha_mm_to_1e8m3
from .province import Province

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

DecisionType: TypeAlias = Literal["D", "C"]

REQUIRED_COLS = ["MinTemp", "MaxTemp", "Precipitation", "ReferenceET", "Date"]


def to_regional_crop(crop: str, province: Optional[str] = None) -> str:
    """将作物转换为区域作物。"""
    if crop in ["Maize", "Rice"]:
        return f"Regional{crop}"
    if crop == "Wheat":
        return decide_wheat_season(crop, province)
    return crop


def decide_wheat_season(crop: str, province: Optional[str] = None) -> Crop:
    """决定小麦的种植季节。"""
    if crop != "Wheat":
        return crop
    if province in ["Henan", "Shandong", "Shanxi", "Shaanxi"]:
        return "RegionalWheat"
    return "Spring_Wheat"


class City(Farmer):
    """每个城市的主体。

    城市是一个灌溉单元，随机的在该地区可耕地上种植。

    农民主体是对 AquaCrop_abses 中农民主体的一个扩展。
    AquaCrop-abses 里，农民会根据其所在地块的：
    1. 土壤类型（沙土、粘土、壤土等）
    2. 作物类型（水稻、玉米、小麦等）
    3. 气象资料（逐日尺度的高温/低温、降水、蒸散发）

    以及农民的农田管理因素如：
    1. 灌溉策略（灌溉量、灌溉频率、灌溉方式等）
    2. 田间管理策略（薄膜、田梗等）

    从而估算农民的灌溉用水需求，以及作物的潜在产量。
    还可以使用遗传算法，根据作物的用水量，估计用水的来源。
    本模型中进行继承，主要是为了模拟农民模仿灌溉策略的社会行为。

    Attributes:
        irr_area:
            灌溉面积，单位是公顷。
        irr_method:
            灌溉方式，取值范围是1-4。
        crop:
            作物类型。

        quota:
            水资源配额。
        surface_water:
            地表水用量。
        ground_water:
            地下水用量。

        boldness:
            大胆程度。
        vengefulness:
            报复心。
        willing:
            是否超过最严格水资源配额进行农业取水的决策。

        e:
            经济得分，取值[0~inf)。
            经济得分是对
        s:
            社会得分，取值[0, 1]。
        payoff:
            综合得分。
            决策单元会对比自己的得分和周围人的得分，从而评估自己的排名。
            而综合得分是经济得分和社会得分各自**排名**的加权平均值。
    """

    valid_decisions: Dict[DecisionType, str] = {
        "C": "Cooperate: compliance with water quota.",
        "D": "Defect: use more water than quota.",
    }

    def __getattr__(self, name: str):
        if name.startswith("dry_yield"):
            crop_name = name.split("_")[-1].capitalize()
            return self.dry_yield.get(crop_name)
        if name.startswith("yield_potential"):
            crop_name = name.split("_")[-1].capitalize()
            return self.yield_potential.get(crop_name)
        return super().__getattr__(name)

    @property
    def climate_datapath(self) -> Path:
        dp = Path(self.ds.city_climate_dir) / f"climate_C{self.city_id}.csv"
        assert dp.exists(), f"Climate data file not found: {dp}"
        return dp

    @cached_property
    def climate_data(self) -> pd.DataFrame:
        """Climate data for this city.

        Loads once on first access. Avoids loading during setup when City_ID
        may not yet be assigned by new_from_gdf.

        Returns:
            DataFrame with daily climate data for all available years.
        """
        from pathlib import Path

        if not hasattr(self.ds, "city_climate_dir"):
            raise ValueError(
                "City climate directory not configured. Please add 'city_climate_dir' to ds."
            )
        climate_dir = Path(self.ds.city_climate_dir)
        climate_file = climate_dir / f"climate_C{self.city_id}.csv"
        if not climate_file.exists():
            raise FileNotFoundError(f"Climate data file not found: {climate_file}")
        df = pd.read_csv(climate_file, parse_dates=["Date"]).copy()
        df = df[REQUIRED_COLS]
        df["ReferenceET"] = df["ReferenceET"].clip(lower=0.1)
        return df

    @property
    def city_id(self) -> int:
        """City unique identifier from City_ID attribute.

        This property returns the City_ID value which is loaded from the
        GeoDataFrame when creating city agents. This is different from the
        ABSESpy internal unique_id.

        Returns:
            The city ID from the City_ID attribute.
        """
        return getattr(self, "City_ID", None)

    @property
    def city_name(self) -> str:
        return f"C{self.city_id}"

    # ========== Properties for data collection ==========

    @property
    def area_maize(self) -> float:
        """Maize irrigation area in ha. For data collection."""
        return self.irr_area.get("Maize", 0.0)

    @property
    def area_wheat(self) -> float:
        """Wheat irrigation area in ha. For data collection."""
        return self.irr_area.get("Wheat", 0.0)

    @property
    def area_rice(self) -> float:
        """Rice irrigation area in ha. For data collection."""
        return self.irr_area.get("Rice", 0.0)

    @property
    def total_yield_maize(self) -> float:
        """Total maize yield in tonnes. For data collection."""
        yield_per_ha = self.dry_yield.get("Maize", 0.0)
        return yield_per_ha * self.area_maize if yield_per_ha else 0.0

    @property
    def total_yield_wheat(self) -> float:
        """Total wheat yield in tonnes. For data collection."""
        yield_per_ha = self.dry_yield.get("Wheat", 0.0)
        return yield_per_ha * self.area_wheat if yield_per_ha else 0.0

    @property
    def total_yield_rice(self) -> float:
        """Total rice yield in tonnes. For data collection."""
        yield_per_ha = self.dry_yield.get("Rice", 0.0)
        return yield_per_ha * self.area_rice if yield_per_ha else 0.0

    @property
    def quota_intensity(self) -> float:
        """Water quota per unit area in m³/ha. For data collection.

        Calculated as: quota (1e8 m³) * 1e8 / total_area (ha) = m³/ha
        """
        if self.total_area <= 0:
            return 0.0
        # quota is in 1e8 m³, convert to m³/ha
        return (self.quota * 1e8) / self.total_area

    @property
    def water_use_intensity(self) -> float:
        """Water use per unit area in m³/ha. For data collection.

        Calculated as: total_wu (1e8 m³) * 1e8 / total_area (ha) = m³/ha
        """
        if self.total_area <= 0:
            return 0.0
        # total_wu is in 1e8 m³, convert to m³/ha
        return (self.total_wu * 1e8) / self.total_area

    @property
    def surface_ratio(self) -> float:
        """Surface water ratio (0-1). For data collection.

        Calculated as: surface_water / (surface_water + ground_water)
        """
        total = self.surface_water + self.ground_water
        if total == 0:
            return 0.0
        return self.surface_water / total

    @property
    def crop_here(self) -> List[str]:
        """当前地块上的作物"""
        return self.irr_area.index.to_list()

    @property
    def irr_area(self) -> pd.Series:
        """总灌溉面积，单位 ha

        Note:
            Irrigated area is stored per-crop as hectare (ha).
            When multiplied by a water depth in mm, use the conversion:
            ha * mm -> m^3 via factor 10 (1 ha = 10,000 m^2; 1 mm = 0.001 m).
        """
        return self.dynamic_var("irr_area")

    @property
    def total_area(self) -> float:
        """总灌溉面积，单位 ha"""
        return self.irr_area.sum()

    @property
    def water_used(self) -> pd.Series:
        """总用水量，单位亿立方米

        Implementation detail:
            Convert ha*mm to 1e8 m^3. The helper uses factor 10 to get m^3
            and then divides by 1e8 to yield 1e8 m^3.
        """
        return convert_ha_mm_to_1e8m3(self.irr_area * self.wui)

    @property
    def total_wu(self) -> float:
        """总用水量"""
        return self.water_used.sum()

    @property
    def net_irr(self) -> float:
        """总灌溉量，单位亿立方米

        Note:
            seasonal_irrigation is in mm, multiply by area (ha) and convert
            to 1e8 m^3 using the same conversion as above.
        """
        if self.seasonal_irrigation is None:
            return 0.0
        ser = self.irr_area * self.seasonal_irrigation
        return convert_ha_mm_to_1e8m3(ser).sum()

    @property
    def province(self) -> Province:
        """该城市所在的省份对象，见`Province`类。
        如果还没设置，则返回 None。

        设置该属性时，可以设置已经创建好的省对象，也可以使用每个省唯一的省份名称。
        """
        return self._province

    @province.setter
    def province(self, province: Province | str) -> None:
        if isinstance(province, str):
            province = Province.create(model=self.model, name_en=province)
        if not isinstance(province, Province):
            raise TypeError("Can only setup province.")
        province.link.to(self, link_name=province.breed, mutual=True)
        self._province = province

    @property
    def province_name(self) -> str:
        return self.province.name_en

    @property
    def wui(self) -> float:
        """
        灌溉用水是每年为灌溉而提取的水量，包括在输送和田间应用过程中的损失。
        因此这个数据是统计得来的，而非模型模拟的。

        Irrigation water use is the annual quantity of water withdrawn for irrigation including the losses during conveyance and field application.
        """
        return self.dynamic_var("wui")

    @property
    def quota(self) -> float:
        """水资源配额，单位：亿立方米 (1e8 m³)

        配额以体积形式存储和比较，与 surface_water、ground_water 的单位一致。

        Internal storage:
            _quota stores raw volume in m^3; getter converts to 1e8 m^3.
        """
        return self._quota / 1e8  # 从立方米转换为亿立方米

    @quota.setter
    def quota(self, volume_m3: float) -> None:
        """设置水资源配额

        Parameters:
            volume_m3:
                配额体积，单位：立方米 (m³)
        """
        self._quota = float(volume_m3)

    @property
    def decision(self) -> DecisionType:
        """实际的用水决策。

        如果利用的黄河地表水资源超过了配额，那么就视为违反了制度（D）
        否则，就是遵守制度（C）

        单位：surface_water 和 quota 都是亿立方米 (1e8 m³)
        """
        if self.surface_water > self.quota:
            return "D"
        return "C"

    @property
    def water_prices(self) -> Dict[str, float]:
        """水价数据，单位是元/立方米。"""
        return self.province.water_prices

    @property
    def crop_prices(self) -> Dict[str, float]:
        """作物价格数据，原始单位是元/kg，转化后单位是元/吨。"""
        return self.province.crop_prices

    @property
    def include_s(self) -> bool:
        """是否包含社会得分。"""
        # return self.time.year >= self.p.include_s_since
        return True

    def setup(self) -> None:
        """添加自动变化的动态数据，完成农民主体的初始化。

        在初始化时，农民主体会完成以下几个步骤：
        1. 初始化农田相关的属性。
        2. 初始化水相关的属性。
        3. 初始化社会相关的属性。
        4. 初始化得分相关的属性。
        """
        self.add_dynamic_variable(
            name="wui",
            data=pd.read_csv(self.ds.irr_wui),
            function=update_city_csv,
        )
        self.add_dynamic_variable(
            name="irr_area",
            data=pd.read_csv(self.ds.irr_area_ha, index_col=0),
            function=update_city_csv,
        )
        # ===== 农田相关 =====
        self.irr_method = 4
        # ===== 水相关 =====
        self._quota = 0.0
        self.surface_water = 0.0
        self.ground_water = 0.0
        # ===== 社会相关 =====
        self.boldness = self.random.random()
        self.vengefulness = self.random.random()
        self.willing = self.make_decision()
        # ===== 得分相关 =====
        # income: -inf~inf
        # social benefits: 0~1
        self.agg_payoff(e=0.0, s=1.0, record=True, rank=False, include_s=self.include_s)

    def calc_max_irr_seasonal(self, crop: str) -> float:
        """计算最大灌溉量。"""
        wui = self.wui[crop]
        sw = self.surface_water / (self.surface_water + self.ground_water)
        gw = self.ground_water / (self.surface_water + self.ground_water)
        return wui * sw * self.province.sw_irr_eff + wui * gw * self.province.gw_irr_eff

    def simulate(self, crop: Optional[Crop] = None, repeats: int = 1) -> pd.DataFrame:
        """模拟一个生长季的作物产量。

        如果作物为空，则模拟所有作物。

        Parameters:
            crop:
                作物类型，如 "Wheat", "Maize", "Rice"。
                如果为 None，则模拟所有在该地区种植的作物。
            repeats:
                重复模拟次数，用于平均多次模拟结果。

        Returns:
            作物模拟结果的 DataFrame。
        """
        if crop is None:
            # Simulate all crops in this area
            results = {crop: self.simulate(crop=crop) for crop in self.crop_here}
            df = pd.DataFrame(results).T
            df["Irrigation volume (1e8m3)"] = convert_ha_mm_to_1e8m3(
                self.irr_area * df["Seasonal irrigation (mm)"]
            )
            self._results = df
            return df

        # Use cached climate data for efficiency
        weather_df = self.climate_data

        # Determine crop type (handle wheat season)
        crop_name = to_regional_crop(crop, self.province_name)
        regionalize = True if crop in ["Maize", "Wheat"] else False
        crop_obj = crop_name_to_crop(crop_name, regionalized=regionalize)
        start_dt, end_dt = get_crop_datetime(crop=crop_obj, year=self.time.year)

        irr_strategy = IrrigationManagement(
            irrigation_method=self.p.irr_method,
            SMT=self.p.SMT,
            AppEff=self.p.irr_eff,
            MaxIrrSeason=self.calc_max_irr_seasonal(crop),
        )
        assert irr_strategy.MaxIrrSeason == self.calc_max_irr_seasonal(crop)
        ac_model = AquaCropModel(
            sim_start_time=start_dt.strftime("%Y/%m/%d"),
            sim_end_time=end_dt.strftime("%Y/%m/%d"),
            weather_df=weather_df,
            soil=Soil("Loam"),
            crop=crop_obj,
            initial_water_content=InitialWaterContent(wc_type="Pct", value=[70]),
            irrigation_management=irr_strategy,
        )
        ac_model.run_model(till_termination=True)
        return ac_model.get_simulation_results().iloc[0]

    def water_withdraw(
        self,
        ufunc: Optional[Callable] = None,
        total_irrigation: Optional[float] = None,
        surface_boundaries: Optional[Tuple[float, float]] = None,
        crop_yield: str = "dry_yield",
        ga_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        """估算取水方式。

        通过遗传算法优化表面水和地下水的取水量。
        用户可以自定义收益函数，这个收益函数需允许接受以下三个参数：
        - crop_yield: 作物产量，是一个 Series，分别是不同类型作物的产量。
        - q_surface: 表面水取水量，是一个浮点数。
        - q_ground: 地下水取水量，是一个浮点数。

        如果用户不输入自定义函数，优化的目标是最大化经济收益。
        需要输入一个 water_prices 的参数，这个参数是一个字典，包含两个键值对：
        - "surface": 表面水价格。
        - "ground": 地下水价格。

        Parameters:
            ufunc:
                自定义的收益函数。
            surface_lb:
                表面水取水量的下限。
            yield_col:
                作物产量的列名。
            ga_kwargs:
                传递给遗传算法的参数。
            **kwargs:
                传递给收益函数的参数。

        Returns:
            最优的表面水和地下水取水量。

        Raises:
            ValueError:
                如果表面水取水量超出范围。
                或者没有提供自定义函数，但是缺少了 water_prices 参数。
        """
        if total_irrigation is None:
            total_irrigation = self.seasonal_irrigation
        if total_irrigation == 0.0:
            logger.warning(f"Zero irr volume for {self.unique_id}.")
            return 0.0, 0.0
        if surface_boundaries is None:
            surface_boundaries = (0.0, total_irrigation)
        surface_lb, surface_ub = surface_boundaries
        if surface_lb < 0.0 or max(surface_lb, surface_ub) > total_irrigation:
            raise ValueError(f"Invalid boundary values: {surface_boundaries}.")
        if ga_kwargs is None:
            ga_kwargs = {}
        if ufunc is None:
            if "water_prices" not in kwargs:
                raise ValueError(
                    "No custom function provided, calculating water costs."
                    "However, Missing arg `water_prices` in kwargs."
                )
            ufunc = economic_payoff
        if isinstance(crop_yield, str):
            crop_yield = getattr(self, crop_yield)

        def fitness(q_surface: np.ndarray) -> float:
            """Objective function for optimization.

            Note: We negate the result because differential_evolution minimizes,
            but we want to maximize the payoff.
            """
            q_surface_val = (
                q_surface[0] if isinstance(q_surface, np.ndarray) else q_surface
            )
            q_ground = total_irrigation - q_surface_val
            kwargs.update(
                {
                    "crop_yield": crop_yield,
                    "q_surface": q_surface_val,
                    "q_ground": q_ground,
                }
            )
            return -ufunc(**kwargs)

        # Use differential_evolution for optimization
        # Merge default parameters with user-provided ga_kwargs
        de_params = {
            "popsize": 15,  # Population size multiplier
            "maxiter": 100,  # Maximum iterations
            "polish": True,  # Use L-BFGS-B to polish final solution
            "seed": None,  # Use model's random state if needed
        }
        de_params.update(ga_kwargs)

        result = differential_evolution(
            func=fitness,
            bounds=[(surface_lb, surface_ub)],
            **de_params,
        )

        q_surface_opt = result.x[0]
        q_ground_opt = total_irrigation - q_surface_opt
        return q_surface_opt, q_ground_opt

    def get_cells(
        self,
        layer: Optional[PatchModule] = None,
    ) -> ActorsList[CropCell]:
        """获取这个城市的所有土地单元。

        Parameters:
            layer:
                土地单元所在的图层。
        """
        if layer is None:
            layer = self.model.nature.major_layer
        return layer.select(self.geometry)

    @property
    def friends(self) -> ActorsList[Farmer]:
        """周围的“朋友”。

        主体会因在周围人的评价中改变自己的行为偏好，以及感知自己的收益。
        为了在多主体模型中模拟这一行为，我们使用来自 ”多元文化理论” 进行建模。

        Returns:
            周围的朋友列表（一个主体列表），其中是与本农民有链接“friend”的主体。
        """
        return self.link.get("friend", default=True)

    @property
    def willing(self) -> DecisionType:
        """农民超过水资源配额的意愿，但意愿并不代表最终决策。

        只有两个可能的取值：
        - D: 如果作物有需要，且经济实惠，则可能会违反最严格水资源配额。
        - C: 遵循最严格水资源配额。作物的额外需要会用地下水进行灌溉。

        当年份大于等于设置的强制年份时，农民会强制遵守规则。
        这一设定对应着一种政策，即在某一年份之后，农民必须遵守水资源配额。
        黄河历史上，强制的水资源分配政策在 1999 年开始正式实施。
        """
        if self.time.year < self.p.get("include_s_since"):
            return "D"
        if self.time.year >= self.p.get("forced_since"):
            return "C"
        return self._willing

    @willing.setter
    def willing(self, value: DecisionType) -> None:
        if value not in self.valid_decisions:
            raise ValueError(f"Invalid decision: {self._willing}.")
        self._willing = value

    def compare(self, attr: str, my: Optional[float] = None) -> float:
        """和朋友之间针对某属性进行比较。

        Parameters:
            attr:
                要计算的属性名称。
                将使用 `ActorsList.array` 方法获取朋友的属性值数组。
            my:
                自己值，如果没有给定，则使用同样的属性值。

        Returns:
            计算归一化的属性值，用[0, 1]的值表征自己在朋友中的属性排名，即：
            - 如果自己的属性值是最大的，那么返回 1.0（最优）
            - 如果所有人的属性值都一样，同样返回 1.0（平均主义）
            - 如果自己的属性值是最小的，那么返回 0.0（最差）
        """
        if not self.friends:
            return 1.0
        arr = self.friends.array(attr=attr)
        my = self.get(attr) if my is None else my
        min_val, max_val = min([my, arr.min()]), max([my, arr.max()])
        if min_val == max_val:
            return 1.0
        return (my - min_val) / (max_val - min_val)

    def calc_social_costs(
        self,
        q_surface: float,
    ) -> float:
        """计算社会成本，包括：
        1. 违反规则而导致的声誉损失
        2. 由于周围村庄违反规则而产生的不满
        两者的权重由参数 `s_enforcement_cost` 和 `s_reputation` 控制。
        使用柯布-道格拉斯方程进行计算，因此每一个社会成本都是0-1的数值。

        当地表水用量超过配额时，视为一个违规行为。
        违规会导致社会成本的增加，但是这个增加是一个非线性的过程。
        同样违反规则的决策，产生的社会成本取决于周围相比较的人的行为：

        1. 如果周围人都是遵守规则的人，那么违规的决策会导致更大的声誉损失
        2. 如果周围人都是违规的人，那么合作的决策会导致自己的社会不满，
        这意味着违反规则反而形成了一种社会风气，反而不会遭受损失。

        Parameters:
            q_surface:
                地表水用量，单位：亿立方米 (1e8 m³)。
                这个数值将被用来和配额进行比较，以判断是否违规。

        Returns:
            社会成本，一个0-1之间的数值。
            是声誉损失和社会不满的等权平均值。
        """
        # 根据地表水可能的用量和配额比较，判断是否违规
        # q_surface 和 self.quota 都是亿立方米 (1e8 m³)
        willing: DecisionType = "D" if q_surface > self.quota else "C"
        dislikes, criticized = self.judge_friends(willing=willing)
        s_enforcement_cost = self.p.get("s_enforcement_cost", 0.5)
        s_reputation = self.p.get("s_reputation", 0.5)
        return lost_reputation(
            s_enforcement_cost,
            s_reputation,
            criticized,
            dislikes,
        )

    def calc_payoff(
        self,
        crop_yield: Dict[str, float],
        q_surface: float,
        q_ground: float,
        water_prices: Optional[dict] = None,
        crop_prices: Optional[dict] = None,
        **kwargs,
    ) -> float:
        """聚合自己的综合得分，取决于自己经济水平和社会水平。

        Parameters:
            crop_yield:
                作物产量，单位是吨/公顷。
            q_surface:
                地表水用量，单位是亿立方米 (1e8 m³)。
            q_ground:
                地下水用量，单位是亿立方米 (1e8 m³)。
            water_prices:
                水资源价格，一个字典，包括地表水和地下水的价格。
            crop_prices:
                作物价格，一个字典，包括作物的价格。

        Returns:
            综合得分，一个0-1之间的数值，代表主体的最后综合收益。
        """
        e = economic_payoff(
            q_surface=q_surface,
            q_ground=q_ground,
            crop_yield=crop_yield,
            water_prices=water_prices,
            crop_prices=crop_prices,
            area=self.irr_area,
            unit="1e8m3",
        )
        s = self.calc_social_costs(q_surface=q_surface)
        return self.agg_payoff(e=e, s=s, include_s=self.include_s, **kwargs)

    def agg_payoff(
        self,
        e: float,
        s: float,
        record: bool = False,
        include_s: bool = True,
        rank: Optional[bool] = False,
    ) -> float:
        """计算最终的综合得分。

        Parameters:
            e:
                经济得分，取值[0~inf)。
            s:
                社会得分，取值[0, 1]。
            record:
                是否记录得分为属性。
                当在遗传算法模拟时，不需要记录得分，
                只需要返回一个当前模拟的结果就好。
                但在模拟完成后，对最优的决策单元进行评估时，
                需要记录得分，以便后续的分析。
            include_s:
                是否将社会得分的考虑包含在最终得分中。
                如果为 False，那么最终得分就是经济得分。
            rank:
                是否将得分转化为排名。
                当在遗传算法模拟时，需要转化为排名。
                在模拟完成后，对最优的决策单元进行评估时，
                不需要转化为排名，而是对真实的得分进行记录。

        Returns:
            综合得分，一个0-1之间的数值，代表主体的最后综合收益。
        """
        # 将经济得分和社会得分转化为在朋友之间的排名
        if rank:
            e = self.compare("e", my=e)
            s = self.compare("s", my=s)
        if include_s:
            payoff = e * s
        else:
            payoff = e
        if record:
            self.e = e
            self.s = s
            self.payoff = payoff
        return payoff

    def make_decision(self) -> DecisionType:
        """
        根据该主体的大胆程度（boldness）随机采取一个决策。
        大胆程度就是随机出`D`决策的概率。
        """
        return "D" if self.random.random() < self.boldness else "C"

    def mutate_strategy(self, probability: float) -> None:
        """随机更新自己的策略。

        为避免陷入局部最优，
        以小概率对关键属性`boldness`与`vengefulness`进行完全的随机。
        如果随机数小于给定的概率阈值，那么就会以各50%的概率触发其中一个属性，
        对所选的属性进行随机（0-1之间）。

        Parameters:
            probability:
                重新随机的概率，应该在0-1之间。
        """
        if self.random.random() > probability:
            return
        if self.random.random() < 0.5:
            self.boldness = self.random.random()
        else:
            self.vengefulness = self.random.random()

    def hate_a_behave(self, behave: DecisionType) -> bool:
        """是否讨厌一个行为，从而产生社会不满。
        判断标准如下：
        1. 自己是违反规则的人，不会讨厌别人。
        2. 对方是遵守规则的人，不会讨厌。
        3. 对方是违反规则的人，但自己可能因为报复心不强而睁一只眼闭一只眼。

        Parameters:
            behave:
                对方的行为。

        Returns:
            是否讨厌这个行为。
        """
        # 自己就是违反规则的人，没资格说别人
        if self.decision == "D":
            return False
        # 对方是遵守规则的人，不会被批评
        if behave == "C":
            return False
        # 对方是违反规则的人，但自己可能睁一只眼闭一只眼
        return self.random.random() <= self.vengefulness

    def judge_friends(self, willing: DecisionType) -> Tuple[int, int]:
        """评价自己的朋友，是不是喜欢它们。

        这个方法来源于前人发表在 Nature Human Behavior 上的一篇文献 [@castillarho2017a]，其核心思想也是基于多元文化理论。
        如果主体发现周围人在违反规则，但自己是遵守规则的，就会觉得社会不公。

        如果主体自己的属性 “vengefulness” （报复心）很强，
        他会讨厌这个“不守规矩”的朋友，
        从而降低自己和那个那个被讨厌的朋友在这个社会中的幸福感。

        Parameters:
            willing:
                自己的决策倾向。
                在遗传算法中，这个参数时一个临时的决定意向，
                如果该意向会给自己带来更多的收益，那么就会倾向于采取这个意向。

        Returns:
            讨厌的朋友数量（影响不满程度），
            以及被批评的朋友数量（影响声誉评估）。
        """
        # 和每一个朋友进行评判
        dislikes, criticized = 0, 0
        for friend in self.friends:
            dislikes += self.hate_a_behave(friend.decision)
            criticized += friend.hate_a_behave(willing)
        return dislikes, criticized

    def change_mind(self, metric: str, how: str) -> bool:
        """由于主体最后的表现不同，主体之间可能存在强化学习行为。

        从表现比较好的朋友处学习。

        Parameters:
            metric:
                评价自己或者同伴表现是否优异的关键指标。
                在这个模型中，我们主要有三种潜在的指标：
                1. `e`: 经济得分。即抛除用水成本之后灌溉得到的经济收益。
                2. `s`: 社会得分。即所有农民在评价朋友后得到的社会满意度。
                3. `payoff`: 综合得分。即经济得分和社会得分的乘积。
            how:
                如果有多个比自己表现优异的其他主体，如何从他们身上学习？
                1. 如果`how='best'`，仅从表现最好的主体身上学习。
                2. 如果`how=random`，可以从任何一个比自己表现好的个体身上学习。

        Returns:
            这个主体是否进行了属性的变更。
        """
        better_friends = self.friends.better(metric=metric, than=self)
        # 如果没有比自己表现更好的，则直接返回 False
        if not better_friends:
            return False
        elif how == "best":
            friend = better_friends.better(metric=metric).random.choice()
        elif how == "random":
            friend = better_friends.random.choice()
        else:
            raise ValueError(f"Invalid how parameter: {how}")
        # 存在比自己表现更好的主体，向他学习。
        self.boldness = friend.boldness
        self.vengefulness = friend.vengefulness
        return True

    def decide_boundaries(
        self,
        seasonal_irr: float,
    ) -> float:
        """决定取用水资源灌溉的下界/上界。

        如果该主体意向是选择了遵守制度，那么用水上限就是水资源配额。
        如果该主体选择了违背制度，那么上限就是今年所有的灌溉量。

        Parameters:
            seasonal_irr:
                本年度的灌溉量，单位：亿立方米 (1e8 m³)。

        Returns:
            包括用水下界（0.0）和上界的元组，单位：亿立方米 (1e8 m³)。
        """
        if self.willing == "C":
            ub = min(self.quota, seasonal_irr)
        else:
            ub = seasonal_irr
        return 0.0, ub

    def irrigating(
        self,
        seasonal_irr: Optional[float] = None,
        water_prices: Optional[dict] = None,
        crop_prices: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        """农民开始触发一次灌溉。
        1. 决定这次灌溉的上下界。
        2. 利用遗传算法，根据地表/地下用水量不同带来的收益差距，估计用水的来源。
        3. 针对最优的地表/地下用水组合，最后计算并记录当年的得分。

        Parameters:
            seasonal_irr:
                本年度的灌溉量。
            water_prices:
                水资源价格，一个字典，包括地表水和地下水的价格。
            crop_prices:
                作物价格，一个字典，包括作物的价格。

        Returns:
            本次灌溉的地表水用量和地下水用量。
        """
        if seasonal_irr is None:
            seasonal_irr = self.total_wu
        boundaries = self.decide_boundaries(seasonal_irr)
        if boundaries[1] <= 0.0:
            self.surface_water = 0.0
            self.ground_water = 0.0
            return 0.0, 0.0
        opt_surface, opt_ground = self.water_withdraw(
            ufunc=self.calc_payoff,
            surface_boundaries=boundaries,
            total_irrigation=seasonal_irr,
            water_prices=water_prices,
            crop_yield=self.dry_yield,
            **kwargs,
        )
        # 用优化过的用水量计算收益，不进行排名，存储结果
        self.calc_payoff(
            crop_yield=self.dry_yield,
            q_surface=opt_surface,
            q_ground=opt_ground,
            water_prices=water_prices,
            crop_prices=crop_prices,
            rank=False,
            record=True,
        )
        return opt_surface, opt_ground

    def step(self) -> None:
        """城市主体的一次行动，包含以下步骤：

        1. 获取当年的天气数据。
        2. 根据天气数据模拟作物的生长。
        3. 根据作物的生长情况，计算灌溉用水量。
        4. 根据水资源价格和作物价格，计算灌溉用水的来源。
        5. 记录地表水和地下水的用量。
        6. 学习表现更好的人，并有一定概率改变策略，决定下一年的策略。
        7. 决定下一年的策略意愿。
        """
        water_prices = self.water_prices
        crop_prices = self.crop_prices
        # 灌溉看水从哪来
        # total_wu is in 1e8 m^3; irrigating() returns (surface, ground) in 1e8 m^3
        sw, gw = self.irrigating(self.total_wu, water_prices, crop_prices)
        # Record volumes in 1e8 m^3 for consistency with quota
        self.surface_water = sw  # 1e8 m^3
        self.ground_water = gw  # 1e8 m^3
        self.simulate(repeats=self.p.get("repeats", 1))  # 农作物就是需要这么多水
        # 学习表现更好的人，并有一定概率改变策略，决定下一年的策略
        self.change_mind(metric="payoff", how="random")
        self.mutate_strategy(probability=self.p["mutation_rate"])
        self.make_decision()
