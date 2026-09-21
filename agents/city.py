#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import math
import warnings
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Self,
    Tuple,
    TypeAlias,
)

import pandas as pd
from abses import ActorsList
from aquacrop import Crop, InitialWaterContent, IrrigationManagement
from aquacrop.core import AquaCropModel
from aquacrop.entities.soil import Soil
from aquacrop_abses.cell import get_crop_datetime
from aquacrop_abses.farmer import Farmer
from aquacrop_abses.load_datasets import crop_name_to_crop

from ..core import (
    aggregate_utility,
    economic_payoff,
    gross_revenue,
    reports_defector,
    social_standing,
    update_city_csv,
)
from ..core.algorithms import require_one_of, require_unit_interval
from ..core.allocation import solve_surface_share
from ..core.culture import (
    GRID_LEVELS,
    grid_from_tightness,
    group_from_index,
    load_city_grid_z,
    load_city_z,
)
from ..core.data_loaders import convert_ha_mm_to_1e8m3
from .province import Province

DecisionType: TypeAlias = Literal["D", "C"]

REQUIRED_COLS = ["MinTemp", "MaxTemp", "Precipitation", "ReferenceET", "Date"]


def validate_policy_years(forced_since: int, include_s_since: int) -> None:
    """Reject a scenario this study does not use.

    `City.willing` tests `include_s_since` before `forced_since`, so in years
    below `include_s_since` the enforcement branch is unreachable. Whether that
    swallows enforcement *entirely* depends on the simulation window — with the
    1980-2012 window of this study and `include_s_since` beyond 2012, a scenario
    asking for enforced compliance runs as "never" instead (issue #59).

    The three published scenarios all satisfy `forced_since >= include_s_since`,
    so rather than reorder the guards to support a combination nobody wants,
    this refuses it.

    Args:
        forced_since: Year from which compliance is mandatory.
        include_s_since: Year from which the social term enters the payoff.

    Raises:
        ValueError: If `forced_since < include_s_since`.
    """
    if forced_since < include_s_since:
        raise ValueError(
            f"City.forced_since ({forced_since}) precedes City.include_s_since "
            f"({include_s_since}). Enforced compliance cannot begin before the "
            "social term does: every year below include_s_since ignores "
            "forced_since, so within this study's 1980-2012 window the run "
            "would silently behave as the 'never' scenario. Use one of the "
            "three scenarios this model supports: strict "
            "(forced_since=include_s_since=1998), baseline "
            "(forced_since=2020, include_s_since=1998), or never "
            "(forced_since=include_s_since=2020)."
        )


def to_regional_crop(crop: str, province: Optional[str] = None) -> str:
    """Convert crop name to regional crop variant.

    This function maps generic crop names to region-specific variants used
    by the AquaCrop model. Regional variants account for local growing
    conditions and seasonal patterns.

    Args:
        crop: Generic crop name (e.g., "Maize", "Rice", "Wheat").
        province: Optional province name for wheat season determination.
            Required for wheat to determine winter vs spring wheat.

    Returns:
        Regional crop name string. For maize and rice, returns "RegionalMaize"
        or "RegionalRice". For wheat, calls decide_wheat_season() to determine
        the appropriate variant.

    Example:
        ```python
        # Maize always becomes RegionalMaize
        to_regional_crop("Maize")  # Returns "RegionalMaize"

        # Wheat depends on province
        to_regional_crop("Wheat", "Henan")  # Returns "RegionalWheat"
        to_regional_crop("Wheat", "Gansu")  # Returns "Spring_Wheat"
        ```
    """
    if crop in ["Maize", "Rice"]:
        return f"Regional{crop}"
    if crop == "Wheat":
        return decide_wheat_season(crop, province)
    return crop


def decide_wheat_season(crop: str, province: Optional[str] = None) -> Crop:
    """Determine wheat season variant based on province location.

    Wheat growing seasons vary by region in China. Northern provinces
    (Henan, Shandong, Shanxi, Shaanxi) typically grow winter wheat
    (RegionalWheat), while other regions grow spring wheat.

    Args:
        crop: Crop name, should be "Wheat" for this function to have effect.
        province: Province name for determining wheat season. If None or
            not in the winter wheat region list, returns "Spring_Wheat".

    Returns:
        Crop object or string. Returns "RegionalWheat" for winter wheat
        regions, "Spring_Wheat" for others, or the original crop if not wheat.

    Note:
        The winter wheat provinces list is based on typical agricultural
        practices in the Yellow River Basin.
    """
    if crop != "Wheat":
        return crop
    if province in ["Henan", "Shandong", "Shanxi", "Shaanxi"]:
        return "RegionalWheat"
    return "Spring_Wheat"


class City(Farmer):
    """City-level agent representing an irrigation unit in the Yellow River Basin.

    The City class extends the Farmer agent from AquaCrop-abses to represent
    city-level irrigation decision-making units. Each city agent makes annual
    decisions about water use, balancing economic benefits (crop yields minus
    water costs) with social factors (reputation, peer pressure).

    The agent integrates several modeling components:
        1. **Crop Simulation**: Uses AquaCrop to simulate crop yields based on:
           - Soil type (loam, clay, sand, etc.)
           - Crop type (rice, wheat, maize)
           - Daily climate data (temperature, precipitation, evapotranspiration)
           - Irrigation management (amount, frequency, method)
        2. **Water Source Optimization**: Uses genetic algorithms to optimize
           the allocation between surface water and groundwater based on
           economic payoffs
        3. **Social Learning**: Implements social learning mechanisms where
           agents observe and learn from better-performing neighbors

    Attributes:
        irr_area (pd.Series): Irrigated area per crop in hectares. Indexed by
            crop names ("Maize", "Wheat", "Rice").
        irr_method (int): AquaCrop irrigation strategy, taken from the
            `irr_method` parameter (1 = soil moisture targets, the setting
            this study uses). Not an efficiency ranking — see
            `aquacrop_abses.farmer.Farmer.irr_methods` for the codes.
        quota (float): Water quota allocated to this city in units of
            1e8 m³ (100 million cubic meters).
        surface_water (float): Annual surface water use in 1e8 m³.
        ground_water (float): Annual groundwater use in 1e8 m³.
        boldness (float): Agent's propensity to violate water quota rules.
            Range [0, 1], where higher values indicate greater willingness
            to exceed quota. Initialized randomly.
        vengefulness (float): Agent's tendency to criticize rule-violating
            neighbors. Range [0, 1], where higher values indicate stricter
            enforcement of social norms. Initialized randomly.
        willing (DecisionType): Current decision tendency ("C" for comply,
            "D" for defect). May be overridden by policy enforcement in
            certain years.
        e (float): Economic score, representing net economic benefit from
            irrigation (crop revenue minus water costs). Range [0, inf).
        s (float): Social score, representing social satisfaction based on
            peer evaluations. Range [0, 1], where 1.0 indicates highest
            satisfaction.
        payoff (float): Combined score calculated as e * s (when social
            factors are included) or just e (when only economic factors
            are considered). Used for ranking and learning.

    Decision Making:
        The agent's decision process involves:
            1. Determining irrigation needs based on crop simulation
            2. Optimizing water source allocation (surface vs. groundwater)
            3. Evaluating economic and social payoffs
            4. Learning from better-performing neighbors
            5. Updating behavioral parameters (boldness, vengefulness)

    Social Network:
        Cities are connected through a social network ("friends") that
        represents information sharing and peer influence. Agents compare
        their performance with friends and may adopt successful strategies.

    Example:
        Access city properties and methods:

        ```python
        # Get a city agent
        city = model.sel_city(city_id=102)

        # Access water use data
        print(f"Quota: {city.quota} 1e8 m³")
        print(f"Surface water: {city.surface_water} 1e8 m³")
        print(f"Decision: {city.decision}")

        # Simulate crop yields
        yields = city.simulate(crop="Maize")

        # Get social network
        friends = city.friends
        ```

    Note:
        The City agent inherits from Farmer, which provides crop simulation
        capabilities through AquaCrop. The social learning and decision-making
        components are specific to this water quota model.

    See Also:
        - `cwatqim.agents.province.Province`: Province agents that allocate quotas
        - `aquacrop_abses.farmer.Farmer`: Base farmer class with crop simulation
    """

    valid_decisions: Dict[DecisionType, str] = {
        "C": "Cooperate: compliance with water quota.",
        "D": "Defect: use more water than quota.",
    }

    #: 已发布过、随后删除的公开方法，以及它们的去向。`cwatqim` 带 DOI（见
    #: `.zenodo.json`、`sync-public-repo.yml`），外部引用断了是引用不到 issue 的，
    #: 所以删名字要出声。语义**没有对应物**（举报从伯努利抽签变成了确定性阈值规则），
    #: 所以给桩而不是垫片。
    #:
    #: 只列**真正公开过**的名字：`v0.1.6` 里有 `hate_a_behave`，没有
    #: `draw_judgements`（它只在内部分支上活过）。为没发布过的名字留桩，会让读者
    #: 以为自己手上的旧版本有它，然后去翻一段不存在的历史。
    _REMOVED_METHODS: Dict[str, str] = {
        "hate_a_behave": (
            "举报不再是抽签。旧的 `hate_a_behave(behave)` 内部掷一枚伯努利硬币；"
            "现在规则是确定性的 `v > 1 − grid`，一条边的两端因此必然给出同样的判断"
            "（见 issue #72、#110）。改用 `City.will_report(behave, my_decision)` ——"
            "**多一个必填参数**，因为谁批评谁取决于双方的决定；或用纯函数 "
            "`cwatqim.core.payoff.reports_defector(vengefulness, enforcement_cost)`。"
        ),
    }

    def __getattr__(self, name: str):
        """Dynamic attribute access for crop yield properties.

        This method enables convenient access to crop yields using attribute
        notation. For example, `city.dry_yield_maize` will return the dry
        yield for maize.

        Supported patterns:
            - `dry_yield_{crop}`: Returns dry yield for the specified crop
            - `yield_potential_{crop}`: Returns potential yield for the crop

        It also answers for the public methods removed in v0.2, so that an
        external caller gets an explanation instead of a bare name error
        (see `_REMOVED_METHODS` and issue #130).

        Warning:
            This is the **only** `__getattr__` on `City`. Defining a second one
            anywhere in the class silently replaces this one, and the crop
            yields above go NaN for a whole run without a single test failing
            — the golden test does not cover yields. Add cases here instead.

        Args:
            name: Attribute name following the pattern above.

        Returns:
            Yield value (float) for the specified crop, or raises AttributeError
            if the pattern doesn't match.

        Example:
            ```python
            # Access dry yield for maize
            maize_yield = city.dry_yield_maize

            # Access potential yield for wheat
            wheat_potential = city.yield_potential_wheat
            ```
        """
        if name.startswith("dry_yield"):
            crop_name = name.split("_")[-1].capitalize()
            return self.dry_yield.get(crop_name)
        if name.startswith("yield_potential"):
            crop_name = name.split("_")[-1].capitalize()
            return self.yield_potential.get(crop_name)
        if name in self._REMOVED_METHODS:
            raise AttributeError(
                f"`City.{name}` 已在 v0.2 中删除："
                f"{self._REMOVED_METHODS[name]} 见 issue #130。"
            )
        return super().__getattr__(name)

    @cached_property
    def climate_data(self) -> pd.DataFrame:
        """Load and cache daily climate data for this city.

        This property loads the city's climate data from CSV on first access
        and caches it for subsequent use. The data includes daily temperature,
        precipitation, and evapotranspiration values required for crop
        simulation.

        The data is filtered to include only required columns and the
        ReferenceET values are clipped to a minimum of 0.1 mm to avoid
        numerical issues in crop simulation.

        Returns:
            DataFrame with daily climate data containing columns:
                - Date: Datetime index
                - MinTemp: Minimum temperature (°C)
                - MaxTemp: Maximum temperature (°C)
                - Precipitation: Daily precipitation (mm)
                - ReferenceET: Reference evapotranspiration (mm, clipped >= 0.1)

        Raises:
            ValueError: If `city_climate_dir` is not configured in the dataset.
            FileNotFoundError: If the climate data file for this city does
                not exist.

        Note:
            This property uses `@cached_property` to ensure the data is loaded
            only once, even if accessed multiple times. This avoids loading
            during setup when City_ID may not yet be assigned.
        """
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
        """Get the city's unique identifier.

        This property returns the City_ID value that was loaded from the
        shapefile when the city agent was created. This identifier is used
        for:
            - Loading city-specific data files (climate, irrigation area, etc.)
            - Identifying cities in analysis and visualization
            - Linking cities to external datasets

        Note:
            This is different from the ABSESpy internal `unique_id`, which
            is a framework-generated identifier. The `city_id` is the
            user-defined identifier from the spatial data.

        Returns:
            Integer city ID, or None if the City_ID attribute has not been
            set (e.g., during initialization before shapefile loading).

        Example:
            ```python
            city_id = city.city_id  # e.g., 102
            city_name = city.city_name  # e.g., "C102"
            ```
        """
        return getattr(self, "City_ID", None)

    @property
    def city_name(self) -> str:
        """Get a formatted string representation of the city ID.

        Returns:
            String in the format "C{city_id}", e.g., "C102" for city ID 102.
            This format is commonly used in data file naming conventions.
        """
        return f"C{self.city_id}"

    @cached_property
    def collectivism_z(self) -> float:
        """This city's standardised prefecture-level collectivism index.

        A `@cached_property` rather than something set in `setup()` for the
        same reason as `climate`: `City_ID` is injected by `new_from_gdf`
        *after* `setup()` runs, so `self.city_id` is still None in there.

        Returns:
            The standardised index for the wave named by
            `City.s_group_wave`.

        Raises:
            KeyError: If this city is absent from the table. Deliberately
                loud — a silent fallback would let a partial table shift the
                whole basin's calibration without anyone noticing.
        """
        wave = int(self.p.get("s_group_wave", 2000))
        table = load_city_z(str(self.ds.city_collectivism), wave)
        try:
            return table[self.city_id]
        except KeyError as err:
            raise KeyError(
                f"{self.city_name} 不在集体主义指数表里（{self.ds.city_collectivism}，"
                f"{wave} 波次）。重新生成：python scripts/build_city_collectivism.py"
            ) from err

    @property
    def s_grid_level(self) -> str:
        """Which resolution `City.s_grid` is calibrated at.

        The single place `City.s_grid_level` is read and validated. Two callers
        need it — `tightness_z` to pick the column, `s_grid` to decide whether
        to short-circuit — and each used to read `self.p.get(...)` with its own
        copy of the default. That is exactly the duplicated-magic-default shape
        `s_grid` itself carries a docstring about removing (#129); it grew back
        on the new key.

        Returns:
            The configured level, one of `GRID_LEVELS`; `"national"` if unset.

        Raises:
            ValueError: The configured level is not one of `GRID_LEVELS`.
                Falling back silently would be worse than failing: the three
                levels are different calibrations, and a typo would hand back
                a complete set of results for the one nobody asked for.
        """
        return require_one_of(
            "City.s_grid_level",
            str(self.p.get("s_grid_level", "national")),
            GRID_LEVELS,
        )

    @cached_property
    def tightness_z(self) -> float:
        """This city's standardised cultural-tightness index.

        A `@cached_property` for the same reason as `collectivism_z`:
        `City_ID` is injected by `new_from_gdf` *after* `setup()` runs.

        Returns:
            The index at the resolution named by `City.s_grid_level`
            —— the province's own value at `province`, that value plus this
            city's draw at `prefecture`.

        Raises:
            KeyError: If this city is absent from the table. Deliberately
                loud, exactly as on the Group side: a silent fallback would
                let a partial table shift the basin's calibration unnoticed.
        """
        level = self.s_grid_level
        table = load_city_grid_z(
            str(self.ds.city_tightness),
            level=level,
            spread=str(self.p.get("s_grid_spread", "se")),
            seed=int(self.p.get("s_grid_draw_seed", 0)),
        )
        try:
            return table[self.city_id]
        except KeyError as err:
            raise KeyError(
                f"{self.city_name} 不在文化紧密度表里（{self.ds.city_tightness}，"
                f"{level} 级）。重新生成：python scripts/build_city_tightness.py"
            ) from err

    @property
    def s_grid(self) -> float:
        """原文式(3) 的 Grid：举报一个违规邻居之后**留下**的 goodwill 份额。

        配置值就是原文符号，不再取补（#210）。举报一次的**代价**因此是
        `1 − grid`，整个模型里唯一的 `1 −` 只有 `social_standing` 里的
        `(1 − group)^n`。

        它在两处进入模型，读的必须是同一个值：`will_report` 用它做举报门槛
        （`v > 1 − grid`），`standing_by_decision` 用它算已举报者剩下的
        goodwill（`grid^m`）。原先两处各写一遍 `self.p.get(...)`，魔数默认值
        重复且无人校验（issue #129）。

        与 Group 侧同一条映射，但多一个分辨率开关，因为紧密度只到省级：
        `City.s_grid_level` 取 `national`（全国标量）、`province`（逐省）或
        `prefecture`（逐省 + 省均值不确定性的抽样）。κ 的**符号**是构念选择，
        两种读法方向相反，见 `core.culture.grid_from_tightness` 的 Warning。

        Returns:
            `national` 或 κ=0 时是配置键 `City.s_grid`（缺省 0.5），否则是
            `grid_from_tightness` 给出的逐城取值。

        Raises:
            ValueError: grid 非有限、落在 [0, 1] 之外，或 `s_grid_level` 不在
                `GRID_LEVELS` 里。越界会让 `grid^m` 给出负的或大于 1 的
                goodwill，社会项就此失去意义；非有限值要显式拒绝，因为
                `nan <= 1` 是 False，区间守卫拦不住它（同一教训见
                `core.culture`）。

        Note:
            `national` 与 κ=0 两个分支都在任何文件 I/O **之前**返回，所以关掉
            这个开关的运行既不需要紧密度表，也与接入之前的模型逐比特相同
            —— 与 `s_group` 的 κ=0 短路是同一条约定，golden 指纹依赖它。
        """
        base = require_unit_interval("City.s_grid", float(self.p.get("s_grid", 0.5)))
        level = self.s_grid_level
        kappa = float(self.p.get("s_grid_kappa", 0.0))
        if level == "national" or kappa == 0.0:
            return base
        return grid_from_tightness(base=base, kappa=kappa, z=self.tightness_z)

    @property
    def s_group(self) -> float:
        """The Group parameter actually used by this city's social term.

        The source's `group`, used as it is: the share of standing **lost**
        each time a neighbour criticises this city. It is complemented exactly
        once, inside `social_standing`'s `(1 - group)^n` (#210).

        Two things share this name and they are not the same: the **config**
        key `City.s_group` is the national scalar (`group_bar`), while this
        **property** is the per-city value derived from it.
        `standing_by_decision` wants the latter.

        Returns:
            `group_bar` itself when `s_group_kappa` is 0, otherwise the
            per-city value from `group_from_index`.

        Note:
            The `kappa == 0.0` branch returns before any file I/O, so a run
            with culture switched off neither needs the index table nor
            differs by a single bit from the pre-culture model. That exact
            comparison is load-bearing: `base + 0.0 * z` would return NaN for
            a NaN `z` instead of `base`, and `cobb_douglas` would not catch it.
        """
        base = float(self.p.get("s_group", 0.5))
        kappa = float(self.p.get("s_group_kappa", 0.0))
        if kappa == 0.0:
            return base
        return group_from_index(base=base, kappa=kappa, z=self.collectivism_z)

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
    def total_withdrawal(self) -> float:
        """Surface water plus groundwater withdrawn this year, in 1e8 m³.

        The single definition of "how much was withdrawn", and therefore the
        single place that decides when the surface/groundwater *share* is
        undefined. Both `surface_ratio` and `calc_max_irr_seasonal` divide by
        it, and each answers the zero case in its own terms (see issue #66).
        """
        return self.surface_water + self.ground_water

    @property
    def surface_ratio(self) -> float:
        """Surface water ratio (0-1). For data collection.

        Calculated as: surface_water / total_withdrawal. Withdrawing nothing
        leaves the ratio undefined; 0.0 is reported as the display default.
        """
        if self.total_withdrawal <= 0:
            return 0.0
        return self.surface_water / self.total_withdrawal

    @property
    def economic_position(self) -> float:
        """Where this year's economic payoff sits among friends, in [0, 1].

        A **min-max normalisation** within the friend set, not a rank: see
        `compare`. Purely diagnostic — no decision rule reads it.

        Reads only recorded state, so collecting it cannot disturb either
        random stream.
        """
        return self.compare("e")

    @property
    def social_position(self) -> float:
        """Where this year's social standing sits among friends, in [0, 1].

        Same min-max caveat as `economic_position`, plus a sharper one: it
        saturates. `social_standing` is 1.0 whenever nobody criticised and
        nobody was criticised, so under enforced compliance a whole province
        can share the maximum and every city then reports 1.0. Read a value
        of 1.0 as "not below anyone here", never as "best" (see issue #70).
        """
        return self.compare("s")

    @property
    def unit_payoff(self) -> float:
        """单位毛收入的效用 —— 社会学习**实际**比较的量。

        `change_mind` 复制的是"比我强"的邻居的性状。用**绝对** payoff 比较时，
        实际规则是"模仿最大的那个城市"：省内经济收益的极差中位数是 11 倍，而社会项
        只是一个 `[0, 1]` 的折扣，几乎不可能改变谁排在前面（实测省内-年 `e·s` 与
        `e` 的序相关中位数是 1.0000）。性状因此按灌溉面积而非行为被选择，
        末年每个省只剩一个 boldness 取值，省均 boldness 与省级违规意图率的相关是
        0.992 —— 模型的自由度塌缩到每省一个初始随机数（issue #110 §2.5、§2.6）。

        除以毛收入让比较与城市规模无关，选择才由行为决定。

        Returns:
            `payoff / revenue`；没有收成、或毛收入非有限时返回 `0.0`。

            **`0.0` 是一个被接受的占位，不是中性值**（issue #133，判定为保留现状）。
            `revenue = 0` 时这个比值是 0/0、数学上无定义。

            `City.payoff_floor` 已于 2026-09-01 删除（#213），效用回到 `e·s`，
            所以亏损主体的 `unit_payoff` 重新是**负数**、`0.0` 这个哨兵重新赢过
            它们——#133 记的 0.84% 共现回到窗口内外都可能出现。重跑后要复核这个数。

            作者判定这在模型语义上可接受：绝收者仍可作为模仿对象。若日后要改，
            哨兵值应落在所有真实取值**之下**（`-inf`），或让 `change_mind` 用
            `ActorsList.select` 把无收成主体排除出候选集——并补一条断言**序**、
            而不只是断言取值的测试。

            非有限要一起挡住：`nan <= 0` 是 False，放过去会让 `unit_payoff` 变成
            NaN，而 `change_mind` 里 `nan > x` 恒为 False —— 主体从此静默地再也
            学不到东西（同一教训见 `core.culture`）。

        Note:
            量纲是"每元毛收入的效用"，因此天然落在 1 附近，可跨主体、跨年份比较。
        """
        if not math.isfinite(self.revenue) or self.revenue <= 0:
            return 0.0
        return self.payoff / self.revenue

    @property
    def crop_here(self) -> List[str]:
        """Get list of crops currently grown in this city.

        Returns:
            List of crop names (strings) that have irrigated area > 0 in
            this city. Typically contains "Maize", "Wheat", and/or "Rice".
        """
        return self.irr_area.index.to_list()

    @property
    def irr_area(self) -> pd.Series:
        """Get irrigated area per crop in hectares.

        Returns:
            Pandas Series with crop names as index and irrigated areas
            (ha) as values. Only crops with area > 0 are included.

        Note:
            Irrigated area is stored per-crop as hectare (ha).
            When multiplied by a water depth in mm, use the conversion:
            ha * mm -> m^3 via factor 10 (1 ha = 10,000 m^2; 1 mm = 0.001 m).
        """
        return self.yearly_dynamic("irr_area")

    @property
    def total_area(self) -> float:
        """Get total irrigated area across all crops in hectares.

        Returns:
            Sum of all irrigated areas in hectares (ha).
        """
        return self.irr_area.sum()

    @property
    def water_used(self) -> pd.Series:
        """Get water use per crop in units of 1e8 m³.

        Returns:
            Pandas Series with crop names as index and water use volumes
            (1e8 m³) as values. Calculated as irrigated area (ha) multiplied
            by water use intensity (mm) and converted to 1e8 m³.

        Implementation detail:
            Convert ha*mm to 1e8 m^3. The helper uses factor 10 to get m^3
            and then divides by 1e8 to yield 1e8 m^3.
        """
        return convert_ha_mm_to_1e8m3(self.irr_area * self.wui)

    @property
    def total_wu(self) -> float:
        """Get total water use across all crops in 1e8 m³.

        Returns:
            Sum of water use for all crops in units of 1e8 m³ (100 million
            cubic meters).
        """
        return self.water_used.sum()

    @property
    def net_irr(self) -> float:
        """Get total net irrigation volume in 1e8 m³.

        Returns:
            Total net irrigation volume across all crops in units of 1e8 m³.
            Calculated from seasonal irrigation depth (mm) multiplied by
            irrigated area (ha) and converted to 1e8 m³.

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
        """Get the province agent that manages this city.

        Returns:
            Province agent instance. Returns None if not yet set during
            initialization.

        Note:
            When setting this property, you can provide either:
            - A Province agent instance that has already been created
            - A province name string (English name), which will create or
              retrieve the province using the singleton pattern
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
        """Get water use intensity (WUI) per crop in millimeters.

        Water use intensity represents the annual quantity of water withdrawn
        for irrigation per unit area, including losses during conveyance and
        field application. This is empirical data loaded from statistics,
        not simulated by the model.

        Returns:
            Pandas Series with crop names as index and WUI values (mm) as
            values. The WUI represents the depth of water applied per hectare
            of irrigated area.

        Note:
            Irrigation water use is the annual quantity of water withdrawn
            for irrigation including the losses during conveyance and field
            application. This data comes from statistics rather than model
            simulation.
        """
        return self.yearly_dynamic("wui")

    @property
    def quota(self) -> float:
        """Get water quota allocated to this city in 1e8 m³.

        The water quota represents the maximum allowable surface water use
        for this city. It is allocated by the province based on irrigated
        area and stored in units consistent with surface_water and
        ground_water for easy comparison.

        Returns:
            Water quota in units of 1e8 m³ (100 million cubic meters).

        Note:
            Internal storage:
            _quota stores raw volume in m^3; getter converts to 1e8 m^3.
        """
        return self._quota / 1e8  # Convert from m³ to 1e8 m³

    @quota.setter
    def quota(self, volume_m3: float) -> None:
        """Set water quota for this city.

        Args:
            volume_m3: Quota volume in cubic meters (m³). The value is stored
                internally in m³ and converted to 1e8 m³ when accessed via the
                property getter.
        """
        self._quota = float(volume_m3)

    @property
    def decision(self) -> DecisionType:
        """Get the actual water use decision based on quota compliance.

        The decision is determined by comparing actual surface water use
        against the allocated quota:
            - "D" (Defect): If surface water use exceeds quota (violation)
            - "C" (Cooperate): If surface water use is within quota (compliance)

        Returns:
            Decision type: "C" for compliance or "D" for defect.

        Note:
            Units: Both surface_water and quota are in 1e8 m³ (100 million
            cubic meters) for consistent comparison.
        """
        return self.decide(self.surface_water)

    @property
    def last_decision(self) -> DecisionType:
        """去年已实现的守约与否——**举报资格与被观察行为**的共同依据。

        `judge_friends` 的两个方向都只读这一列：数"我批评了谁"时读我自己的和邻居
        的 `last_decision`，数"谁批评我"时读邻居的。当年意图 `willing` 一度被试着
        用在被观察的那一侧（`multirun/eq3_willing`），2026-08-31 撤回——统一在一个
        时间断面上，社会判定才是边的属性而不是遍历顺序的产物（#72、#209）。

        与 `decision` 的区别是时点：`decision` 读 `self.surface_water`，而那个值
        在本年度 `step` 里会被覆盖，且城市是乱序推进的。于是同一次社会判定曾经同时
        混了三个时间断面：自己的资格用去年、被评的候选用今年、朋友的行为则是"已经
        走过的用今年、没走过的用去年"（issue #72 附加问题 2）。

        模型在任何城市 step 之前调 `snapshot_decision` 统一冻结这一列，所以整年里
        所有人读到的都是同一个 t−1 断面。这也是 `judge_friends` 里 `m` 与候选分支
        无关的根源：候选决策属于今年，而资格只看这一列。

        Returns:
            "C" 或 "D"。

        Note:
            这是**同步更新**的约定，要写进 ODD+D：主体观察的是上一年的已实现状态，
            而不是同年内滚动更新的状态。
        """
        return self._last_decision

    def snapshot_decision(self) -> None:
        """把当前已实现的守约与否冻结成本年度的观察集。

        由 `CWatQIModel.step` 在 `cities.shuffle_do("step")` **之前**统一调用。
        不消耗随机数，因此可以按固定顺序执行。
        """
        self._last_decision = self.decision

    def decide(self, q_surface: float) -> DecisionType:
        """Whether a given surface-water use counts as a breach.

        The single definition of "over quota". `decision` applies it to what
        was actually withdrawn; `calc_social_standing` applies it to the
        candidate under evaluation. Same rule, one place (see issue #72).

        Args:
            q_surface: Surface water use in 1e8 m³, actual or candidate.

        Returns:
            "D" when it exceeds the quota, "C" otherwise.
        """
        return "D" if q_surface > self.quota else "C"

    @property
    def water_prices(self) -> Dict[str, float]:
        """Get water prices for surface and groundwater in RMB/m³.

        Returns:
            Dictionary with keys "surface" and "ground" containing water
            prices in RMB per cubic meter (RMB/m³).
        """
        return self.province.water_prices

    @property
    def crop_prices(self) -> Dict[str, float]:
        """Get crop prices in RMB per tonne.

        Returns:
            Dictionary mapping crop names to prices. Original data is in
            RMB/kg but is converted to RMB/t (multiplied by 1000) for
            consistency with yield units (t/ha).
        """
        return self.province.crop_prices

    @property
    def include_s(self) -> bool:
        """Check whether social factors should be included in payoff calculation.

        The threshold is the same `include_s_since` parameter that gates
        `willing`, so both paths share one policy year: before it, agents
        neither form social decisions nor carry the social term in their
        payoff.

        Returns:
            True from `include_s_since` onwards, False before it.

        Raises:
            KeyError: If `include_s_since` is missing from the parameters.

        See Also:
            - `cwatqim.agents.city.City.willing`: The other consumer of
              `include_s_since`
            - `cwatqim.agents.city.City.agg_payoff`: Where the flag selects
              the branch that returns `e` instead of `e · s`
        """
        return self.time.year >= self.p["include_s_since"]

    def setup(self) -> None:
        """Initialize the city agent with dynamic variables and attributes.

        This method is called automatically during model setup to configure
        the city agent with:
            1. **Dynamic Variables**: Time-varying data loaded from CSV files:
               - `wui`: Water use intensity (mm) per crop, varies by year
               - `irr_area`: Irrigated area (ha) per crop, varies by year
            2. **Farm Attributes**: Irrigation method and related settings
            3. **Water Attributes**: Initial values for quota and water use
            4. **Social Attributes**: Random initialization of behavioral
               parameters (boldness, vengefulness) and initial decision
            5. **Score Attributes**: Initial economic and social scores

        The dynamic variables use update functions that extract year-specific
        data from the loaded DataFrames based on the current simulation year.

        Note:
            This method should not be called manually. It is invoked
            automatically by the ABSESpy framework during model initialization.

        Raises:
            FileNotFoundError: If required data files (irr_wui, irr_area_ha)
                are not found in the configured data paths.
            KeyError: If required columns are missing from the data files.

        See Also:
            - `cwatqim.core.data_loaders.update_city_csv`: Function for updating
                city data from CSV files
        """
        # 每个城市每年只读一次 dynamic variable（见 #68）
        self._dynamic_cache: Dict[Tuple[str, int, Optional[int]], Any] = {}
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
        # ===== Water-related attributes =====
        self._quota = 0.0
        self.surface_water = 0.0
        self.ground_water = 0.0
        # ===== Social-related attributes =====
        self.boldness = self.random.random()
        self.vengefulness = self.random.random()
        self.willing = self.make_decision()
        # 上一年已实现的守约与否，模型在任何城市 step 之前统一快照（见 #72）
        self._last_decision: DecisionType = self.decision
        # ===== Score-related attributes =====
        # income: -inf~inf
        # social benefits: 0~1
        self.agg_payoff(
            e=0.0, s=1.0, revenue=0.0, record=True, include_s=self.include_s
        )

    def yearly_dynamic(self, name: str) -> Any:
        """Read a dynamic variable, at most once per model year per city.

        ABSESpy's `dynamic_var` looks like it caches, but the cache never
        hits, and the reason matters: its key is `time.tick`, which resolves
        to `model.steps` — mesa's step counter, not model time — while every
        update function here filters on `time.year`. Nothing appends to
        `_updated_ticks` either, so each read re-runs the update function:
        `inspect.getsource` on it (~200 us), then a full-table mask and copy
        (~285 us). Measured at ~526 us per read (see issue #68).

        That cost lands in two bad places: `calc_max_irr_seasonal` reads
        `wui` once per crop per year, and `calc_payoff` reads `irr_area`
        inside the allocation objective, which the solver evaluates several
        times per agent-year — it was 90% of every candidate evaluation back
        when the solver was a differential evolution (removed in issue #94).

        The key is `(name, year, city_id)` — the full set of inputs
        `update_city_csv` reads. Two of those three are load-bearing:

        - **year**, not tick: `tick` is `model.steps`, so anything that moves
          time outside `step()` (`tests.helper.time_to`, a warm-up, a direct
          `time.to()`) desynchronises them permanently and freezes every city
          on one year's data.
        - **city_id**: `update_city_csv` returns all zeros when it is None
          (`data_loaders.py`), which happens before the shapefile attributes
          are assigned. Without it in the key, one early read would pin a city
          to zeros for the whole year — a silent-zeros failure, the worst kind
          here.

        Note:
            `Province` deliberately does not use this: it reads its one
            dynamic variable once per year already (`update_data`), so a cache
            would buy ~36 ms per run and cost a second copy of this logic.

            Only correct for variables that vary by year. A monthly variable
            would need `time.dt` in the key.

            The cached object is shared by readers of the same city-year.
            Nothing in this model mutates these Series in place — they are
            only read, or combined into new Series — but an in-place edit
            would now be visible to the other readers.

        Args:
            name: Dynamic variable name, as registered in `setup`.

        Returns:
            The variable's value for the current year.
        """
        key = (name, self.time.year, self.city_id)
        if key not in self._dynamic_cache:
            self._dynamic_cache[key] = self.dynamic_var(name)
        return self._dynamic_cache[key]

    def calc_max_irr_seasonal(self, crop: str) -> float:
        """Seasonal cap on irrigation reaching the field, in mm.

        **This is the first of the model's two efficiency layers.** Water
        travels from the source to the crop root zone through two losses,
        applied at different places and by different parameters:

        1. **Conveyance** (here): `wui` is the gross withdrawal per hectare
           taken at the source, from irrigation statistics. Weighting it by
           the provincial coefficients `sw_irr_eff` / `gw_irr_eff` (the
           渠系水利用系数, 0.40-0.89 depending on province and source) gives
           the depth that survives the canal network and arrives at the field.
           That is what this method returns, and what `simulate` hands to
           AquaCrop as `MaxIrrSeason` — a cap on *cumulative applied*
           irrigation.
        2. **Field application**: AquaCrop then applies `AppEff`
           (the `irr_eff` parameter) to each irrigation event; see `simulate`.

        The two layers compose rather than duplicate: water delivered to the
        root zone is roughly `wui * conveyance_eff * AppEff/100`. Reporting
        only one of them understates the losses (see issue #62).

        Because the surface/groundwater mix sets the weighting, the cap — and
        with it the season's water-limited yield — moves with the source
        portfolio. Groundwater has the higher coefficient in every province,
        so substituting groundwater raises the delivered depth for the same
        gross withdrawal. That is the mechanism behind the efficiency results,
        and the reason the coefficients deserve a sensitivity analysis
        (see issue #64).

        Args:
            crop: Crop name ("Maize", "Wheat", or "Rice") for which to
                calculate maximum irrigation.

        Returns:
            Maximum seasonal irrigation depth in millimeters (mm), measured at
            the field boundary — i.e. after conveyance losses, before field
            application losses.

        Formula:
            max_irr = WUI * (sw_ratio * sw_eff + gw_ratio * gw_eff)

            Where:
            - WUI: Gross withdrawal per hectare for the crop (mm)
            - sw_ratio: Surface water proportion
            - gw_ratio: Groundwater proportion
            - sw_eff: Surface water conveyance efficiency (province)
            - gw_eff: Groundwater conveyance efficiency (province)

        Note:
            Withdrawing nothing means irrigating nothing: with
            `total_withdrawal` at zero the share is undefined, and this returns
            0.0 mm so that AquaCrop runs the season rainfed — a value in its
            vocabulary, not a sentinel. Dividing by the zero total instead
            produced a NaN `MaxIrrSeason`, which then tripped the equality
            assertion in `simulate` (`NaN != NaN`) rather than reaching
            AquaCrop (see issue #66).

            The guard comes before reading `wui`: the year's first read still
            goes through a full-table scan (see `yearly_dynamic` and issue
            #68), and on this path it would be wasted.
        """
        total = self.total_withdrawal
        if total <= 0:
            return 0.0
        sw_share = self.surface_water / total
        gw_share = self.ground_water / total
        return self.wui[crop] * (
            sw_share * self.province.sw_irr_eff + gw_share * self.province.gw_irr_eff
        )

    def simulate(self, crop: Optional[Crop] = None, repeats: int = 1) -> pd.DataFrame:
        """Simulate crop growth and yield for one growing season.

        This method runs the AquaCrop model to simulate crop growth based on:
            - Daily climate data (temperature, precipitation, ET)
            - Soil properties (default: loam)
            - Crop type and regional variant
            - Irrigation management strategy
            - Initial soil water content

        The simulation can be run for a single crop or all crops grown in
        the city. When simulating all crops, the results are aggregated into
        a single DataFrame.

        Args:
            crop: Crop name to simulate ("Wheat", "Maize", "Rice"). If None,
                simulates all crops that have irrigated area > 0 in this city.
            repeats: Number of simulation repeats to average. Currently not
                fully implemented - each repeat would use the same conditions.

        Returns:
            DataFrame containing simulation results. For a single crop, returns
            a Series-like row with columns:
                - "Seasonal irrigation (mm)": Total irrigation applied
                - "Yield (tonne/ha)": Crop yield
                - "Seasonal transpiration (mm)": Water transpired
                - Other AquaCrop output variables

            For multiple crops, returns a DataFrame with one row per crop,
            plus an additional column "Irrigation volume (1e8m3)" showing
            total irrigation volume converted to 1e8 m³.

        Note:
            The simulation uses cached climate data for efficiency. The crop
            type is automatically regionalized (e.g., "Wheat" -> "RegionalWheat"
            for winter wheat regions) based on the province location.

        Irrigation efficiency:
            **This is the second of the model's two efficiency layers** — the
            field application one. `calc_max_irr_seasonal` has already taken
            conveyance losses out; `AppEff` (the `irr_eff` parameter, 50)
            handles what is lost between applying water to the field and it
            entering the root zone. AquaCrop uses it twice, and the two uses
            are *not* inverses of each other:

            - when deciding how much to apply, it inflates the request,
              `IrrReq *= ((100 - AppEff) + 100) / 100` — so `AppEff=50` asks
              for 1.5x the root-zone deficit
              (`aquacrop/solution/irrigation.py:173`);
            - when infiltrating, it keeps `Irr * AppEff / 100` — so half of
              what was applied actually reaches the soil
              (`aquacrop/solution/infiltration.py:98`).

            `MaxIrrSeason` caps the *applied* (pre-infiltration) total, so the
            two layers compose: root-zone water is about
            `wui * conveyance_eff * AppEff/100` (see issue #62).

            Note that AquaCrop's inflation factor is not `100/AppEff`: at
            `AppEff=50` it requests 1.5x rather than 2x, so a deficit is not
            fully closed even when the seasonal cap is slack.

        Raises:
            FileNotFoundError: If climate data file is missing.
            ValueError: If crop name is invalid or crop has no irrigated area.

        See Also:
            - `aquacrop.core.AquaCropModel`: The underlying crop simulation model
            - `cwatqim.agents.city.to_regional_crop`: Function for regionalizing crops
            - `cwatqim.agents.city.City.calc_max_irr_seasonal`: Conveyance layer
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

        # 这里原本还有一句 `assert irr_strategy.MaxIrrSeason == calc_max_irr_seasonal(crop)`：
        # 纯函数与自己比，恒真，只是把 `wui` 的整表扫描又付了一遍（见 #66、#68）。
        irr_strategy = IrrigationManagement(
            irrigation_method=self.irr_method,
            SMT=self.p.SMT,
            AppEff=self.p.irr_eff,
            MaxIrrSeason=self.calc_max_irr_seasonal(crop),
        )
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
        kink: Optional[float] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        """Solve the surface/ground split by enumerating the corners.

        The objective is **piecewise affine** in `q_surface`: crop revenue does
        not depend on the split (`crop_yield` is bound before the solve, and
        `simulate` runs afterwards), water cost is linear, and the social term
        steps exactly once at the quota. A piecewise-affine function attains its
        maximum at an endpoint, so the answer is one of at most three points and
        no search is needed — see `cwatqim.core.allocation` and issue #94.

        This replaced `scipy.optimize.differential_evolution`, which was both
        slower and less exact: it left a median relative residual of 0.0009
        between its answer and the corner it was converging to.

        The problem is:
            maximize: payoff(crop_yield, q_surface, q_ground, ...)
            subject to: q_surface + q_ground = total_irrigation
                        q_surface in [surface_lb, surface_ub]

        Args:
            ufunc: Custom utility/payoff function. If None, uses
                `economic_payoff` which maximizes net economic benefit.
                The function must accept:
                - crop_yield: Dict[str, float] of crop yields (t/ha)
                - q_surface: float, surface water use
                - q_ground: float, groundwater use
                - Additional kwargs (water_prices, crop_prices, area, unit)
            total_irrigation: Total irrigation requirement in mm. If None,
                uses `self.seasonal_irrigation`.
            surface_boundaries: Tuple of (lower_bound, upper_bound) for
                surface water use in mm. If None, uses (0.0, total_irrigation).
            crop_yield: Attribute name or dict containing crop yields.
                Default "dry_yield" accesses `self.dry_yield`.
            kink: Where the objective breaks — the quota. Passing it adds the
                compliant corner `q_surface = quota` to the candidate set, and
                makes the affine check run per piece instead of over the whole
                interval. None means the objective is affine throughout.
            **kwargs: Additional arguments passed to the payoff function.
                Required if ufunc is None:
                - water_prices: Dict with "surface" and "ground" keys (RMB/m³)
                - crop_prices: Dict with crop names as keys (RMB/t)
                - area: Optional, irrigation area (ha)

        Returns:
            Tuple of (q_surface_opt, q_ground_opt) in mm, representing the
            optimal allocation of surface water and groundwater.

        Raises:
            ValueError: If surface_boundaries are invalid (negative, exceed
                total_irrigation, or lower > upper), or if ufunc is None but
                water_prices is missing from kwargs.
            Warning: If total_irrigation is zero, returns (0.0, 0.0) and logs
                a warning.

        Example:
            Optimize water allocation with default economic payoff:

            ```python
            water_prices = {"surface": 0.5, "ground": 0.8}  # RMB/m³
            crop_prices = {"Maize": 2000, "Wheat": 2500}  # RMB/t

            sw, gw = city.water_withdraw(
                total_irrigation=500.0,  # mm
                water_prices=water_prices,
                crop_prices=crop_prices,
                area=city.total_area
            )
            ```

        Note:
            The solve is exact and deterministic: it draws no random numbers,
            so the same inputs always give the same split. `water_withdraw`
            therefore no longer touches the model RNG at all (it used to, see
            issue #18) — which also means removing it shifts every downstream
            random draw.
        """
        if total_irrigation is None:
            total_irrigation = self.seasonal_irrigation
        if total_irrigation == 0.0:
            warnings.warn(f"Zero irr volume for {self.unique_id}.")
            return 0.0, 0.0
        if ufunc is None:
            if "water_prices" not in kwargs:
                raise ValueError(
                    "No custom function provided, calculating water costs."
                    "However, Missing arg `water_prices` in kwargs."
                )
            ufunc = economic_payoff
        if isinstance(crop_yield, str):
            crop_yield = getattr(self, crop_yield)

        # 求解本身不依赖智能体，放在 `core.allocation` 里（见 issue #27）。
        # 这里只负责把 City 的状态翻成它要的参数：默认灌溉量、默认收益函数、
        # 作物单产，以及目标函数的拐点（配额）。
        return solve_surface_share(
            ufunc,
            total_irrigation,
            surface_boundaries,
            kink=kink,
            crop_yield=crop_yield,
            **kwargs,
        )

    def link_friends(self, neighbours: Iterable[Self | None]) -> int:
        """Link this agent to each given neighbour with a mutual "friend" edge.

        The agent-side half of `MainModel.update_network`'s `observed` branch.
        It lives here rather than on the model for the same reason
        `Province.update_graph` does: creating a city's social ties is the
        city's business, and the model should hand over a neighbour list rather
        than reach in and wire links itself.

        Args:
            neighbours: The agents to befriend. `None` entries are skipped, so
                the caller can pass a lookup's misses straight through — a
                network node outside this run's city set is a routine case, not
                an error (12 of the 69 network nodes are outside the basin).

        Returns:
            How many links were created.

        Note:
            Re-linking an existing friend is a no-op that preserves ordering
            (`abses.human.links.add_a_link`), which is what lets the network be
            rebuilt every year without `City.friends` reordering — and that
            order is load-bearing for reproducibility (see `friends`).
        """
        linked = 0
        for neighbour in neighbours:
            if neighbour is not None:
                self.link.to(neighbour, "friend", mutual=True)
                linked += 1
        return linked

    @property
    def friends(self) -> ActorsList[Self]:
        """Get neighboring agents in the social network ("friends").

        The social network represents information sharing and peer influence
        between cities. Agents observe their friends' decisions and
        performance, which influences their own behavioral preferences and
        perceived payoffs. This mechanism is based on multi-cultural theory
        for modeling social learning in multi-agent systems.

        Returns:
            ActorsList of City agents that are linked to this agent through
            the "friend" relationship, in link creation order. These are the
            agents whose behavior and performance this agent can observe and
            learn from.

        Note:
            The order matters for reproducibility, because `judge_friends`
            draws from the shared model RNG once per friend. ABSESpy guarantees
            a stable order only since v0.11.7 — before that the link store
            returned an unordered `set` — hence the version floor in
            `pyproject.toml`.
        """
        return self.link.get("friend", default=True)

    @property
    def willing(self) -> DecisionType:
        """Get the agent's willingness to exceed water quota (decision tendency).

        This property represents the agent's behavioral tendency, which may
        differ from the actual decision based on policy enforcement. The
        willingness can take two values:
            - "D" (Defect): Willing to violate quota if crops need water and
              it is economically beneficial. Additional water needs will be
              met using surface water beyond quota.
            - "C" (Cooperate): Willing to comply with quota. Additional water
              needs will be met using groundwater instead of exceeding quota.

        Policy enforcement:
            - Before `include_s_since` year: Always returns "D" (no social
              factors considered)
            - After `forced_since` year: Always returns "C" (mandatory
              compliance enforced by policy)
            - Between these years: Returns the agent's internal `_willing`
              value (behavioral tendency)

        The first branch shadows the second whenever
        `forced_since < include_s_since`, which would silently turn a
        "strict enforcement" scenario into a "never" one. Rather than reorder
        the guards, `setup` rejects that combination outright — the study
        does not use it, so the honest answer is to refuse it rather than to
        quietly simulate something else (see issue #59).

        Note:
            This property models the historical policy change in the Yellow
            River Basin, where mandatory water allocation policies were
            officially implemented starting in 1999.

        Returns:
            Decision tendency: "C" for compliance or "D" for defect.

        Raises:
            KeyError: If `include_s_since` or `forced_since` is missing from
                the `City` parameters. Both gate the policy timeline, so a
                missing key is a configuration error rather than something to
                paper over with a default.
        """
        if self.time.year < self.p["include_s_since"]:
            return "D"
        if self.time.year >= self.p["forced_since"]:
            return "C"
        return self._willing

    @willing.setter
    def willing(self, value: DecisionType) -> None:
        if value not in self.valid_decisions:
            raise ValueError(f"Invalid decision: {self._willing}.")
        self._willing = value

    def compare(self, attr: str, my: Optional[float] = None) -> float:
        """Compare own attribute value with friends' values and return normalized rank.

        This method calculates a normalized ranking of the agent's attribute
        value relative to friends in the social network. The ranking is used
        for social learning and payoff calculation.

        Args:
            attr: Attribute name to compare. The method uses `ActorsList.array`
                to get an array of friends' attribute values.
            my: Own attribute value. If None, uses `self.get(attr)` to get
                the value from the agent's attributes.

        Returns:
            Normalized rank in range [0, 1], where:
                - 1.0: Best rank (own value is maximum among all)
                - 1.0: Also returned if all values are equal (egalitarian)
                - 0.0: Worst rank (own value is minimum among all)
            The value represents the agent's relative position in the social
            network for this attribute.

        Note:
            If the agent has no friends, returns 1.0 (best rank by default).
        """
        if not self.friends:
            return 1.0
        arr = self.friends.array(attr=attr)
        my = self.get(attr) if my is None else my
        min_val, max_val = min([my, arr.min()]), max([my, arr.max()])
        if min_val == max_val:
            return 1.0
        return (my - min_val) / (max_val - min_val)

    def calc_social_standing(
        self,
        q_surface: float,
        standing: Optional[Dict[DecisionType, float]] = None,
    ) -> float:
        """Social standing this agent would retain under a given withdrawal.

        Turns a candidate `q_surface` into a compliance decision and looks the
        answer up in the two-entry table `standing_by_decision` builds. The
        peer counting and the call into `social_standing` happen **there**, not
        here — see that function for the returned multiplier and, importantly,
        its direction (issue #60).

        The social term depends on the candidate allocation only through
        whether it breaches the quota, so the table has exactly two entries and
        this method is a lookup. Peer evaluation follows the social sub-model of
        [@castillarho2017a]; the functional form is its Supplementary equation (3).

        Args:
            q_surface: Surface water use in units of 1e8 m³ (100 million m³).
                Compared with `self.quota` to decide whether this candidate
                withdrawal counts as a violation.
            standing: Precomputed `{"C": ..., "D": ...}` from
                `standing_by_decision`. The optimizer passes it so the two
                values are computed once per year instead of once per
                candidate evaluation (see issue #72).

        Returns:
            Retained social standing in range [0, 1], passed straight to
            `agg_payoff` as the `s` factor.

        Note:
            This method was called `calc_social_costs`, which named the
            complement of what it returns (see issue #60).

        See Also:
            - `cwatqim.core.payoff.social_standing`: The underlying function
            - `cwatqim.agents.city.City.judge_friends`: Method for evaluating
                neighbor behavior
        """
        if standing is None:
            standing = self.standing_by_decision()
        return standing[self.decide(q_surface)]

    def standing_by_decision(self) -> Dict[DecisionType, float]:
        """This year's social standing under each of the two decisions.

        The social term depends on the candidate allocation only through
        whether it breaches the quota, so over the whole feasible domain it
        takes **two** values — it is a step function at the quota, not a
        continuous one. Computing it here, once, instead of inside every
        objective evaluation is therefore exact rather than approximate
        (see issue #72).

        Measured: 7 cities over 20 years called the social term 18,962 times;
        this reduces that to 2 per agent-year.

        Returns:
            Mapping from decision ("C" / "D") to the standing retained.

        Note:
            One of the two is often irrelevant: when `willing` is "C",
            `decide_boundaries` caps the upper bound at the quota, so every
            candidate is compliant and the social term is a positive constant
            — which cannot move the argmax at all. It is still computed here
            because `irrigating` records the realised payoff afterwards.
        """
        grid = self.s_grid
        # 逐城取值：`City.s_group_kappa` 为 0 时等于配置里的全国标量。
        group = self.s_group
        # `judge_friends` 对两个候选各跑一遍邻居循环，其中 `dislikes` 两支必然
        # 相同、而 "C" 支的 `criticized` 必然是 0——所以**看起来**有一半是白算的。
        # 不合并是有意的：合并要把这两条性质在这里再写一遍，等于给计数规则开第二
        # 处定义（#129），也破坏 #72 要的「判断是边的属性」。省下的量级也不值：
        # 整个目标函数只占约 3% 墙钟，AquaCrop 才是大头（见 `core.allocation`）。
        standing: Dict[DecisionType, float] = {}
        for candidate in self.valid_decisions:
            dislikes, criticized = self.judge_friends(decision=candidate)
            standing[candidate] = social_standing(
                grid,
                group,
                criticized,
                dislikes,
            )
        return standing

    def calc_payoff(
        self,
        crop_yield: Dict[str, float],
        q_surface: float,
        q_ground: float,
        water_prices: Optional[dict] = None,
        crop_prices: Optional[dict] = None,
        standing: Optional[Dict[DecisionType, float]] = None,
        record: bool = False,  # 最后优化完了，算一次并记录分数
    ) -> float:
        """Calculate combined economic and social payoff.

        This method aggregates the agent's economic and social performance
        into a single payoff value. The payoff combines:
            - Economic score (e): Net economic benefit from crop production
              minus water costs
            - Social standing retained (s): what survives peer criticism, in
              [0, 1] where 1.0 means nobody criticised (issue #60)
            - Gross revenue (R): the yardstick the social term is priced
              against

        The utility is multiplicative:
            - With the social term: `U = e · s`
            - Before `include_s_since`: `U = e` exactly (a branch, not a weight)

        The economic score is calculated using `economic_payoff` and the gross
        revenue by `gross_revenue` — the same function supplies the minuend of
        the former, so the two cannot drift. The social standing comes from
        `calc_social_standing`, which considers rule compliance and peer
        behavior.

        Args:
            crop_yield: Dictionary mapping crop names to yields in tonnes/ha.
                Keys should be "Maize", "Wheat", "Rice".
            q_surface: Surface water use in 1e8 m³.
            q_ground: Groundwater use in 1e8 m³.
            water_prices: Dictionary with water prices in RMB/m³. Should
                contain keys "surface" and "ground". If None, uses
                `self.water_prices`.
            crop_prices: Dictionary with crop prices in RMB/t. Keys should
                match crop names. If None, uses `self.crop_prices`.
            standing: Precomputed `{"C": ..., "D": ...}` from
                `standing_by_decision`, so the two values are computed once per
                agent-year instead of once per candidate (issue #72).
            record: Whether to store `e` / `s` / `revenue` / `payoff` on the
                agent. The optimiser leaves it False; `irrigating` sets it once
                after the solve.

        Returns:
            The utility `U`, in the units of `e` (RMB). `include_s` decides
            whether it is `e * s` (floored) or plain `e` — see `agg_payoff`.

        Note:
            This method is typically called during water source optimization
            to evaluate different allocation strategies. The final payoff
            after optimization is recorded using `record=True`.

        See Also:
            - `cwatqim.core.payoff.economic_payoff`: Economic benefit calculation
            - `cwatqim.agents.city.City.calc_social_standing`: Retained
                social standing
            - `cwatqim.agents.city.agg_payoff`: Payoff aggregation method
        """
        if water_prices is None:
            water_prices = self.water_prices
        if crop_prices is None:
            crop_prices = self.crop_prices
        e = economic_payoff(
            q_surface=q_surface,
            q_ground=q_ground,
            crop_yield=crop_yield,
            water_prices=water_prices,
            crop_prices=crop_prices,
            area=self.irr_area,
            unit="1e8m3",
        )
        # 社会项的标尺。与 `economic_payoff` 的被减数是同一个函数，不会漂移。
        revenue = gross_revenue(crop_yield, crop_prices, self.irr_area)
        # 政策年之前 `agg_payoff` 会丢掉 s，没必要先算出来再扔——除非这一次
        # 调用要记录它（`record=True` 会把 s 写进主体、进而被采集）。短路只
        # 发生在优化循环里，落盘的值一个不差（见 issue #72）。
        if self.include_s or record:
            s = self.calc_social_standing(q_surface=q_surface, standing=standing)
        else:
            s = 1.0
        return self.agg_payoff(
            e=e,
            s=s,
            revenue=revenue,
            include_s=self.include_s,
            record=record,
        )

    def agg_payoff(
        self,
        e: float,
        s: float,
        revenue: float,
        record: bool = False,
        include_s: bool = True,
    ) -> float:
        """Aggregate economic and social scores into final payoff.

        The aggregation is **multiplicative** — criticism discounts the whole
        payoff rather than subtracting a priced share of it:

            U = e * s

        This restores the form used by the reference literature, which is what
        makes the deterrent ratio `s(C)/s(D)` the right reading again
        (`water_quota_analysis.analysis.social_cost`). It reverses PR #132.

        The sign defect that comes with it (issue #121: `e < 0` inverts the
        penalty, ~12.5% of city-years) is **deliberately kept** as of
        2026-09-01: where the economics is already a negative incentive, the
        social term turning into a reward is an acceptable reading. The
        `City.payoff_floor` that used to correct the sign was removed instead,
        because it flattened the objective wherever it bound and left 12.5% of
        agent-years with no allocation decision at all (issue #213); the full
        argument is in `aggregate_utility`. What remains is the saturation of
        the ratio channel (issue #110).

        There is no social weight: `lambda` has nowhere to sit in a product, so
        `City.s_weight` was retired with this change.

        Args:
            e: Economic score, representing net economic benefit. Range
                typically [0, inf), but can be negative if costs exceed
                revenue.
            s: Social standing **retained**, in [0, 1], where 1.0 means nobody
                criticised (issue #60 — read it as what survives, never as a
                cost to subtract).
            revenue: Gross crop revenue. It no longer enters the utility —
                the multiplicative form has no yardstick to price against — but
                it is still recorded (`self.revenue`) because the analysis side
                reads it, notably to express the economic temptation and the
                deterrent on the same scale. Deliberately has **no default**:
                the caller always knows it, and a 0.0 fallback would put a
                wrong number in the collected column.
            record: If True, stores the scores as agent attributes (self.e,
                self.s, self.revenue, self.payoff). Set to False during the
                solve to avoid side effects.
            include_s: If True, `U = e * s`. If False, U is **exactly** `e` —
                a branch, since a product has no weight to zero out.

        Returns:
            The utility U, in the units of `e` (RMB). It carries the sign of
            `e`; on loss-making years `U` therefore *rises* as `s` falls (see
            `aggregate_utility`).

        Raises:
            ValueError: Propagated from `aggregate_utility` when `e` or `s` is
                non-finite, or `s` falls outside [0, 1].

        Example:
            Calculate and record the final utility:

            ```python
            payoff = city.agg_payoff(
                e=economic_score,
                s=social_score,
                revenue=gross_revenue_score,
                record=True,  # Store for analysis
            )
            # Now city.e, city.s, city.revenue, city.payoff are set
            ```

        Note:
            返回的始终是**绝对**效用。这里曾有一个 `rank=` 开关，按朋友集把效用
            min-max 归一；它随加性效用一起退役（`relative_utility` 已删），退役后
            零调用者，于是连同它那段文档一并删掉。要比位次请直接用
            `City.compare`——`economic_position` / `social_position` 走的就是它。
        """
        # 乘性形式没有权重旋钮，所以 `include_s` 是一个**分支**而不是 λ=0：
        # 窗口外直接返回 e（**不夹下限**——没有乘法就不该改动经济收益，否则
        # `never` 情景会跟着动，而它逐位不变是一条有用的一致性检验），
        # 窗口内才乘上 s。
        payoff = aggregate_utility(economic=e, standing=s) if include_s else e
        if record:
            self.e = e
            self.s = s
            self.revenue = revenue
            self.payoff = payoff
        return payoff

    def make_decision(self) -> DecisionType:
        """Make a random decision based on the agent's boldness parameter.

        The decision is probabilistically determined by the agent's boldness
        value, which represents the probability of choosing "D" (defect)
        over "C" (cooperate).

        Returns:
            Decision type: "D" with probability equal to boldness, "C" otherwise.
        """
        return "D" if self.random.random() < self.boldness else "C"

    def mutate_strategy(self, probability: float) -> None:
        """Randomly mutate behavioral strategy with given probability.

        This method implements strategy mutation to avoid getting trapped in
        local optima. With a small probability, it randomly resets one of the
        key behavioral parameters (boldness or vengefulness) to a new random
        value.

        The mutation process:
            1. Generate random number; if > probability, no mutation occurs
            2. If mutation occurs, randomly select boldness or vengefulness
               (50% chance each)
            3. Reset selected parameter to random value in [0, 1]

        Args:
            probability: Probability of mutation occurring, should be in range
                [0, 1]. Typical values are small (e.g., 0.01-0.1) to allow
                occasional exploration without disrupting convergence.
        """
        if self.random.random() > probability:
            return
        if self.random.random() < 0.5:
            self.boldness = self.random.random()
        else:
            self.vengefulness = self.random.random()

    def will_report(self, behave: DecisionType, my_decision: DecisionType) -> bool:
        """Whether this agent criticises a neighbour showing `behave`.

        Reporting a peer is a **decision**, not a coin flip. Filing a report
        costs goodwill — `s_grid` is the share that **survives** one report, so
        the cost is its complement `1 - grid` — and an agent files only when
        the norm matters to it more than the report costs. Folding the marginal
        cost into the same utility as everything else, the economic payoff `e`
        cancels and leaves a dimensionless rule:

            report the (m+1)-th defector  iff  v > (1 - grid) * grid^m

        Derivation: under `U = e * s`, one more report multiplies the surviving
        goodwill by `grid`, so it costs `e * grid^m * (1 - grid) / 2` of
        utility; write the normative satisfaction as `e * v / 2`, i.e. `v`
        times the most goodwill anyone can hold, and divide both sides by `e`.

        **That division assumes `e > 0`.** Where the economic payoff is negative
        — about 11% of city-years, and 18.6% of the decisions where the quota
        actually binds — dividing flips the inequality, so the rule as
        implemented is not the one the derivation gives. This is the same
        multiplicative sign defect as issue #121, reaching enforcement rather
        than compliance; the implementation deliberately keeps the flat rule
        `v > 1 - grid` for every agent.

        The threshold **falls** with `m`, because the Cobb-Douglas form makes
        each further report cheaper in absolute terms. Enforcement is therefore
        all-or-nothing: an agent that files the first report files every one,
        and the rule collapses to `v > 1 - grid`. That is what this method
        implements, and it is why the fraction of compliant agents who enforce
        is `grid` for `v ~ U(0, 1)`.

        Two honest caveats:

        1. At `grid = 1` the threshold is 0, so the rule becomes `v > 0` —
           every eligible agent always reports (reporting has become free). It
           does **not** revert to "v is the reporting probability"; that was
           the old Bernoulli rule (`draw <= vengefulness`), and nothing here
           restores it.
        2. The `v / 2` on the benefit side is a **normalisation choice**, not a
           derived quantity: it reads the satisfaction of enforcing a norm as
           `v` times half of the most goodwill anyone can hold. Under the
           product form `s = grid^m * (1 - group)^n` that ceiling is 1, so the
           halving is now a bare convention rather than something the average
           form handed us. Writing it as `v * 1` instead would give
           `v > (1 - grid) / 2`, moving the enforcement share from `grid` to
           `(1 + grid) / 2` — 39% to 70% at the calibrated 0.39. The *shape*
           is robust (monotone in `grid`, all-or-nothing per agent); the level
           rides on this choice and belongs in the sensitivity analysis.

        Why it changed (issues #110, #72): the old rule was
        `draw <= self.vengefulness`, an unconditional Bernoulli. `grid` had no
        way in, so it could only ever act through the *ratio* `s(C)/s(D)` — and
        that channel is saturated, because a single critic already produces a
        deterrent of `1 / (1 - group) = 1.89` while the economic temptation is
        1.114 at the median. Enforcement is the **extensive** margin.

        Under t-1 eligibility this rule is `grid`'s **only** channel into
        compliance, and its sign is therefore unambiguous: a larger `grid`
        means cheaper reporting, more enforcers, a larger `n`, and a stronger
        deterrent. The
        `grid^m` factor is identical on both branches (see `judge_friends`),
        so it cancels out of `s(C)/s(D)` entirely and cannot pull the other
        way. That second channel exists only under candidate-based
        eligibility, which was tried and rejected on 2026-08-31 (issue #209).

        Args:
            behave: The neighbour's decision being judged, "C" or "D".
            my_decision: This agent's **own** compliance state, which decides
                whether it is entitled to criticise at all. Deliberately has no
                default: both callers pass `last_decision`, but they read it
                off *different agents*, and a default would silently pick one.
                See `judge_friends`.

        Returns:
            True if this agent files a report.

        Note:
            Deterministic — it draws nothing. Both sides of a pair therefore
            agree by construction, which is what #72 asked for: a judgement is
            a property of the **edge**, and the old per-agent `_judgements`
            table kept two independent random ledgers for the same event.

            Eligibility is always read off `last_decision`, which the model
            snapshots before any city steps (`snapshot_decision`), so the
            observation set is the realised state of year t-1 for everyone —
            no longer a mix of three time slices, and not a function of the
            candidate branch being priced.
        """
        # 自己违规就没有资格批评别人（按去年已实现的状态判定）。资格由调用方
        # **显式**给出，因为两个方向要读同一个断面（见 issue #72、#209）。
        if my_decision == "D":
            return False
        # 对方守约，没什么可批评的
        if behave == "C":
            return False
        # 规范重视程度要压过举报的代价。规则只有一份定义，分析层画执法率的
        # 那张图走的是同一份（`core.payoff.enforcement_share`，见 #129）。
        return reports_defector(self.vengefulness, self.s_grid)

    def judge_friends(self, decision: DecisionType) -> Tuple[int, int]:
        """Count who criticises whom — the two exponents of the social term.

        The social term is `s = grid^m * (1 - group)^n` (Castilla-Rho et al.
        2017, SI eq. 3; the config keys `s_grid` / `s_group` hold the source's
        symbols as they are — see `core.payoff.social_standing` and issue
        #210). This method produces the pair `(m, n)`:

            m = `dislikes`, neighbours **this** agent criticises
            n = `criticized`, neighbours who criticise **this** agent

        **Eligibility on both directions reads year t-1** (`last_decision`),
        never the candidate `decision` being priced. Two consequences worth
        keeping straight, because they decide what the whole social channel
        can and cannot do:

        1. `m` is **the same on both branches** — whether I intend to comply
           this year does not change whether I complied last year. So
           `grid^m` cancels out of the deterrent
           `s(C)/s(D) = (1 - group)^(-n)`, and `grid` reaches compliance only
           through the enforcement share `grid` in `will_report`, which sets
           the distribution of `n`.
        2. The deterrent is therefore always `>= 1`: complying is never
           socially worse than defecting.

        Candidate-based eligibility ("defecting this year strips my standing
        to criticise") was implemented and measured on 2026-08-31 to put
        `grid` back into the ratio. It does, but it also makes ~19% of the
        parameter plane invert (1. above fails), raises the breach rate by
        2.9 pp, and breaks #72's edge consistency on the outgoing direction.
        Rejected — see issue #209 and `multirun/floor_elig` for the run.

        Args:
            decision: The candidate branch being priced, "C" or "D". It moves
                `criticized` only; `dislikes` is invariant to it by design.

        Returns:
            `(dislikes, criticized)` = `(m, n)`, to be passed to
            `core.payoff.social_standing` in that order.

        See Also:
            - `cwatqim.agents.city.City.will_report`: the per-edge rule.
            - `water_quota_analysis.analysis.social_cost.standing_by_branch`:
              the closed form of the two branches, kept in parity with this
              method by `tests/analysis/test_social_cost.py::TestModelParity`.
        """
        dislikes, criticized = 0, 0
        for friend in self.friends:
            # 我的资格看自己的 t−1：候选决策改不了我去年守没守约。
            dislikes += self.will_report(
                friend.last_decision, my_decision=self.last_decision
            )
            # 朋友的资格同样看他的 t−1：他今年的行为在我求解时还不存在。
            criticized += friend.will_report(decision, my_decision=friend.last_decision)
        return dislikes, criticized

    def change_mind(self, metric: str, how: str) -> bool:
        """Learn behavioral strategies from better-performing neighbors.

        This method implements social learning where agents observe and adopt
        the behavioral parameters (boldness, vengefulness) of neighbors who
        perform better according to a specified metric. This creates an
        evolutionary dynamic where successful strategies spread through the
        social network.

        The learning process:
            1. Identify friends who perform better on the specified metric
            2. Select one friend based on the `how` parameter
            3. Copy that friend's boldness and vengefulness values
            4. Return True if learning occurred, False otherwise

        This mechanism allows the model to explore the strategy space while
        also exploiting successful strategies found by peers.

        Args:
            metric: Performance metric for comparison. Options:
                - "unit_payoff": utility per unit of gross revenue. This is
                  what `step` uses, and the only one that is scale-free —
                  see the property's docstring for why comparing absolute
                  `payoff` degenerates into "imitate the biggest city".
                - "e": Economic score (net economic benefit)
                - "s": Social score (social standing retained)
                - "payoff": Absolute utility, in RMB
            how: Learning strategy when multiple better neighbors exist:
                - "best": Learn from the neighbor with the highest metric value
                - "random": Learn from a randomly selected better neighbor

        Returns:
            True if the agent learned from a neighbor (attributes were
            updated), False if no better neighbors were found or learning
            did not occur.

        Example:
            Learn from best-performing friend:

            ```python
            learned = city.change_mind(metric="unit_payoff", how="best")
            if learned:
                print(f"Updated boldness: {city.boldness}")
                print(f"Updated vengefulness: {city.vengefulness}")
            ```

        Note:
            This method only updates behavioral parameters, not decision
            outcomes. The new parameters will influence future decisions but
            don't retroactively change past behavior.

        See Also:
            - `cwatqim.agents.city.friends`: Social network connections
            - `cwatqim.agents.city.mutate_strategy`: Random strategy mutation
        """
        better_friends = self.friends.better(metric=metric, than=self)
        # If no better-performing friends, return False
        if not better_friends:
            return False
        elif how == "best":
            friend = better_friends.better(metric=metric).random.choice()
        elif how == "random":
            friend = better_friends.random.choice()
        else:
            raise ValueError(f"Invalid how parameter: {how}")
        # Learn from the better-performing friend
        self.boldness = friend.boldness
        self.vengefulness = friend.vengefulness
        return True

    def decide_boundaries(
        self,
        seasonal_irr: float,
    ) -> Tuple[float, float]:
        """Determine lower and upper bounds for water withdrawal.

        This method sets the constraints for water source optimization based
        on the agent's decision tendency:
            - If willing to comply ("C"): Upper bound is the minimum of quota
              and seasonal irrigation (cannot exceed quota)
            - If willing to defect ("D"): Upper bound is seasonal irrigation
              (can use all needed water)

        Args:
            seasonal_irr: Seasonal irrigation requirement in 1e8 m³.

        Returns:
            Tuple of (lower_bound, upper_bound) in 1e8 m³. Lower bound is
            always 0.0. Upper bound depends on decision tendency.
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
        """Execute irrigation decision-making process.

        This method orchestrates the annual irrigation decision, which involves:
            1. Determining irrigation boundaries based on decision tendency
            2. Optimizing water source allocation (surface vs. groundwater)
               using genetic algorithm based on payoff differences
            3. Calculating and recording final scores for the optimal allocation

        Args:
            seasonal_irr: Seasonal irrigation requirement in 1e8 m³. If None,
                uses `self.total_wu`.
            water_prices: Dictionary with water prices in RMB/m³, containing
                keys "surface" and "ground". If None, uses `self.water_prices`.
            crop_prices: Dictionary with crop prices in RMB/t, with crop names
                as keys. If None, uses `self.crop_prices`.
            **kwargs: Additional arguments passed to optimization and payoff
                calculation methods.

        Returns:
            Tuple of (surface_water, ground_water) in 1e8 m³, representing
            the optimal allocation for this irrigation season.
        """
        if seasonal_irr is None:
            seasonal_irr = self.total_wu
        # 显式解析价格：优化目标必须和事后记录的目标是同一个函数，
        # 漏传 `crop_prices` 会让差分进化只最小化水费（见 issue #15）。
        if water_prices is None:
            water_prices = self.water_prices
        if crop_prices is None:
            crop_prices = self.crop_prices
        boundaries = self.decide_boundaries(seasonal_irr)
        if boundaries[1] <= 0.0:
            self.surface_water = 0.0
            self.ground_water = 0.0
            return 0.0, 0.0
        # 社会项对候选解只有二值依赖（在配额处跳变），所以这两个值一年算一次
        # 就够，不必每评估一个候选解重算一遍（见 issue #72）。优化和事后记录
        # 共用同一份，两者的目标函数也就必然是同一个。
        standing = self.standing_by_decision()
        opt_surface, opt_ground = self.water_withdraw(
            ufunc=self.calc_payoff,
            surface_boundaries=boundaries,
            total_irrigation=seasonal_irr,
            water_prices=water_prices,
            crop_prices=crop_prices,
            crop_yield=self.dry_yield,
            standing=standing,
            # 配额就是目标函数的拐点，也是"守约角点"。`willing == "D"` 时它落在
            # 可行域内部，于是社会项能把违规意图改判回守约（见 #94、#110）。
            kink=self.quota,
            **kwargs,
        )
        # 用最优配水再算一次并落盘：优化期间 `record=False`，这里才写进主体
        self.calc_payoff(
            crop_yield=self.dry_yield,
            q_surface=opt_surface,
            q_ground=opt_ground,
            water_prices=water_prices,
            crop_prices=crop_prices,
            standing=standing,
            record=True,
        )
        return opt_surface, opt_ground

    def step(self) -> None:
        """Execute one annual time step for the city agent.

        This method orchestrates the city's annual decision-making cycle,
        which includes:
            1. **Water Allocation Optimization**: Determines optimal allocation
               of surface water and groundwater based on economic and social
               payoffs
            2. **Crop Simulation**: Simulates crop growth and yield using
               AquaCrop based on climate and irrigation
            3. **Performance Evaluation**: Calculates and records economic
               and social scores
            4. **Social Learning**: Updates behavioral parameters by learning
               from better-performing neighbors
            5. **Strategy Mutation**: Randomly mutates strategies with small
               probability to avoid local optima
            6. **Decision Update**: Updates the agent's decision tendency for
               the next year

        The execution order ensures that:
            - Water allocation is optimized before crop simulation
            - Crop yields are available for payoff calculation
            - Learning occurs after performance evaluation
            - Next year's strategy is set before the time step ends

        Note:
            This method is called automatically by the model's step() method
            for all city agents. The order of execution is randomized
            (shuffle_do) to avoid systematic biases.

        See Also:
            - `cwatqim.agents.city.irrigating`: Water allocation optimization
            - `cwatqim.agents.city.simulate`: Crop yield simulation
            - `cwatqim.agents.city.change_mind`: Social learning mechanism
            - `cwatqim.agents.city.mutate_strategy`: Strategy mutation
        """
        water_prices = self.water_prices
        crop_prices = self.crop_prices
        # Optimize water source allocation
        # total_wu is in 1e8 m^3; irrigating() returns (surface, ground) in 1e8 m^3
        sw, gw = self.irrigating(self.total_wu, water_prices, crop_prices)
        # Record volumes in 1e8 m^3 for consistency with quota
        self.surface_water = sw  # 1e8 m^3
        self.ground_water = gw  # 1e8 m^3
        self.simulate(
            repeats=self.p.get("repeats", 1)
        )  # Crops require this amount of water
        # Learn from better performers and potentially mutate strategy for next year.
        # 比的是**单位毛收入**的效用，不是绝对值：后者等于"模仿最大的城市"，
        # 性状会按灌溉面积而不是行为被选择（见 `unit_payoff` 与 issue #110）。
        self.change_mind(metric="unit_payoff", how="random")
        self.mutate_strategy(probability=self.p["mutation_rate"])
        # Assign, don't just call: `make_decision` leaves `self` untouched (it
        # only draws from the model RNG), so dropping its return value froze
        # `_willing` at its `setup` value forever (#16).
        self.willing = self.make_decision()
