#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import warnings
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
        irr_method (int): Irrigation method code (1-4), where higher numbers
            typically indicate more efficient methods.
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

    def __getattr__(self, name: str):
        """Dynamic attribute access for crop yield properties.

        This method enables convenient access to crop yields using attribute
        notation. For example, `city.dry_yield_maize` will return the dry
        yield for maize.

        Supported patterns:
            - `dry_yield_{crop}`: Returns dry yield for the specified crop
            - `yield_potential_{crop}`: Returns potential yield for the crop

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
        return super().__getattr__(name)

    @property
    def climate_datapath(self) -> Path:
        """Get the file path to this city's climate data.

        The climate data file is expected to be named `climate_C{city_id}.csv`
        and located in the directory specified by `self.ds.city_climate_dir`.

        Returns:
            Path object pointing to the climate data CSV file.

        Raises:
            AssertionError: If the climate data file does not exist. This
                typically indicates a configuration error or missing data file.

        Note:
            The file should contain daily climate data with columns:
            - Date: Date in datetime format
            - MinTemp: Minimum temperature (°C)
            - MaxTemp: Maximum temperature (°C)
            - Precipitation: Daily precipitation (mm)
            - ReferenceET: Reference evapotranspiration (mm)
        """
        dp = Path(self.ds.city_climate_dir) / f"climate_C{self.city_id}.csv"
        assert dp.exists(), f"Climate data file not found: {dp}"
        return dp

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
        return self.dynamic_var("irr_area")

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
        return self.dynamic_var("wui")

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
        if self.surface_water > self.quota:
            return "D"
        return "C"

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

        Returns:
            True if social factors should be included, False otherwise.
            Currently always returns True, but can be configured based on
            simulation year if needed.
        """
        # return self.time.year >= self.p.include_s_since
        return True

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
        # ===== Farm-related attributes =====
        self.irr_method = 4
        # ===== Water-related attributes =====
        self._quota = 0.0
        self.surface_water = 0.0
        self.ground_water = 0.0
        # ===== Social-related attributes =====
        self.boldness = self.random.random()
        self.vengefulness = self.random.random()
        self.willing = self.make_decision()
        # ===== Score-related attributes =====
        # income: -inf~inf
        # social benefits: 0~1
        self.agg_payoff(e=0.0, s=1.0, record=True, rank=False, include_s=self.include_s)

    def calc_max_irr_seasonal(self, crop: str) -> float:
        """Calculate maximum seasonal irrigation depth for a crop.

        This method calculates the maximum irrigation depth (in mm) that can
        be applied to a crop, accounting for:
            - The crop's water use intensity (WUI)
            - The proportion of surface water vs. groundwater
            - Irrigation efficiency for each water source

        The calculation weights the WUI by the surface/groundwater ratio
        and their respective irrigation efficiencies. This accounts for the
        fact that different water sources have different application
        efficiencies.

        Args:
            crop: Crop name ("Maize", "Wheat", or "Rice") for which to
                calculate maximum irrigation.

        Returns:
            Maximum seasonal irrigation depth in millimeters (mm). This value
            is used by AquaCrop to constrain irrigation applications.

        Formula:
            max_irr = WUI * (sw_ratio * sw_eff + gw_ratio * gw_eff)

            Where:
            - WUI: Water use intensity for the crop (mm)
            - sw_ratio: Surface water proportion
            - gw_ratio: Groundwater proportion
            - sw_eff: Surface water irrigation efficiency
            - gw_eff: Groundwater irrigation efficiency

        Note:
            If both surface_water and ground_water are zero, the calculation
            will result in NaN. This should be handled by the calling code.
        """
        wui = self.wui[crop]
        sw = self.surface_water / (self.surface_water + self.ground_water)
        gw = self.ground_water / (self.surface_water + self.ground_water)
        return wui * sw * self.province.sw_irr_eff + wui * gw * self.province.gw_irr_eff

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

        Raises:
            FileNotFoundError: If climate data file is missing.
            ValueError: If crop name is invalid or crop has no irrigated area.

        See Also:
            - `aquacrop.core.AquaCropModel`: The underlying crop simulation model
            - `cwatqim.agents.city.to_regional_crop`: Function for regionalizing crops
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
        """Optimize water source allocation using genetic algorithm.

        This method uses differential evolution (a genetic algorithm) to find
        the optimal allocation of surface water and groundwater that maximizes
        the agent's payoff function. The optimization considers:
            - Crop yields (which depend on total irrigation)
            - Water prices (different for surface and groundwater)
            - Crop prices
            - Social costs (if included in the payoff function)

        The optimization problem is:
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
            ga_kwargs: Additional parameters for differential_evolution:
                - popsize: Population size multiplier (default: 15)
                - maxiter: Maximum iterations (default: 100)
                - polish: Use L-BFGS-B to polish solution (default: True)
                - seed: Random seed (default: None)
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
            The optimization uses scipy's differential_evolution, which is
            robust to local optima but may be slower than gradient-based
            methods. The default parameters are tuned for this application.
        """
        if total_irrigation is None:
            total_irrigation = self.seasonal_irrigation
        if total_irrigation == 0.0:
            warnings.warn(f"Zero irr volume for {self.unique_id}.")
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
        """Get all land cells (patches) within this city's geometry.

        Args:
            layer: The patch layer to select from. If None, uses the model's
                major layer (typically the main spatial layer).

        Returns:
            ActorsList of CropCell objects representing land patches within
            the city's boundary.
        """
        if layer is None:
            layer = self.model.nature.major_layer
        return layer.select(self.geometry)

    @property
    def friends(self) -> ActorsList[Farmer]:
        """Get neighboring agents in the social network ("friends").

        The social network represents information sharing and peer influence
        between cities. Agents observe their friends' decisions and
        performance, which influences their own behavioral preferences and
        perceived payoffs. This mechanism is based on multi-cultural theory
        for modeling social learning in multi-agent systems.

        Returns:
            ActorsList of City agents that are linked to this agent through
            the "friend" relationship. These are the agents whose behavior
            and performance this agent can observe and learn from.
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

        Note:
            This property models the historical policy change in the Yellow
            River Basin, where mandatory water allocation policies were
            officially implemented starting in 1999.

        Returns:
            Decision tendency: "C" for compliance or "D" for defect.
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

    def calc_social_costs(
        self,
        q_surface: float,
    ) -> float:
        """Calculate social costs based on rule compliance and peer behavior.

        This method implements a social cost function that captures two
        mechanisms:
            1. **Reputation Loss**: Cost from being criticized by neighbors
               when violating rules
            2. **Social Discontent**: Cost from observing neighbors violate
               rules while oneself complies

        The social cost is calculated using a Cobb-Douglas function, which
        creates a non-linear relationship between violations and costs. The
        cost depends not only on the agent's own behavior but also on the
        behavior of neighbors in the social network.

        Key mechanisms:
            - If an agent violates rules (q_surface > quota) when neighbors
              comply, they face high reputation loss
            - If an agent complies when neighbors violate, they experience
              social discontent (feeling of unfairness)
            - If both agent and neighbors violate, social costs are lower
              (violation becomes normalized)

        The calculation uses parameters:
            - `s_enforcement_cost`: Weight for social discontent (default: 0.5)
            - `s_reputation`: Weight for reputation loss (default: 0.5)

        Args:
            q_surface: Surface water use in units of 1e8 m³ (100 million m³).
                This value is compared with `self.quota` to determine if
                the agent is violating rules.

        Returns:
            Social cost value in the range [0, 1], where:
                - 0.0: No social cost (best case)
                - 1.0: Maximum social cost (worst case)
            The value is the equal-weighted average of enforcement cost and
            reputation loss.

        Note:
            The social cost calculation is based on multi-cultural theory
            and peer evaluation mechanisms. The Cobb-Douglas function ensures
            that costs increase non-linearly with the number of violations
            observed or committed.

        See Also:
            - `cwatqim.core.payoff.lost_reputation`: Function calculating
                reputation loss using Cobb-Douglas
            - `cwatqim.agents.city.judge_friends`: Method for evaluating
                neighbor behavior
        """
        # Compare potential surface water use with quota to determine violation
        # Both q_surface and self.quota are in 1e8 m³
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
        """Calculate combined economic and social payoff.

        This method aggregates the agent's economic and social performance
        into a single payoff value. The payoff combines:
            - Economic score (e): Net economic benefit from crop production
              minus water costs
            - Social score (s): Social satisfaction based on peer evaluations

        The final payoff is calculated as:
            - If social factors included: payoff = e * s
            - If only economic: payoff = e

        The economic score is calculated using `economic_payoff`, which
        considers crop revenue and water costs. The social score is calculated
        using `calc_social_costs`, which considers rule compliance and peer
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
            **kwargs: Additional arguments passed to `agg_payoff`, including:
                - record: Whether to store scores as attributes (default: False)
                - rank: Whether to convert scores to rankings (default: False)
                - include_s: Whether to include social factors (default: True)

        Returns:
            Combined payoff value. The range depends on whether ranking is
            used:
                - With ranking: [0, 1] (normalized relative to peers)
                - Without ranking: [0, inf) for economic, [0, 1] for social

        Note:
            This method is typically called during water source optimization
            to evaluate different allocation strategies. The final payoff
            after optimization is recorded using `record=True`.

        See Also:
            - `cwatqim.core.payoff.economic_payoff`: Economic benefit calculation
            - `cwatqim.agents.city.calc_social_costs`: Social cost calculation
            - `cwatqim.agents.city.agg_payoff`: Payoff aggregation method
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
        """Aggregate economic and social scores into final payoff.

        This method combines economic and social scores into a single payoff
        value that represents the agent's overall performance. The method
        supports two modes:
            1. **Ranking mode**: Converts absolute scores to relative rankings
               within the social network (used during optimization)
            2. **Absolute mode**: Uses raw scores (used for final evaluation)

        The payoff calculation:
            - If ranking: Converts both e and s to [0, 1] rankings, then
              multiplies: payoff = rank_e * rank_s
            - If not ranking: Uses raw scores: payoff = e * s (or just e)

        Args:
            e: Economic score, representing net economic benefit. Range
                typically [0, inf), but can be negative if costs exceed
                revenue.
            s: Social score, representing social satisfaction. Range [0, 1],
                where 1.0 is maximum satisfaction.
            record: If True, stores the scores as agent attributes (self.e,
                self.s, self.payoff). Set to False during optimization to
                avoid side effects.
            include_s: If True, includes social factors in payoff calculation.
                If False, payoff = e (economic only).
            rank: If True, converts scores to relative rankings within the
                social network before aggregation. This is useful during
                optimization to compare performance relative to peers rather
                than absolute values.

        Returns:
            Final payoff value. Range depends on mode:
                - Ranking mode: [0, 1]
                - Absolute mode with social: [0, inf) (depends on e)
                - Absolute mode without social: [0, inf) (same as e)

        Example:
            Calculate payoff during optimization (with ranking):

            ```python
            payoff = city.agg_payoff(
                e=economic_score,
                s=social_score,
                rank=True,  # Compare relative to friends
                record=False  # Don't store during optimization
            )
            ```

            Calculate and record final payoff:

            ```python
            payoff = city.agg_payoff(
                e=economic_score,
                s=social_score,
                rank=False,  # Use absolute scores
                record=True  # Store for analysis
            )
            # Now city.e, city.s, city.payoff are set
            ```

        Note:
            The ranking mechanism uses the `compare` method to normalize
            scores relative to friends in the social network. This creates
            a competitive dynamic where agents compare themselves to peers.
        """
        # Convert economic and social scores to relative rankings among friends
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

    def hate_a_behave(self, behave: DecisionType) -> bool:
        """Determine whether to dislike a behavior, generating social discontent.

        This method implements the judgment mechanism for evaluating others'
        behavior. The decision follows these rules:
            1. If the agent itself is violating rules ("D"), it will not
               criticize others (no moral authority)
            2. If the other agent is complying ("C"), there is nothing to
               criticize
            3. If the other agent is violating ("D"), the agent may or may
               not criticize based on its vengefulness parameter

        Args:
            behave: The other agent's decision behavior ("C" or "D").

        Returns:
            True if the agent dislikes the behavior (will generate social
            discontent), False otherwise.
        """
        # If agent itself is violating rules, has no authority to criticize
        if self.decision == "D":
            return False
        # If other agent is complying, nothing to criticize
        if behave == "C":
            return False
        # If other agent is violating, may criticize based on vengefulness
        return self.random.random() <= self.vengefulness

    def judge_friends(self, willing: DecisionType) -> Tuple[int, int]:
        """Evaluate friends' behavior and calculate social costs.

        This method implements peer evaluation based on multi-cultural theory,
        inspired by research published in Nature Human Behavior [@castillarho2017a].
        The mechanism captures how agents perceive fairness and social norms:

        Key mechanisms:
            - If an agent observes neighbors violating rules while itself
              complies, it experiences social discontent (feeling of unfairness)
            - If an agent's vengefulness is high, it will strongly dislike
              rule-violating friends, reducing both agents' social satisfaction
            - This creates a feedback mechanism where social costs depend on
              both own and neighbors' behavior

        Args:
            willing: The agent's own decision tendency. In the genetic algorithm
                context, this is a temporary decision intention being evaluated.
                If this intention leads to higher payoff, the agent will tend
                to adopt it.

        Returns:
            Tuple of (dislikes, criticized) where:
                - dislikes: Number of friends the agent dislikes (affects
                  social discontent level)
                - criticized: Number of friends who criticize the agent's
                  behavior (affects reputation assessment)

        Note:
            The method iterates through all friends and evaluates each
            relationship bidirectionally.
        """
        # Evaluate each friend
        dislikes, criticized = 0, 0
        for friend in self.friends:
            dislikes += self.hate_a_behave(friend.decision)
            criticized += friend.hate_a_behave(willing)
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
                - "e": Economic score (net economic benefit)
                - "s": Social score (social satisfaction)
                - "payoff": Combined score (e * s)
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
            learned = city.change_mind(metric="payoff", how="best")
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
        # Calculate payoff with optimized water allocation, without ranking, store results
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
        # Learn from better performers and potentially mutate strategy for next year
        self.change_mind(metric="payoff", how="random")
        self.mutate_strategy(probability=self.p["mutation_rate"])
        self.make_decision()
