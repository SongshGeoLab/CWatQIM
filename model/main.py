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


class CWatQIModel(MainModel):
    """agent-based model for simulating Yellow River water quota allocation.

    This class represents the main model that orchestrates the simulation of
    water quota allocation in the Yellow River Basin. It manages the interactions
    between provinces (water quota allocators) and cities (irrigation units),
    simulating the decision-making processes and social learning mechanisms
    that influence water use compliance.

    The model is built on the ABSESpy framework and integrates with AquaCrop
    for crop yield simulation. It simulates annual time steps where:
        1. Provinces update and allocate water quotas to cities
        2. Cities make irrigation decisions based on economic and social factors
        3. Social networks are updated to reflect information sharing
        4. Agents learn from better-performing neighbors

    Attributes:
        provinces: An ActorsList containing all Province agents in the model.
        cities: An ActorsList containing all City agents in the model.

    Example:
        Create and run a simulation:

        ```python
        from cwatqim import CWatQIModel
        from hydra import compose, initialize

        with initialize(config_path="config"):
            cfg = compose(config_name="config")
            model = CWatQIModel(parameters=cfg)
            model.setup()

            # Run simulation for 20 years
            for _ in range(20):
                model.step()

            model.end()
        ```

    Note:
        The model requires configuration files specifying:
        - City shapefile with City_ID and Province_n attributes
        - Water quota data
        - Climate data for each city
        - Model parameters (social learning rates, mutation rates, etc.)

    See Also:
        - `cwatqim.agents.city.City`: City-level agents
        - `cwatqim.agents.province.Province`: Province-level agents
    """

    def setup(self) -> None:
        """Initialize the model by creating city agents from spatial data.

        This method reads a shapefile containing city boundaries and attributes,
        then creates City agents for each city in the dataset. The setup process
        extracts city identifiers and province assignments from the shapefile
        attributes.

        The shapefile must contain:
            - `City_ID`: Unique identifier for each city (integer)
            - `Province_n`: Province name for each city (string)

        The created City agents will be automatically linked to their respective
        Province agents during the simulation.

        Raises:
            FileNotFoundError: If the city shapefile specified in `self.ds.cities.shp`
                does not exist.
            ValueError: If required attributes are missing from the shapefile.

        Note:
            This method is called automatically by the ABSESpy framework during
            model initialization. It should not be called manually unless
            reinitializing the model.

        See Also:
            - `cwatqim.agents.city.City`: The city agent class being created
        """
        cities = gpd.read_file(self.ds.cities.shp)
        self.agents.new_from_gdf(
            gdf=cities,
            agent_cls=City,
            attrs={"Province_n": "province", "City_ID": "City_ID"},
        )

    @property
    def provinces(self) -> ActorsList[Province]:
        """Get all province agents in the model.

        Returns:
            An ActorsList containing all Province agents. The list supports
            standard ABSESpy operations like filtering, selection, and batch
            operations.

        Example:
            Access all provinces and perform batch operations:

            ```python
            # Get all provinces
            provinces = model.provinces

            # Filter by name
            henan = provinces.select({"name_en": "Henan"})

            # Perform batch update
            provinces.shuffle_do("update_data")
            ```
        """
        return self.agents[Province]

    @property
    def cities(self) -> ActorsList[City]:
        """Get all city agents in the model.

        Returns:
            An ActorsList containing all City agents. Each city represents
            an irrigation unit that makes water use decisions based on economic
            and social factors.

        Example:
            Access all cities and perform operations:

            ```python
            # Get all cities
            cities = model.cities

            # Select a specific city by ID
            city = cities.select({"city_id": 102}).item("only")

            # Get cities in a specific province
            henan_cities = cities.select({"province_name": "Henan"})
            ```
        """
        return self.agents[City]

    def sel_city(self, city_id: Optional[int] = None) -> City:
        """Select a city agent by its unique identifier.

        This method provides a convenient way to retrieve a specific city agent
        from the model. If no city_id is provided, a random city is returned.

        Args:
            city_id: The unique identifier of the city (from the City_ID
                attribute in the shapefile). If None, returns a randomly
                selected city.

        Returns:
            The City agent matching the given city_id, or a random city if
            city_id is None.

        Raises:
            ValueError: If the specified city_id does not exist in the model,
                or if multiple cities match the criteria (should not occur).

        Example:
            Select a specific city:

            ```python
            # Select city with ID 102
            city = model.sel_city(city_id=102)

            # Get a random city
            random_city = model.sel_city()
            ```
        """
        if city_id is None:
            return self.cities.random.choice()
        return self.cities.select({"city_id": city_id}).item("only")

    def sel_prov(self, name_en: Optional[str] = None) -> Province:
        """Select a province agent by its English name.

        This method provides a convenient way to retrieve a specific province
        agent from the model. If no name is provided, a random province is
        returned.

        Args:
            name_en: The English name of the province (e.g., "Henan", "Shandong").
                If None, returns a randomly selected province.

        Returns:
            The Province agent matching the given name, or a random province
            if name_en is None.

        Raises:
            ValueError: If the specified province name does not exist in the
                model, or if multiple provinces match the criteria (should not
                occur with valid province names).

        Example:
            Select a specific province:

            ```python
            # Select Henan province
            henan = model.sel_prov(name_en="Henan")

            # Get a random province
            random_prov = model.sel_prov()
            ```
        """
        if name_en is None:
            return self.provinces.random.choice()
        return self.provinces.select({"name_en": name_en}).item("only")

    def step(self) -> None:
        """Execute one simulation time step (one year).

        This method orchestrates the annual simulation cycle, which includes:
            1. Updating province-level data (water quotas, allocations)
            2. Updating social networks between cities
            3. Executing city-level decisions (irrigation, learning)
            4. Collecting data for analysis
            5. Logging progress

        The execution order is:
            - Provinces update their quota data and allocate to cities
            - Provinces update social networks (friendship links between cities)
            - Cities execute their step() method (irrigation decisions, learning)
            - Data collector records agent states
            - Logger records the current year

        Note:
            This method is called automatically by the ABSESpy framework's
            scheduler. The order of operations (shuffle_do) ensures that
            provinces and cities are processed in random order to avoid
            systematic biases.

        See Also:
            - `cwatqim.agents.province.Province.update_data`: Updates quota data
            - `cwatqim.agents.province.Province.update_graph`: Updates social networks
            - `cwatqim.agents.city.City.step`: City-level decision making
        """
        # preparing parameters
        logger.info(f"Starting a new year: {self.time.year}")
        self.provinces.shuffle_do("update_data")
        self.provinces.shuffle_do("update_graph", l_p=self.p["l_p"])
        self.cities.shuffle_do("step")

        # 收集数据
        self.datacollector.collect(self)

    def end(self) -> None:
        """Finalize the simulation and save results to disk.

        This method is called automatically at the end of the simulation run.
        It performs the following operations:
            1. Creates the output directory if it doesn't exist
            2. Retrieves all collected City agent data from the datacollector
            3. Saves the data to a CSV file named `{run_id}_cities.csv`

        The output CSV file contains all agent variables that were collected
        during the simulation, including:
            - Water use (surface_water, ground_water, total_wu)
            - Water quota (quota)
            - Crop yields (maize, wheat, rice)
            - Economic and social scores (e, s, payoff)
            - Decision variables (decision, willing)
            - And other attributes defined in the model configuration

        Note:
            Validation and analysis of the results should be performed using
            the separate `water_quota_analysis` package, which provides tools
            for DID analysis, indicator calculation, and visualization.

        Raises:
            PermissionError: If the output directory cannot be created or
                the file cannot be written.

        See Also:
            - `water_quota_analysis.analysis.data_loader.DataLoader`: For loading
                and processing simulation results
        """
