#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Literal, Optional, TypeAlias

import geopandas as gpd
from abses import ActorsList, MainModel
from loguru import logger

from ..agents.city import City, validate_policy_years
from ..agents.province import Province
from ..core.algorithms import require_one_of
from ..core.network import NETWORK_MODES, isolated_city_ids, load_city_links

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
            ValueError: If required attributes are missing from the shapefile,
                or if the policy timeline is one this model refuses to run
                (see `cwatqim.agents.city.validate_policy_years`).

        Note:
            This method is called automatically by the ABSESpy framework during
            model initialization. It should not be called manually unless
            reinitializing the model.

        See Also:
            - `cwatqim.agents.city.City`: The city agent class being created
        """
        # 情景是一次运行的属性，不是每个主体各自的属性，所以在这里查一次，
        # 而不是在 59 个 `City.setup` 里各查一遍；放在读 shapefile 之前，
        # 配错情景时零 I/O 就失败（见 #59）。
        validate_policy_years(
            forced_since=self.settings.City["forced_since"],
            include_s_since=self.settings.City["include_s_since"],
        )
        cities = gpd.read_file(self.ds.cities.shp)
        self.agents.new_from_gdf(
            gdf=cities,
            agent_cls=City,
            attrs={"Province_n": "province", "City_ID": "City_ID"},
            # 灌溉策略必须在构造时注入：`Farmer.__init__` 先调 `super().__init__()`
            # （那才是跑 `setup()` 的地方），之后才写自己的默认值 4，所以在
            # `City.setup` 里赋值一定会被盖掉——主体自报 Net Irrigation，而
            # `simulate` 实际跑的是配置里的 Soil Moisture Targets（见 #62）。
            irr_method=self.settings.City["irr_method"],
        )
        self._warn_about_isolated_cities()

    def _warn_about_isolated_cities(self) -> None:
        """Log which agents the observed network leaves with no friends.

        Isolation is a real property of that data — four agents have it — so it
        must not be fatal. But it must not be **silent** either: an isolated
        agent has `s = 1` on both branches, so the social channel is off for it
        and it never learns from anyone. A reader comparing scenarios deserves
        to know that from the run log rather than by re-deriving it.

        Does nothing under `within_province`, where every agent has its whole
        province as friends by construction.

        Raises:
            ValueError: `model.network` is not one of `NETWORK_MODES`. Checked
                here so a typo fails during `setup`, with zero file I/O.
            FileNotFoundError: `observed` is selected but the edge table is
                missing — also worth hitting at setup rather than a year in.
        """
        if self._network_mode() != "observed":
            return
        links = load_city_links(str(self.ds.city_network))
        isolated = isolated_city_ids(links, [city.city_id for city in self.cities])
        if isolated:
            logger.warning(
                f"协作网络下有 {len(isolated)} 个城市没有任何好友，"
                f"社会通道对它们关闭：{['C%d' % i for i in isolated]}"
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

    def _network_mode(self) -> str:
        """Read and validate `model.network`.

        A separate method because two callers need it: `setup` checks it before
        any file is opened, so a typo fails with zero I/O (the same convention
        as `validate_policy_years`, see #59), and `update_network` reads it
        every year.

        Returns:
            The validated mode, one of `NETWORK_MODES`.

        Raises:
            ValueError: The configured mode is not one of `NETWORK_MODES`.
                Silently falling back to either branch would be worse than
                failing: the two are different mechanisms, and a typo would
                quietly produce results for the one nobody asked for.
        """
        return require_one_of(
            "model.network",
            str(self.p.get("network", "within_province")),
            NETWORK_MODES,
        )

    def update_network(self) -> None:
        """Rebuild the "friend" network for this year, from the chosen source.

        Dispatches on `model.network`:

        * `within_province` — every pair of cities inside a province is linked
          with probability `l_p`, which is what `Province.update_graph` has
          always done. **This branch is bit-identical to the model before the
          observed network existed**, down to the random numbers: it is the
          same call in the same order, so the golden fingerprint holds.
        * `observed` — the collaboration network read from `ds.city_network`.
          No random numbers are drawn at all, which is itself a difference:
          `random.link` calls the RNG once per candidate pair even at `p=1.0`,
          so the two branches cannot be compared draw for draw. They are
          different mechanisms, not two settings of one.

        Raises:
            ValueError: `model.network` is not one of `NETWORK_MODES`.
            FileNotFoundError: `observed` is selected and the edge table is
                missing.

        Note:
            Links are re-added every year in both branches. That is harmless —
            `add_a_link` keeps an existing link in its original position — and
            it is what the old code did, so the ordering that `City.friends`
            depends on is unchanged.

        See Also:
            - `cwatqim.core.network`: the loader, and the three mismatches that
              come with the observed topology.
        """
        mode = self._network_mode()
        if mode == "within_province":
            self.provinces.shuffle_do("update_graph", l_p=self.p["l_p"])
            return
        links = load_city_links(str(self.ds.city_network))
        by_id = {city.city_id: city for city in self.cities}
        for city_id, neighbours in links.items():
            source = by_id.get(city_id)
            if source is not None:
                source.link_friends(by_id.get(n) for n in neighbours)

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
        self.update_network()
        # 冻结同侪观察集：所有人整年看到的都是同一个 t−1 断面。放在 shuffle_do
        # 之前是必须的——`decision` 读 `surface_water`，而它在 step 里会被覆盖，
        # 城市又是乱序推进的（见 issue #72 与 `City.last_decision`）。
        # 不消耗随机数，所以按固定顺序执行，不用 shuffle_do。
        self.cities.do("snapshot_decision")
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

        A single (non-batch) run has no `run_id`; it is written as
        `0_cities.csv`, so that the analysis layer — which parses the run id
        back out of the file name — can still read it.

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
        logger.info("Simulation ends.")
        # 确保输出目录存在
        self.outpath.mkdir(parents=True, exist_ok=True)
        df_cities = self.datacollector.get_agent_vars_dataframe("City")
        # 单次运行没有 run_id，用 0 占位：分析层按文件名前缀解析整数 run_id，
        # 落成 `None_cities.csv` 会被它静默跳过
        run_id = 0 if self.run_id is None else self.run_id
        outfile = self.outpath / f"{run_id}_cities.csv"
        df_cities.to_csv(outfile)
        logger.info(f"City records saved to {outfile}.")
