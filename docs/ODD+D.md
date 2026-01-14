---
csl: nature
---

# Appendix S3: Agent-based Model ODD+D Protocol

## 1.1 Overview

### 1.1.1 Purpose

The agent-based model (ABM) presented here simulates the coupled human-water system in the Yellow River Basin (YRB) to investigate how water quota institutions shape irrigation water withdrawal decisions and their system-wide consequences. The model is designed to understand the mechanisms through which administrative water quotas, implemented at the provincial level and enforced at the prefecture level, influence the composition of water sources (surface water versus groundwater), irrigation efficiency, and crop productivity. The model enables counterfactual analysis to assess policy effects by comparing scenarios with different enforcement regimes.

The model is designed for river basin scientists, water resource managers, decision-makers, and policy analysts interested in understanding the complex trade-offs arising from water quota institutions in transboundary river basins. It provides a mechanistic framework for analyzing how local agencies respond to institutional constraints and how these responses cascade through the coupled human-water system.

### 1.1.2 Entities, State Variables, and Scales

#### 1.1.2.1 Entities

The model contains two types of agents. Prefecture-level agents (N = 59) represent prefecture-level water management agencies that make decisions regarding irrigation water withdrawals. Each prefecture-level agent corresponds to a prefecture-level administrative unit through which the Yellow River flows. These agents are the primary decision-making units that optimize their water source portfolio (surface water and groundwater) to maximize their utility while facing quota constraints.

#### 1.1.2.2 State Variables

Prefecture-level agents maintain several categories of state variables. Water-related variables include the surface water quota allocated from the province (in units of 1e8 m³, hundred million cubic meters), actual surface water and groundwater withdrawals (both in 1e8 m³), total water use (surface plus groundwater, in 1e8 m³), and net irrigation water after accounting for conveyance and on-farm losses (in 1e8 m³). Agricultural variables include irrigation area by crop type (maize, wheat, rice) in hectares, water use intensity by crop type in millimeters, actual crop yield by crop type in tonnes per hectare, and potential crop yield by crop type. Behavioral traits include boldness, which represents the probability of choosing to defect (exceed quota) and ranges from 0 to 1, vengefulness, which represents sensitivity to others' non-compliance and ranges from 0 to 1, and willing, which is the current decision intention, either "C" (compliance) or "D" (defect). Performance metrics include the economic payoff (revenue minus water costs), the social payoff (compliance-related satisfaction, ranging from 0 to 1), and the aggregate utility (the product of economic and social payoffs, ranging from 0 to 1 when ranked). Spatial attributes include a unique identifier for the prefecture, the name of the province to which the prefecture belongs, and the spatial geometry of the prefecture boundary. Prefecture-level agents also have dynamic variables for irrigation area, water use intensity, and a dynamic quota variable indexed by year. More information can be found in Table S1.3.

#### 1.1.2.3 Spatial and Temporal Scales

The model covers the Yellow River Basin, encompassing 8 provinces and 59 prefecture-level cities. The spatial resolution is 0.1 degrees (approximately 11 km at the equator, or 121 km² per cell). The temporal extent spans from 1980 to 2010 (31 years) with annual time steps. The strict enforcement regime begins in 1998, which corresponds to year 19 of the simulation and serves as the policy intervention point. Model selection ended in 2010 because after that, a new "strictest water resources management" system changed the administrative incentives for prefecture-level agencies after then.

### 1.1.3 Process Overview and Scheduling

The model follows an annual cycle (see Figure S1.1 for the workflow). At each time step (year), the sequence of processes begins with province-level updates. Next, prefecture-level decision-making occurs in random order. Each prefecture-level agent loads annual climate data and updates dynamic variables including irrigation area and water use intensity. Each agent then simulates crop growth using AquaCrop to estimate irrigation requirements and potential yields. Based on the agent's boldness trait and policy constraints, each agent determines its decision intention (willing). Each agent then optimizes its water source portfolio (surface water versus groundwater) to maximize utility and records actual water withdrawals and crop yields. Following decision-making, social learning occurs in random order. Each prefecture-level agent compares its performance (payoff) with connected neighbors. Agents may adopt behavioral traits (b and v, see Appendix S1) from better-performing neighbors and may mutate their strategy with a small probability (parameter u, also see Appendix S1).

## 1.2 Design Concepts

### 1.2.1 Basic Principles

The model is grounded in several theoretical frameworks. First, evolutionary game theory provides the foundation, where agents engage in a public goods game in which compliance with water quotas represents cooperation and violation represents defection. The social payoff component captures how agents' utility depends on their own actions and the actions of others in their network. Second, the Plural Rationality Approach (PRA) framework informs the social cost calculation, positing that agents' compliance behavior depends on two dimensions: individualism/collectivism and norm valuation. These dimensions are captured by the boldness and vengefulness traits. Third, agents are boundedly rational utility maximizers. They optimize their water source portfolio within constraints but do not have perfect information or computational capacity. Fourth, the model explicitly couples human decision-making (social subsystem) with biophysical processes (natural subsystem through AquaCrop), enabling feedback between irrigation decisions and crop productivity.

### 1.2.2 Emergence

System-level patterns emerge from individual agent decisions. When surface water quotas are strictly enforced, agents shift to groundwater to meet irrigation needs. This groundwater substitution emerges with spatial heterogeneity in compliance that some provinces (e.g., Shandong, Ningxia, Neimeng, Henan, and Gansu) consistently exceeding quotas while others comply. Efficiency improvements emerge as agents face stricter constraints and optimize their water use. This improvement emerges from the optimization process rather than being explicitly programmed. Finally, productivity gains emerge despite reduced surface water availability.

### 1.2.3 Adaptation

The agent includes two key adaptation mechanisms: adapting to climate change and adapting to policy changes. Climate change affects the volume of water potentially needed for irrigation, while policy changes affect the amount of water available for irrigation. The agent adapts by adjusting the components of water use.

### 1.2.4 Objectives

Prefecture-level agents seek to maximize their aggregate utility $U_{i,t}=f\left(\widetilde{c}_{e,i,t},\widetilde{c}_{s,i,t}\right)=\widetilde{c}_{e,i,t}\cdot\widetilde{c}_{s,i,t}$ (i.e., Appendix S1, Equation 5), where $\widetilde{c}_{e,i,t}$ is the ranked economic payoff (crop revenue minus water abstraction costs) and $\widetilde{c}_{s,i,t}$ is the ranked social payoff (compliance-related satisfaction, ranging from 0 to 1).

### 1.2.5 Learning

Social learning occurs at the end of each year, when agents compare their performance (payoff) with connected neighbors. If neighbors perform better, agents adopt the neighbors' behavioral traits, allowing successful strategies to diffuse through the network. Strategy mutation occurs with a small probability, where agents randomly reset either boldness or vengefulness to a random value in [0, 1]. This prevents the population from converging to local optima and maintains behavioral diversity. Learning occurs annually, allowing agents to respond to changing conditions such as policy enforcement and climate variability. Also see Appendix S1, Equation (7).

### 1.2.6 Prediction

Agents use the AquaCrop crop model to predict crop yields and irrigation requirements. For each prefecture $i$ and crop $c$ (maize, wheat, rice) in year $t$, AquaCrop provides the irrigation water requirement $W_{i,c,t}^{req}$ and the realized yield $Y_{i,c,t}$ as a function of local climate, soils, and crop parameters (see Appendix S1, Equation 1). Agents do not predict future climate or policy changes; they make decisions based on current-year conditions and historical learning.

### 1.2.7 Sensing

Agents sense several types of information as summarized in Appendix S1 Table S1.3.

### 1.2.8 Interaction

Interactions occur through several mechanisms. Direct interactions occur through the social network, where prefecture-level agents are connected and can compare their payoff ranks. Agents observe neighbors' decisions, performance, and traits, and agents' social payoffs depend on neighbors' compliance behavior. Indirect interactions occur through institutional mechanisms, as quota allocation creates competition among prefectures within the same province. Policy enforcement affects all agents simultaneously. Environmental feedback occurs as agents' water withdrawal decisions affect crop yields through AquaCrop, and crop yields feed back into economic payoffs, influencing future decisions.

### 1.2.9 Stochasticity

Several processes involve randomness. During initialization, boldness and vengefulness are initialized as random values in [0, 1] for each prefecture-level agent, and the initial willing decision is determined randomly based on boldness. Social network formation involves each Province agent creating links between prefecture-level agents with probability $l_p$, and the network structure varies each year (recreated annually). In decision-making, the willing decision is stochastic: willing equals "D" with probability boldness, otherwise "C". The optimization algorithm (differential evolution) uses random initialization and mutation. In social learning, when multiple better-performing neighbors exist, selection can be random, and strategy mutation occurs with probability. Social cost calculation involves random draws to determine if an agent dislikes a neighbor's behavior, based on vengefulness. Stochasticity ensures that the model can explore multiple pathways and results are not deterministic. Multiple runs (5 replicates) are used to assess robustness.

### 1.2.10 Collectives

All agents are ranked collectively when evaluating payoffs (see Appendix S1 equation 5) and then learn better strategies from the collective (see Appendix S1 equation 7).

### 1.2.11 Observation

The model collects data for analysis at multiple levels. Agent-level variables are collected annually for each prefecture-level agent, including water use variables (quota, surface water, ground water, total water use, net irrigation), agricultural variables (irrigation area by crop, actual crop yield by crop, potential crop yield by crop), performance variables (economic payoff, social payoff, aggregate utility), and behavioral variables (boldness, vengefulness, willing, decision). Aggregated outputs include basin-scale totals (sum across all prefectures), province-level aggregates, crop-specific aggregates, and efficiency indicators (calculated post-simulation). All variables are collected at the end of each time step (year), stored in CSV format for post-simulation analysis, and automatically recorded by the model's datacollector for all specified agent variables.

## 1.3 Details

### 1.3.1 Initialization

The model is initialized in a specific sequence. First, the model class is instantiated with configuration parameters, and set up the spatial environment. Second, the spatial environment is initialized by loading prefecture boundaries from a shapefile and applying irrigation area and soil type raster data to grid cells if available. Third, prefecture-level agents are created from the shapefile, with each agent assigned a unique identifier and province name from the shapefile attributes, and each agent's spatial geometry stored. Fourth, all agents are linked within a fully-connected social network. Fifth, prefecture-level agents are initialized by loading dynamic variables including water use intensity by crop type and irrigation area by crop type from CSV files indexed by year. Behavioral traits are initialized with boldness and vengefulness set to random values in [0, 1], and willing determined stochastically based on boldness. Water-related variables are initialized to 0.0, and performance metrics are initialized with economic payoff set to 0.0, social payoff set to 1.0, and aggregate utility set to 0.0. Sixth, Provinces are initialized by loading the dynamic quota variable from a CSV file indexed by year, loading water prices and crop prices from CSV files, and loading irrigation efficiency coefficients from parameters. Seventh, time is initialized with the simulation start year set to 1980 and the time step counter initialized. The model is ready to begin the first time step after initialization.

### 1.3.2 Input Data

The model requires several categories of input data. Spatial data includes prefecture boundaries from a shapefile of prefecture-level administrative units, irrigation area from raster data indicating irrigated areas, and soil type from raster data of soil classifications (optional, defaults to loam). Climate data comes from the China Meteorological Forcing Dataset (CMFD)2, including daily precipitation in millimeters, maximum and minimum temperature in degrees Celsius, and reference evapotranspiration in millimeters (spatial resolution is 0.1 degrees, temporal coverage spans 1980-2010). In the ABM the format is pre-processed CSV files, one per prefecture.

Institutional data includes annual provincial quotas from the "87 Water Allocation Scheme" in a CSV file indexed by province and year, and the policy enforcement year is 1998 (when the hard constraint begins). Economic data includes water prices (surface water and groundwater prices by province in a CSV file) and crop prices (rice, wheat, and maize prices by province from the China Agricultural Price Survey Yearbook in a CSV file). Water prices are in RMB per cubic meter, and crop prices are in RMB per tonne. Agricultural data includes annual irrigation area by crop type and prefecture in a CSV file indexed by prefecture and year (units: hectares) and annual water use intensity by crop type and prefecture in a CSV file indexed by prefecture and year (units: millimeters), sourced from the national water resources survey and statistical yearbooks. Technical parameters include irrigation efficiency coefficients for surface water and groundwater by province (from literature, accounting for conveyance and on-farm losses) and AquaCrop parameters for rice, wheat, and maize (calibrated against provincial yield statistics). All data files are specified in the configuration file. Missing data for specific provinces or crops are handled by the model (e.g., some provinces lack rice price data).

### 1.3.3 Submodels

The Crop-Water Quota Irrigation Model (CWatQIM) integrates a crop growth model (AquaCrop) as the natural subsystem and an agent-based model (ABM) as the social subsystem. The AquaCrop, developed by FAO, simulates crops yield under different climate and irrigation conditions, balancing accuracy, simplicity, and robustness. See Appendix S1 for more details about the two submodels.

## 1.4 Decision-Making

### 1.4.1 Decision Subjects and Objects

Prefecture-level agents (prefecture-level water management agencies) are the primary decision-makers in the model. The main decision is the allocation of irrigation water between surface water and groundwater sources. Specifically, agents decide how much surface water to withdraw ($W_s$), how much groundwater to withdraw ($W_g$), and whether to comply with or violate the surface water quota ($Q_i$). Decision-making occurs at the individual agent level (prefecture-level). There is no collective decision-making; each agent makes independent decisions based on its own constraints and objectives.

### 1.4.2 Basic Rationality

Agents are boundedly rational utility maximizers. They seek to maximize their aggregate payoff ($U$), have limited information (only know their own and neighbors' states), use heuristic optimization (differential evolution) rather than analytical solutions, make decisions based on current-year conditions (no forward-looking behavior), and adapt behavior through social learning (backward-looking). This bounded rationality reflects real-world constraints: water managers cannot perfectly predict future conditions, have limited computational resources, and must make decisions under uncertainty.

### 1.4.3 Objectives and Success Criteria

The primary objective is to maximize aggregate utility $U_i=E_i\times S_i$. The economic component ($E_i$) involves maximizing crop revenue ($c(P_c\cdot Y_c\cdot A_c)$) and minimizing water costs ($(C_s\cdot W_s+C_g\cdot W_g)$), resulting in a net economic payoff that can be negative. The social component ($S_i$) involves maximizing compliance-related satisfaction, avoiding reputational costs from non-compliance, and avoiding social dissatisfaction from others' non-compliance, with a range from 0 (maximum dissatisfaction) to 1 (full satisfaction). Success criteria involve agents comparing their payoff (ranked utility) with neighbors, where higher payoff indicates better performance. Agents learn from better-performing neighbors. There is no explicit threshold for "success"; performance is relative.

### 1.4.4 Decision Process

The decision-making process occurs in several stages.

**Stage 1** is the preliminary intention decision. At the beginning of each time step, each agent determines its willing (decision intention). If the policy is strictly enforced (year >= 1998 in the strict scenario), agents are forced to comply (willing equals "C"). Otherwise, agents stochastically choose based on boldness: the probability of choosing "D" equals boldness. This intention determines the feasible region for optimization but does not guarantee the final decision, as optimization may find that compliance is optimal even if it was willing to defect.

**Stage 2** is boundary condition determination. Based on willing, the agent determines the upper bound for surface water withdrawal. If willing equals "C", surface water is constrained by quota, so the upper bound equals the minimum of quota and total irrigation demand. If willing equals "D", surface water can equal total demand in the baseline scenario. In the strict enforcement scenario (post-1998), the hard constraint $W_s\le Q_i$ is always enforced regardless of willing.

**Stage 3** is optimization. The agent optimizes its water source portfolio using differential evolution. For objective function evaluation, given a candidate $W_s$, the agent calculates $W_g=W_{total}-W_s$, calculates available water as $Q_{lim}=W_s\cdot(1-\eta_{sw})+W_g\cdot(1-\eta_{gw})$, runs AquaCrop simulation with $Q_{lim}$ to estimate crop yields, calculates economic payoff as crop revenue minus water costs, calculates social payoff as a function of $W_s$, quota, and neighbors' decisions, and returns utility $U=E\times S$. The optimization algorithm searches for $W_s^\ast$ that maximizes $U$ within the search space $W_s\in[0,W_{ub}]$ using algorithm parameters including population size multiplier of 15, maximum iterations of 100, and L-BFGS-B polishing. The solution yields optimal surface water ($W_s^\ast$) and optimal groundwater ($W_g^\ast=W_{total}-W_s^\ast$).

### 1.4.5 Decision Rules

Quota constraints differ by scenario. In the baseline scenario, the constraint is soft (violation incurs social cost but is allowed). In the strict enforcement scenario (post-1998), the hard constraint $W_s\le Q_i$ is enforced in optimization bounds. The water balance constraint $W_s+W_g=W_{total}$ is always enforced, where $W_{total}$ is determined exogenously from statistical data (irrigation area times water use intensity). Non-negativity constraints $W_s\geq0$ and $W_g\geq0$ are enforced in optimization bounds.

### 1.4.6 Adaptation to Changing Conditions

Agents adapt their behavior in response to several factors. Policy changes occur when strict enforcement begins in 1998, forcing agents to comply (willing equals "C"). This changes the feasible region, leading to different optimal solutions, and agents may shift to groundwater to compensate for reduced surface water. Climate variability affects crop water requirements through AquaCrop, and agents adjust their water source portfolio to meet varying demands. There is no explicit climate prediction; decisions are based on current-year conditions. Social learning allows agents to observe neighbors' performance and adapt traits. Successful strategies (high payoff) spread through the network, creating spatial contagion in compliance behavior. Economic conditions change when water prices or crop prices change (though in this study, are exogenous and input data is static).

### 1.4.7 Role of Social Norms and Cultural Values

Social norms are modeled through the PRA framework. Norm valuation is captured by vengefulness, which represents how much agents value compliance norms. High vengefulness means agents strongly dislike non-compliance, while low vengefulness means agents are tolerant of non-compliance. Individualism versus collectivism is captured by boldness, which represents individual risk-taking versus collective compliance. High boldness means agents are more likely to defect, while low boldness means agents prefer compliance. Social pressure is enabled by the network structure, where agents observe neighbors' behavior and adjust their social payoffs accordingly. Non-compliance becomes less costly when many neighbors also defect. Cultural transmission occurs as traits (boldness and vengefulness) are transmitted through social learning, allowing norms to evolve over time. Successful strategies (high utility) become more common.

### 1.4.8 Spatial Aspects

Spatial aspects influence decisions through several mechanisms. Climate heterogeneity means each prefecture has its own climate time series, climate affects crop water requirements through AquaCrop, and spatial variation in climate creates heterogeneity in irrigation needs. Administrative boundaries create a hierarchical structure where prefectures belong to provinces, quota allocation occurs within provinces, and social networks are fully connected within provinces, reflecting that government performance is publicly reported and easily comparable. Spatial aggregation means decisions are made at the prefecture level (aggregated across space), there is no explicit spatial interaction beyond social networks, and irrigation area and water use are aggregated to the prefecture level.

### 1.4.9 Temporal Aspects

Temporal aspects play important roles. Annual time steps mean decisions are made annually (matching agricultural and administrative cycles), climate data varies annually affecting water requirements, and quota allocations may change annually from time series. Policy timing means strict enforcement begins in 1998 (year 19), creating a clear before/after comparison for policy analysis, and agents adapt to policy changes within the same year. Learning timing means social learning occurs at the end of each year (after decisions are made), agents observe outcomes before adapting, and this creates a lag between policy change and behavioral adaptation. Historical memory means agents do not explicitly remember past years, learning is based on current-year performance only, but trait evolution through learning creates implicit memory.

### 1.4.10 Uncertainty

Uncertainty is incorporated through several mechanisms. Stochastic decision-making means the willing decision is stochastic (based on boldness), optimization uses random initialization and mutation, and social cost calculations involve random draws. Strategy mutation means random mutation maintains diversity and prevents convergence to local optima. Model uncertainty means AquaCrop model predictions have uncertainty (not explicitly modeled), parameter uncertainty (e.g., irrigation efficiency) is not propagated, and multiple model runs (replicates) are used to assess robustness. Agents do not explicitly account for uncertainty in their decisions; they optimize based on expected outcomes (AquaCrop predictions and current information). Given this paper's focus on institutional-level impacts, we conducted a sensitivity analysis on several key parameters (mutation rate $u$, see Appendix S1, Equation 8; group and grid, see Appendix S1, Figure S1.2), while most of the remaining parameters are exogenous (summarized in Appendix S1, Table S1.3).
