# CWatQIM: Crop-Water Quota Irrigation Model

[![Release](https://img.shields.io/github/v/release/SongshGeoLab/CWatQIM)](https://github.com/SongshGeoLab/CWatQIM/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4305038.svg)](https://doi.org/10.5281/zenodo.4305038)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CoMSES](https://img.shields.io/badge/CoMSES-Model-blue)](https://www.comses.net)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

An agent-based model (ABM) for simulating water quota allocation and irrigation decisions in China's Yellow River Basin.

## Overview

CWatQIM (Crop-Water Quota Irrigation Model) is a agent-based model that simulates the coupled human-water system in the Yellow River Basin. The model investigates how water quota institutions shape irrigation water withdrawal decisions and their system-wide consequences, focusing on the mechanisms through which administrative water quotas influence water source composition (surface water versus groundwater), irrigation efficiency, and crop productivity.

### Key Features

- **Multi-scale agents**: Province-level and prefecture-level (city) agents representing water management agencies
- **Crop modeling integration**: Built-in integration with AquaCrop for crop yield simulation
- **Social learning mechanisms**: Implements Standing strategy (evolutionary game theory) for behavioral adaptation
- **Policy analysis**: Enables counterfactual analysis to assess policy effects under different enforcement regimes

## Installation

### From GitHub (Recommended)

Clone the repository to get the full model with configurations:

```bash
git clone https://github.com/SongshGeoLab/CWatQIM.git
cd CWatQIM
pip install -e .
```

### From PyPI

```bash
pip install cwatqim
```

## Publication

This model is published on:

- **Zenodo**: [10.5281/zenodo.4305038](https://doi.org/10.5281/zenodo.4305038)
- **CoMSES Net**: [Link will be added after submission]

For citation and archival purposes, please use the Zenodo DOI.

## Quick Start

After cloning the repository, run the model from the `cwatqim` directory (the package root):

```bash
cd cwatqim

# Run with demo configuration (uses sample data in data/sample/)
python -m cwatqim config_name=demo

# Override configuration parameters
python -m cwatqim config_name=demo exp.repeats=5 exp.num_process=4

# Override time range
python -m cwatqim config_name=demo time.start=1985 time.end=1990
```

### Using Python API

```python
from cwatqim import CWatQIModel
from hydra import compose, initialize

# Initialize configuration (from config/ directory in cwatqim package)
with initialize(config_path="config", version_base=None):
    cfg = compose(config_name="demo")  # Use demo configuration

    # Create and run model
    model = CWatQIModel(parameters=cfg)
    model.setup()

    # Run simulation
    for _ in range(10):
        model.step()

    model.end()
```

## Configuration

The package includes a demo configuration file for quick start:

- **`config/demo.yaml`**: Complete demo configuration with sample data paths

The demo configuration uses sample data located in `data/sample/` directory, which includes:
- City climate data (`city_climate/` directory)
- City boundaries shapefile (`YR_cities_sample.*`)
- City code mapping (`city_codes.xlsx`)
- Water quotas (`quotas.csv`)
- Irrigation data (`irr_intensity.csv`, `irr_area_ha.csv`)
- Crop prices (`prices.csv`)

All paths in `demo.yaml` are relative to the `cwatqim` package root directory. You can override any configuration parameter via command line arguments or create your own configuration files.

For example, to change the simulation time range:
```bash
python -m cwatqim config_name=demo time.start=1985 time.end=1990
```

## Model Components

### Agents

- **Province**: Province-level agents managing water quota allocation
- **City**: Prefecture-level agents making irrigation water withdrawal decisions
- **Farmer**: Individual farmer agents (optional, for future extensions)

### Core Modules

- **CWatQIModel**: Main model class orchestrating the simulation
- **Algorithms**: Optimization algorithms for water source portfolio decisions
- **Data Loaders**: Utilities for loading climate, quota, and agricultural data
- **Payoff**: Economic and social payoff calculations

## Documentation

- [ODD+D Protocol](docs/ODD+D.md) - Complete model description following the ODD+D protocol
- [CHANGELOG](CHANGELOG.md) - Version history and release notes
- [API Reference](https://github.com/SongshGeoLab/CWatQIM) - Full API documentation

## Requirements

- Python >=3.11, <3.12
- See [pyproject.toml](pyproject.toml) for full dependency list

## Citation

If you use this model in your research, please cite:

```bibtex
@software{cwatqim2026,
  title = {CWatQIM: Crop-Water Quota Irrigation Model},
  author = {Song, Shuang},
  year = {2026},
  url = {https://github.com/SongshGeoLab/CWatQIM},
  doi = {10.5281/zenodo.4305038}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Supported by National Natural Science Foundation of China (No. 42041007, No. U2243601)
- Built on the [ABSESpy](https://github.com/AB-SES/absespy) framework
- Integrates with [AquaCrop](https://www.fao.org/aquacrop) for crop modeling

## Contact

- **Author**: Shuang Song
- **Email**: songshgeo@gmail.com
- **GitHub**: [@SongshGeo](https://github.com/SongshGeo)
- **Website**: [https://cv.songshgeo.com/](https://cv.songshgeo.com/)
