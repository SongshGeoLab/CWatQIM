# CWatQIM: Crop-Water Quota Irrigation Model

[![Release](https://img.shields.io/github/v/release/SongshGeoLab/CWatQIM)](https://github.com/SongshGeoLab/CWatQIM/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CoMSES](https://img.shields.io/badge/CoMSES-Model-blue)](https://www.comses.net)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

An agent-based model (ABM) for simulating water quota allocation and irrigation decisions in China's Yellow River Basin.

## Overview

CWatQIM (Crop-Water Quota Irrigation Model) is a multi-agent model that simulates the coupled human-water system in the Yellow River Basin. The model investigates how water quota institutions shape irrigation water withdrawal decisions and their system-wide consequences, focusing on the mechanisms through which administrative water quotas influence water source composition (surface water versus groundwater), irrigation efficiency, and crop productivity.

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

- **Zenodo**: [DOI link will be added after first release]
- **CoMSES Net**: [Link will be added after submission]

For citation and archival purposes, please use the Zenodo DOI.

## Quick Start

After cloning the repository, run the model from the repository root:

```bash
# Run with default configuration
python -m cwatqim

# Override configuration parameters
python -m cwatqim exp.repeats=5 exp.num_process=4

# Use a different dataset configuration
python -m cwatqim ds=mac
```

### Using Python API

```python
from cwatqim import CWatQIModel
from hydra import compose, initialize

# Initialize configuration (from config/ directory in repository)
with initialize(config_path="config", version_base=None):
    cfg = compose(config_name="config")

    # Create and run model
    model = CWatQIModel(parameters=cfg)
    model.setup()

    # Run simulation
    for _ in range(10):
        model.step()

    model.end()
```

## Configuration

The repository includes Hydra configurations in the `config/` directory:

- **`config/config.yaml`**: Main configuration with model parameters
- **`config/ds/default.yaml`**: Default dataset paths (uses relative paths)
- **`config/ds/mac.yaml`**: macOS-specific paths (for local development)
- **`config/exp/test.yaml`**: Test experiment configuration
- **`config/exp/exp.yaml`**: Full experiment configuration

You can override any configuration parameter via command line arguments or create your own configuration files.

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
  doi = {10.5281/zenodo.XXXXXXX}
}
```

> **Note**: Please replace `XXXXXXX` with the actual Zenodo DOI after the first release.

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
