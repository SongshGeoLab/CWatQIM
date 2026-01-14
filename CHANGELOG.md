# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1](https://github.com/SongshGeoLab/CWatQIM/compare/cwatqim-v0.1.0...cwatqim-v0.1.1) (2026-01-14)


### Code Refactoring

* **documentation:** enhance package descriptions and docstrings for clarity; improve examples and usage instructions across modules in the CWatQIM framework ([7137485](https://github.com/SongshGeoLab/CWatQIM/commit/71374853be956f98636f1060f6ad0fe2268077e6))
* **project:** remove Nature and update model structure to use CWatQIModel; add main execution script for batch experiments ([8f9196e](https://github.com/SongshGeoLab/CWatQIM/commit/8f9196ef899d930bcf3c8b17db3dc260a65fc1de))
* **project:** seperate the project into model and analysis two parts. ([c3174bf](https://github.com/SongshGeoLab/CWatQIM/commit/c3174bf23c866c53e4ce68b032dcc47f2bc690c0))
* **tests:** update test fixtures for CWatQIModel; enhance documentation and improve model instance creation for clarity and consistency in testing ([d77b2d8](https://github.com/SongshGeoLab/CWatQIM/commit/d77b2d87b288b8fc27153570d13788653149d06a))

## [0.1.0] - 2026-01-14

### Added

- Initial release of CWatQIM (Crop-Water Quota Irrigation Model)
- Province-level water resource management agents
- City-level agricultural irrigation agents
- Water quota allocation mechanisms based on Yellow River "87 Agreement"
- Integration with ABSESpy framework for agent-based modeling
- Support for AquaCrop model integration via aquacrop-abses
- Climate data processing from ERA5 reanalysis
- Groundwater and surface water source switching logic
- Payoff calculation for irrigation decisions
