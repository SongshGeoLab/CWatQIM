# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1](https://github.com/SongshGeoLab/yr-water-quota/compare/v0.1.0...v0.1.1) (2026-01-14)


### Bug Fixes

* **ci:** :bug: improve sync workflow for public repository; add checks for remote branch existence and enhance error handling during subtree pull and push operations ([f446b43](https://github.com/SongshGeoLab/yr-water-quota/commit/f446b431399b4abaa2485457314c157daefe5dcf))
* **ci:** :bug: update release configuration for cwatqim; add package-specific settings and enhance workflow documentation ([16822f1](https://github.com/SongshGeoLab/yr-water-quota/commit/16822f116b95d01947343438873dc5ef28a51785))
* **citation:** :memo: correct indentation for ORCID entry in CITATION.cff file ([acce24d](https://github.com/SongshGeoLab/yr-water-quota/commit/acce24dd9b6a4dfd7ff490c69fe3c17fbe9c832a))
* **metadata:** :memo: update model description and keywords in CWatQIM files; change "multi-agent simulation" to "ABM simulation" and correct author details in CITATION.cff ([4a1b9ce](https://github.com/SongshGeoLab/yr-water-quota/commit/4a1b9ce17516896b1db0a1df606564ff17c181d0))


### Code Refactoring

* **documentation:** enhance package descriptions and docstrings for clarity; improve examples and usage instructions across modules in the CWatQIM framework ([99f402d](https://github.com/SongshGeoLab/yr-water-quota/commit/99f402dea2bcae4f7058fd64ebc752f261f72f2b))
* **project:** remove Nature and update model structure to use CWatQIModel; add main execution script for batch experiments ([2f81153](https://github.com/SongshGeoLab/yr-water-quota/commit/2f811531f52eef75e43d4413c8812aa176fd9e32))
* **project:** seperate the project into model and analysis two parts. ([0e1b86d](https://github.com/SongshGeoLab/yr-water-quota/commit/0e1b86da9a073b91cd42cbc3bd752099e51f73a5))
* **tests:** update test fixtures for CWatQIModel; enhance documentation and improve model instance creation for clarity and consistency in testing ([60e0342](https://github.com/SongshGeoLab/yr-water-quota/commit/60e0342a2494bc6aac8ef062ccbd33d7ed18d6b7))


### Documentation

* **citation:** :memo: add CITATION.cff file for proper software citation and metadata; include author details, abstract, and keywords for CWatQIM ([31135a8](https://github.com/SongshGeoLab/yr-water-quota/commit/31135a800ba86c9f9992ed1c89f0ca5a2fd66a4f))

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
