# Sample data

This directory holds the **complete input dataset** the demo configuration runs on:
59 prefecture-level agents across the eight Yellow River provinces, 1980–2018.
It is not a placeholder and not a subset — `python -m cwatqim` runs end to end
against exactly these files.

Every path below is referenced from `config/demo.yaml`.

## What is here

| File | Rows | Contents |
| --- | --- | --- |
| `city_climate/climate_C<id>.csv` | 61 files | Daily weather per agent, **1980-01-01 → 2018-12-31**. Columns: `MinTemp`, `MaxTemp`, `Precipitation`, `ReferenceET`, `Date`, `City_ID`, `Province`, `Latitude`, `Longitude`. The file name is built as `climate_C{City_ID}.csv`. |
| `quotas.csv` | 34 | Provincial surface-water quotas, 10⁸ m³. One row per year, one column per province (`Gansu`, `Henan`, `Neimeng`, `Ningxia`, `Qinghai`, `Shaanxi`, `Shandong`, `Shanxi`). |
| `irr_intensity.csv` | 2891 | Irrigation water-use intensity, **mm**. Columns: `City_ID`, `Year`, `Province_n`, `Rice`, `Wheat`, `Maize`. |
| `irr_area_ha.csv` | 2891 | Irrigated area, **ha**, same keys and crops. |
| `prices.csv` | 8 | Per-province prices, one row per province keyed by `name_en`. |
| `city_collectivism.csv` | 59 | Prefecture collectivism index (`index_2000/2010/2020` and their z-scores). Only read when `City.s_group_kappa > 0`. |
| `city_tightness.csv` | 59 | Provincial cultural-tightness index. Only read when `City.s_grid_level` is `province` or `prefecture`. |
| `city_network_edges.csv` | 145 | Observed inter-prefecture collaboration edges (`City_ID_a`, `City_ID_b`). Only read when `model.network: observed`. |
| `YR_cities_sample.shp` (+ `.dbf/.shx/.prj/.cpg`) | 59 | Agent geometries. Attributes used by the model: `City_ID`, `Province_n`. |
| `city_codes.xlsx` | 341 | `City_ID` → province/city lookup covering all of China. Shipped for reference; **no code path reads it**. |

## ⚠️ Units, if you substitute your own tables

- **`prices.csv` crop prices are RMB per _kilogram_** (rice ≈ 2.75, wheat ≈ 2.0,
  maize ≈ 1.85). `agents/province.py` multiplies them by 1000 to match yields,
  which AquaCrop reports in tonnes. **A table denominated in RMB per tonne
  inflates crop revenue by 1000×.** Nothing raises: the revenue term simply
  swamps the social term in `U = E · S`, and every compliance result flips.
- `prices.csv` also carries two **water** prices per province, `surface` and
  `ground`, in RMB per m³ (groundwater is a flat 0.68 throughout).
- `irr_intensity.csv` is in **mm**, `irr_area_ha.csv` in **ha**; the model
  multiplies them to get volumes. Mixing in m³ or km² is the other easy way to
  be off by orders of magnitude without an error.
- `quotas.csv` is in **10⁸ m³**, matching how the basin authority publishes them.

## Provenance

Climate is derived from the China Meteorological Forcing Dataset; quotas come
from the Yellow River Conservancy Commission; irrigation intensity and area are
compiled from provincial statistical yearbooks. The collectivism, tightness and
collaboration-network tables are described in the paper's Supplementary
Information, which also states the windows they cover and how they were built.

The simulated period reported in the paper is **1980–2012**; the climate files
run to 2018 so that the window can be extended without re-deriving them.
