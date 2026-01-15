# Sample Data

This directory contains sample data files for running CWatQIM with a subset of cities.

## Required Files

To run the model with sample data, you need the following files:

1. **city_climate/** - Directory containing climate CSV files for each city
   - Format: One CSV file per city (e.g., `C1.csv`, `C2.csv`)
   - Columns: Date, MinTemp, MaxTemp, Precipitation, ReferenceET

2. **quotas.csv** - Provincial water quotas (1e8 m³)
   - Columns: Year, Province1, Province2, ...
   - Index: Year

3. **irr_intensity.csv** - Water use intensity by crop (mm)
   - Columns: Year, City_ID, Maize, Wheat, Rice
   - Index: Year, City_ID

4. **irr_area_ha.csv** - Irrigation area by crop (ha)
   - Columns: Year, City_ID, Maize, Wheat, Rice
   - Index: Year, City_ID

5. **prices.csv** - Crop prices (RMB per tonne)
   - Columns: Year, Province, Rice, Wheat, Maize
   - Index: Year, Province

6. **YR_cities_sample.shp** - Shapefile with sample city boundaries
   - Required attributes: City_ID, Province_Name

7. **city_codes.csv** - City code mapping table
   - Columns: City_ID, City_Name, Province_Name

## Sample Data Preparation

The sample data should include:
- 2-3 representative cities from different provinces
- 5 years of data (e.g., 1985-1990)
- All required climate, quota, and agricultural data

## Note

This is a placeholder directory. Actual sample data files should be prepared based on your specific needs. You can use a subset of your full dataset or create synthetic data for demonstration purposes.
