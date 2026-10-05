# External Domain Datasets and Metadata References

This directory contains external reference datasets, transit network topologies, holiday calendars, and meteorological standards that contextualize and validate the core **10,000-record Metro Traffic Dataset**.

---

## 📁 Files in this Directory:

### 1. `metro_stations_metadata.csv`
Contains the spatial layout, platform capacities, land-use classifications, and geographical coordinates for all 30 metro stations (`Station_ID` 1 through 30).
* **Fields:**
  - `Station_ID`: Primary key matching the operational records in `metro_traffic_dataset_10000.csv`.
  - `Station_Name`: Name of the station.
  - `Line_ID`: Transit line route (Lines 1 to 5).
  - `Zone`: Urban transit zone (Downtown Core, Tech Corridor, Academic District, Airport Link, Suburbs).
  - `Platform_Capacity`: Designed maximum passenger holding capacity.
  - `Interchange_Flag`: Identifies multimodal junctions linking multiple metro lines or interstate rail.
  - `Surrounding_Land_Use`: Commercial, IT, Civic, Residential, Stadium, Industrial.
  - `Latitude`, `Longitude`: Spatial coordinates for GIS mapping and station cluster analysis.

### 2. `transit_lines_metadata.csv`
Defines operational characteristics, route specifications, and design capacities for the 5 metro lines (`Line_ID` 1 to 5).
* **Fields:**
  - `Line_ID`: Transit line index (1 to 5).
  - `Line_Name`: Route nomenclature.
  - `Color_Code`: Standard transit map hex code.
  - `Total_Length_km`: Track route length in kilometers.
  - `Operating_Stations`: Number of active stations on the line.
  - `Peak_Headway_min`: Minimum scheduled minutes between consecutive trains during peak hours.
  - `OffPeak_Headway_min`: Scheduled headway during off-peak and night operations.
  - `Max_Commercial_Speed_kmph`: Maximum speed limit for the rolling stock.
  - `Train_Capacity_Passengers`: Total passenger crush capacity per train.
  - `Route_Description`: Narrative describing the urban corridor served.

### 3. `holiday_and_events_calendar.csv`
Provides the reference calendar mapping dates, day of week, public holidays, and major stadium/civic events that drive the binary indicators:
* `Holiday_Flag` ($1$ = Public Holiday, $0$ = Regular Day)
* `Special_Event_Flag` ($1$ = Major sports match, concert, or festival causing passenger surges)
* Maps each event to impacted station clusters and expected crowd dynamics.

### 4. `weather_classification_standards.csv`
Defines meteorological threshold classifications (IMD and WMO standard transit guidelines) for:
* `Rainfall_mm`: No Rain, Light Rain, Moderate Rain, Heavy Rain / Monsoon.
* `Temperature_C`: Cool, Mild, Warm, Heatwave conditions.
* `Visibility_km`: Low Visibility, Moderate Visibility, Clear line of sight.
* Details the operational dispatch rules, headway adjustments, and dwell time impacts associated with each weather condition.

---

## 🔗 Integration with Machine Learning Pipeline:
* These external tables provide domain validation for model predictions. For instance:
  - High predictions for `Station_ID = 8` during evening hours on weekends coincide with `Special_Event_Flag = 1` events at National Stadium.
  - Slower speeds and elevated dwell times during high `Rainfall_mm` align with the safety dispatch protocols documented in `weather_classification_standards.csv`.
