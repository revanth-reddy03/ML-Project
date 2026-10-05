import os
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT_DIR / "data" / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Metro Stations Metadata (Stations 1 to 30)
# -----------------------------------------------------------------------------
stations_data = [
    {"Station_ID": 1, "Station_Name": "Central Grand Terminal", "Line_ID": 1, "Zone": "Zone 1 - Core Downtown", "Platform_Capacity": 5000, "Interchange_Flag": 1, "Surrounding_Land_Use": "Commercial & Intermodal Hub", "Latitude": 12.9716, "Longitude": 77.5946},
    {"Station_ID": 2, "Station_Name": "Financial District", "Line_ID": 1, "Zone": "Zone 1 - Core Downtown", "Platform_Capacity": 4200, "Interchange_Flag": 0, "Surrounding_Land_Use": "Corporate & Banking", "Latitude": 12.9754, "Longitude": 77.5982},
    {"Station_ID": 3, "Station_Name": "City Hall & High Court", "Line_ID": 1, "Zone": "Zone 1 - Core Downtown", "Platform_Capacity": 3800, "Interchange_Flag": 0, "Surrounding_Land_Use": "Civic & Government", "Latitude": 12.9790, "Longitude": 77.5912},
    {"Station_ID": 4, "Station_Name": "Market Square", "Line_ID": 1, "Zone": "Zone 1 - Core Downtown", "Platform_Capacity": 4000, "Interchange_Flag": 0, "Surrounding_Land_Use": "Retail & Mixed Use", "Latitude": 12.9650, "Longitude": 77.5850},
    {"Station_ID": 5, "Station_Name": "North Interchange", "Line_ID": 1, "Zone": "Zone 1 - Core Downtown", "Platform_Capacity": 4800, "Interchange_Flag": 1, "Surrounding_Land_Use": "Commercial Transit Hub", "Latitude": 12.9860, "Longitude": 77.5920},
    {"Station_ID": 6, "Station_Name": "Tech Park South", "Line_ID": 1, "Zone": "Zone 2 - Tech Corridor", "Platform_Capacity": 4500, "Interchange_Flag": 0, "Surrounding_Land_Use": "IT & Business Park", "Latitude": 12.9352, "Longitude": 77.6245},
    {"Station_ID": 7, "Station_Name": "Innovation Valley", "Line_ID": 2, "Zone": "Zone 2 - Tech Corridor", "Platform_Capacity": 4600, "Interchange_Flag": 0, "Surrounding_Land_Use": "R&D & Tech Hub", "Latitude": 12.9280, "Longitude": 77.6320},
    {"Station_ID": 8, "Station_Name": "National Stadium", "Line_ID": 2, "Zone": "Zone 1 - Core Downtown", "Platform_Capacity": 6000, "Interchange_Flag": 1, "Surrounding_Land_Use": "Sports Complex & Events", "Latitude": 12.9610, "Longitude": 77.6010},
    {"Station_ID": 9, "Station_Name": "University Main Campus", "Line_ID": 2, "Zone": "Zone 2 - Academic Hub", "Platform_Capacity": 3500, "Interchange_Flag": 0, "Surrounding_Land_Use": "Educational Institutions", "Latitude": 12.9420, "Longitude": 77.5680},
    {"Station_ID": 10, "Station_Name": "Medical City & Hospital", "Line_ID": 2, "Zone": "Zone 2 - Academic Hub", "Platform_Capacity": 3400, "Interchange_Flag": 0, "Surrounding_Land_Use": "Healthcare & Research", "Latitude": 12.9510, "Longitude": 77.5750},
    {"Station_ID": 11, "Station_Name": "East Junction", "Line_ID": 2, "Zone": "Zone 2 - Transit Node", "Platform_Capacity": 4400, "Interchange_Flag": 1, "Surrounding_Land_Use": "Interchange & Mixed Use", "Latitude": 12.9680, "Longitude": 77.6400},
    {"Station_ID": 12, "Station_Name": "Harbor Bay Terminal", "Line_ID": 2, "Zone": "Zone 3 - Coastal/Port", "Platform_Capacity": 3000, "Interchange_Flag": 0, "Surrounding_Land_Use": "Maritime & Tourism", "Latitude": 12.9910, "Longitude": 77.6580},
    {"Station_ID": 13, "Station_Name": "Botanical Gardens", "Line_ID": 3, "Zone": "Zone 1 - Cultural District", "Platform_Capacity": 3200, "Interchange_Flag": 0, "Surrounding_Land_Use": "Parklands & Heritage", "Latitude": 12.9500, "Longitude": 77.5850},
    {"Station_ID": 14, "Station_Name": "South End Boulevard", "Line_ID": 3, "Zone": "Zone 3 - Suburban District", "Platform_Capacity": 2800, "Interchange_Flag": 0, "Surrounding_Land_Use": "Residential High-Rise", "Latitude": 12.9150, "Longitude": 77.5800},
    {"Station_ID": 15, "Station_Name": "Industrial Zone West", "Line_ID": 3, "Zone": "Zone 3 - Industrial Corridor", "Platform_Capacity": 3100, "Interchange_Flag": 0, "Surrounding_Land_Use": "Manufacturing & Logistics", "Latitude": 12.9620, "Longitude": 77.5250},
    {"Station_ID": 16, "Station_Name": "West Gateway", "Line_ID": 3, "Zone": "Zone 2 - Western Suburbs", "Platform_Capacity": 3900, "Interchange_Flag": 1, "Surrounding_Land_Use": "Suburban Transit Hub", "Latitude": 12.9800, "Longitude": 77.5350},
    {"Station_ID": 17, "Station_Name": "Arts & Exhibition Center", "Line_ID": 3, "Zone": "Zone 1 - Cultural District", "Platform_Capacity": 4500, "Interchange_Flag": 0, "Surrounding_Land_Use": "Conventions & Exhibitions", "Latitude": 12.9730, "Longitude": 77.5780},
    {"Station_ID": 18, "Station_Name": "Airport Link Station", "Line_ID": 4, "Zone": "Zone 3 - Airport Corridor", "Platform_Capacity": 4800, "Interchange_Flag": 1, "Surrounding_Land_Use": "Airport Rail Transfer", "Latitude": 13.0500, "Longitude": 77.6100},
    {"Station_ID": 19, "Station_Name": "Aero City Complex", "Line_ID": 4, "Zone": "Zone 3 - Airport Corridor", "Platform_Capacity": 3500, "Interchange_Flag": 0, "Surrounding_Land_Use": "Aviation Logistics & Hotels", "Latitude": 13.0850, "Longitude": 77.6400},
    {"Station_ID": 20, "Station_Name": "International Airport Terminal", "Line_ID": 4, "Zone": "Zone 3 - Airport Terminal", "Platform_Capacity": 5500, "Interchange_Flag": 0, "Surrounding_Land_Use": "International Airport Terminals", "Latitude": 13.1986, "Longitude": 77.7066},
    {"Station_ID": 21, "Station_Name": "Cyber Gateway", "Line_ID": 4, "Zone": "Zone 2 - Tech Corridor", "Platform_Capacity": 4700, "Interchange_Flag": 0, "Surrounding_Land_Use": "Software Parks & Tech Hubs", "Latitude": 12.9950, "Longitude": 77.6750},
    {"Station_ID": 22, "Station_Name": "Silicon Enclave", "Line_ID": 4, "Zone": "Zone 2 - Tech Corridor", "Platform_Capacity": 4400, "Interchange_Flag": 0, "Surrounding_Land_Use": "IT Enterprise Campuses", "Latitude": 13.0100, "Longitude": 77.6900},
    {"Station_ID": 23, "Station_Name": "Heritage Palace", "Line_ID": 5, "Zone": "Zone 1 - Old City", "Platform_Capacity": 3300, "Interchange_Flag": 0, "Surrounding_Land_Use": "Tourism & Heritage Bazaars", "Latitude": 12.9800, "Longitude": 77.5650},
    {"Station_ID": 24, "Station_Name": "Textile Quarter", "Line_ID": 5, "Zone": "Zone 1 - Old City", "Platform_Capacity": 3100, "Interchange_Flag": 0, "Surrounding_Land_Use": "Wholesale Trade & Markets", "Latitude": 12.9700, "Longitude": 77.5550},
    {"Station_ID": 25, "Station_Name": "Hillview Heights", "Line_ID": 5, "Zone": "Zone 3 - Northern Hills", "Platform_Capacity": 2600, "Interchange_Flag": 0, "Surrounding_Land_Use": "Quiet Residential & Parks", "Latitude": 13.0300, "Longitude": 77.5450},
    {"Station_ID": 26, "Station_Name": "Greenwood Woods", "Line_ID": 5, "Zone": "Zone 3 - Northern Suburbs", "Platform_Capacity": 2500, "Interchange_Flag": 0, "Surrounding_Land_Use": "Suburban Residential", "Latitude": 13.0600, "Longitude": 77.5500},
    {"Station_ID": 27, "Station_Name": "Railway Station West Link", "Line_ID": 5, "Zone": "Zone 1 - Core Downtown", "Platform_Capacity": 4600, "Interchange_Flag": 1, "Surrounding_Land_Use": "Interstate Rail Intermodal Hub", "Latitude": 12.9780, "Longitude": 77.5700},
    {"Station_ID": 28, "Station_Name": "Outer Ring Road East", "Line_ID": 4, "Zone": "Zone 2 - Arterial Highway", "Platform_Capacity": 4100, "Interchange_Flag": 0, "Surrounding_Land_Use": "High-Density Commercial", "Latitude": 12.9450, "Longitude": 77.6950},
    {"Station_ID": 29, "Station_Name": "Valley Stream Colony", "Line_ID": 3, "Zone": "Zone 3 - Southern Valley", "Platform_Capacity": 2700, "Interchange_Flag": 0, "Surrounding_Land_Use": "Low-Density Residential", "Latitude": 12.8900, "Longitude": 77.5900},
    {"Station_ID": 30, "Station_Name": "Aerospace SEZ Park", "Line_ID": 4, "Zone": "Zone 3 - Industrial SEZ", "Platform_Capacity": 3600, "Interchange_Flag": 0, "Surrounding_Land_Use": "Aviation Engineering SEZ", "Latitude": 13.1500, "Longitude": 77.6700}
]
stations_df = pd.DataFrame(stations_data)
stations_df.to_csv(EXTERNAL_DIR / "metro_stations_metadata.csv", index=False)
print("Saved data/external/metro_stations_metadata.csv")

# -----------------------------------------------------------------------------
# 2. Transit Lines Metadata (Lines 1 to 5)
# -----------------------------------------------------------------------------
lines_data = [
    {
        "Line_ID": 1,
        "Line_Name": "Blue Line - Central Corridor",
        "Color_Code": "#1f77b4",
        "Total_Length_km": 28.5,
        "Operating_Stations": 6,
        "Peak_Headway_min": 4.0,
        "OffPeak_Headway_min": 10.0,
        "Max_Commercial_Speed_kmph": 75.0,
        "Train_Capacity_Passengers": 1500,
        "Route_Description": "Connects Core Central Business District with South Tech Corridor"
    },
    {
        "Line_ID": 2,
        "Line_Name": "Red Line - Stadium & Academic Link",
        "Color_Code": "#d62728",
        "Total_Length_km": 32.0,
        "Operating_Stations": 6,
        "Peak_Headway_min": 4.5,
        "OffPeak_Headway_min": 12.0,
        "Max_Commercial_Speed_kmph": 80.0,
        "Train_Capacity_Passengers": 1600,
        "Route_Description": "Transverses Academic District, National Stadium, and Coastal Harbor"
    },
    {
        "Line_ID": 3,
        "Line_Name": "Green Line - West-South Suburban Loop",
        "Color_Code": "#2ca02c",
        "Total_Length_km": 26.8,
        "Operating_Stations": 6,
        "Peak_Headway_min": 5.0,
        "OffPeak_Headway_min": 12.5,
        "Max_Commercial_Speed_kmph": 70.0,
        "Train_Capacity_Passengers": 1400,
        "Route_Description": "Connects West Industrial Logistics Corridor with South Valley Suburbs"
    },
    {
        "Line_ID": 4,
        "Line_Name": "Yellow Line - Airport Express & Cyber Corridor",
        "Color_Code": "#ff7f0e",
        "Total_Length_km": 42.4,
        "Operating_Stations": 7,
        "Peak_Headway_min": 5.0,
        "OffPeak_Headway_min": 15.0,
        "Max_Commercial_Speed_kmph": 95.0,
        "Train_Capacity_Passengers": 1800,
        "Route_Description": "High-speed line connecting International Airport with Cyber Tech Parks"
    },
    {
        "Line_ID": 5,
        "Line_Name": "Purple Line - Old Heritage & Northern Ring",
        "Color_Code": "#9467bd",
        "Total_Length_km": 24.2,
        "Operating_Stations": 5,
        "Peak_Headway_min": 6.0,
        "OffPeak_Headway_min": 14.0,
        "Max_Commercial_Speed_kmph": 65.0,
        "Train_Capacity_Passengers": 1300,
        "Route_Description": "Heritage circuit linking Old Town Bazaars, Rail Terminals, and Hillview"
    }
]
lines_df = pd.DataFrame(lines_data)
lines_df.to_csv(EXTERNAL_DIR / "transit_lines_metadata.csv", index=False)
print("Saved data/external/transit_lines_metadata.csv")

# -----------------------------------------------------------------------------
# 3. Holiday and Special Events Calendar Reference
# -----------------------------------------------------------------------------
calendar_data = [
    {"Date": "2025-01-01", "Day_of_Week": 2, "Holiday_Flag": 1, "Special_Event_Flag": 0, "Event_Name": "New Year's Day", "Impacted_Stations": "All Core Stations", "Expected_Impact": "High mid-day leisure traffic"},
    {"Date": "2025-01-15", "Day_of_Week": 2, "Holiday_Flag": 1, "Special_Event_Flag": 0, "Event_Name": "Harvest Festival / Pongal", "Impacted_Stations": "Intermodal Hubs & Bazaars", "Expected_Impact": "Suburban outbound surge"},
    {"Date": "2025-01-26", "Day_of_Week": 6, "Holiday_Flag": 1, "Special_Event_Flag": 1, "Event_Name": "Republic Day & Central Parade", "Impacted_Stations": "Stations 1, 3, 5, 8", "Expected_Impact": "Severe crowd surges near Civic centers"},
    {"Date": "2025-02-14", "Day_of_Week": 4, "Holiday_Flag": 0, "Special_Event_Flag": 1, "Event_Name": "International Stadium Cricket Derby", "Impacted_Stations": "Station 8 (National Stadium)", "Expected_Impact": "Massive post-match egress surge at 22:00"},
    {"Date": "2025-03-14", "Day_of_Week": 4, "Holiday_Flag": 1, "Special_Event_Flag": 0, "Event_Name": "Spring Festival / Holi", "Impacted_Stations": "All Residential & Central Stations", "Expected_Impact": "Morning shutdown; afternoon leisure surge"},
    {"Date": "2025-04-14", "Day_of_Week": 0, "Holiday_Flag": 1, "Special_Event_Flag": 0, "Event_Name": "Regional New Year", "Impacted_Stations": "Heritage & Bazaar Stations", "Expected_Impact": "Moderate daytime demand"},
    {"Date": "2025-05-01", "Day_of_Week": 3, "Holiday_Flag": 1, "Special_Event_Flag": 0, "Event_Name": "International Workers' Day", "Impacted_Stations": "Industrial & Tech Stations", "Expected_Impact": "Low morning commuter volume"},
    {"Date": "2025-06-18", "Day_of_Week": 2, "Holiday_Flag": 0, "Special_Event_Flag": 1, "Event_Name": "Global Tech Summit 2025", "Impacted_Stations": "Stations 6, 7, 21, 22", "Expected_Impact": "Heavy morning and evening tech commute"},
    {"Date": "2025-08-15", "Day_of_Week": 4, "Holiday_Flag": 1, "Special_Event_Flag": 1, "Event_Name": "Independence Day Celebration", "Impacted_Stations": "Stations 1, 3, 8", "Expected_Impact": "Large morning civic gatherings"},
    {"Date": "2025-09-05", "Day_of_Week": 4, "Holiday_Flag": 0, "Special_Event_Flag": 1, "Event_Name": "International Stadium Rock Concert", "Impacted_Stations": "Station 8 (National Stadium)", "Expected_Impact": "Night egress crowd peaking at 23:00"},
    {"Date": "2025-10-02", "Day_of_Week": 3, "Holiday_Flag": 1, "Special_Event_Flag": 0, "Event_Name": "National Day of Non-Violence", "Impacted_Stations": "Central Heritage Stations", "Expected_Impact": "Off-peak leisure traffic"},
    {"Date": "2025-10-20", "Day_of_Week": 0, "Holiday_Flag": 1, "Special_Event_Flag": 1, "Event_Name": "Festival of Lights / Diwali", "Impacted_Stations": "Market Square, Terminal Hubs", "Expected_Impact": "Massive retail shopping surge preceding evening"},
    {"Date": "2025-12-25", "Day_of_Week": 3, "Holiday_Flag": 1, "Special_Event_Flag": 0, "Event_Name": "Christmas Day", "Impacted_Stations": "Central & Harbor Stations", "Expected_Impact": "Evening leisure & shopping congestion"}
]
calendar_df = pd.DataFrame(calendar_data)
calendar_df.to_csv(EXTERNAL_DIR / "holiday_and_events_calendar.csv", index=False)
print("Saved data/external/holiday_and_events_calendar.csv")

# -----------------------------------------------------------------------------
# 4. Meteorological Classification Standards
# -----------------------------------------------------------------------------
weather_standards = [
    {"Variable": "Rainfall_mm", "Range_Low": 0.0, "Range_High": 0.1, "Classification": "None / Dry", "Operational_Impact": "Normal adherence to dispatch schedule; regular dwell times"},
    {"Variable": "Rainfall_mm", "Range_Low": 0.1, "Range_High": 2.5, "Classification": "Light Rain", "Operational_Impact": "Minimal effect; slight increase in concourse boarding time"},
    {"Variable": "Rainfall_mm", "Range_Low": 2.5, "Range_High": 7.5, "Classification": "Moderate Rain", "Operational_Impact": "Moderate speed restrictions on open viaduct sections (+10s dwell)"},
    {"Variable": "Rainfall_mm", "Range_Low": 7.5, "Range_High": 50.0, "Classification": "Heavy Rain / Monsoon", "Operational_Impact": "Track drainage protocols activated; signal delay warnings (+30-60s headway)"},
    {"Variable": "Temperature_C", "Range_Low": 15.0, "Range_High": 22.0, "Classification": "Cool / Mild", "Operational_Impact": "Low HVAC cooling energy demand on rolling stock"},
    {"Variable": "Temperature_C", "Range_Low": 22.0, "Range_High": 30.0, "Classification": "Warm / Moderate", "Operational_Impact": "Standard carriage ventilation and air conditioning"},
    {"Variable": "Temperature_C", "Range_Low": 30.0, "Range_High": 45.0, "Classification": "Hot / Heatwave", "Operational_Impact": "Elevated auxiliary energy draw (HVAC peak); platform cooling active"},
    {"Variable": "Visibility_km", "Range_Low": 0.0, "Range_High": 2.0, "Classification": "Low Visibility / Fog", "Operational_Impact": "Cautionary speed limits enforced on overground rail tracks"},
    {"Variable": "Visibility_km", "Range_Low": 2.0, "Range_High": 6.0, "Classification": "Moderate Visibility", "Operational_Impact": "Automated signaling operates with standard headway margins"},
    {"Variable": "Visibility_km", "Range_Low": 6.0, "Range_High": 20.0, "Classification": "Clear Visibility", "Operational_Impact": "Optimal visual and automated operation at full commercial speeds"}
]
weather_df = pd.DataFrame(weather_standards)
weather_df.to_csv(EXTERNAL_DIR / "weather_classification_standards.csv", index=False)
print("Saved data/external/weather_classification_standards.csv")

# -----------------------------------------------------------------------------
# 5. Comprehensive Documentation in data/external/README.md
# -----------------------------------------------------------------------------
readme_content = """# External Domain Datasets and Metadata References

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
"""

with open(EXTERNAL_DIR / "README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
print("Saved data/external/README.md")

print("\nExternal folder successfully populated with 4 reference datasets and documentation!")
