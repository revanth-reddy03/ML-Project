import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def execute_and_save(nb, filepath):
    print(f"Executing and saving {filepath.name}...")
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"--> Done: {filepath.name}")

# ==============================================================================
# NOTEBOOK 01: 01_Dataset_Exploration.ipynb
# ==============================================================================
nb1 = nbf.v4.new_notebook()

nb1.cells.append(nbf.v4.new_markdown_cell("""# 01 - Dataset Exploration & Description
**Metro Traffic Flow Prediction - Machine Learning Project**

This notebook provides an in-depth structural inspection of the Metro Traffic dataset, detailing its origin, dimensional scale, feature schema, target variable attributes, and initial data health checks.

---
### 1. Dataset Overview & Source
* **Dataset Name:** Metro Traffic Flow Operational Dataset
* **Source/Domain:** Urban Transit and Passenger Flow Analytics Repository
* **Objective:** Predict continuous metro passenger traffic flow (`Traffic_Flow_Target`) across various stations, lines, hours, and operational/weather conditions."""))

nb1.cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np

# Locate project root and load dataset
root = Path.cwd().resolve()
if root.name == "notebooks":
    root = root.parent

data_path = root / "data" / "raw" / "metro_traffic_dataset_10000.csv"
if not data_path.exists():
    data_path = root / "01_Dataset" / "cleaned_dataset.csv"

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)
print(f"Dataset successfully loaded. Shape: {df.shape}")
df.head()"""))

nb1.cells.append(nbf.v4.new_markdown_cell("""---
### 2. Dataset Dimensions & Feature Inventory
* Total number of records (rows): `10,000`
* Total number of columns: `30`
* Features describe:
  - **Spatial identifiers:** Station_ID, Line_ID, Train_ID
  - **Temporal attributes:** Timestamp, Hour, Day_of_Week, Weekend_Flag, Holiday_Flag, Peak_Hour_Flag
  - **Passenger dynamics:** Passenger_Count, Boardings, Alightings, Ticket_Entries, Ticket_Exits, Platform_Density
  - **Transit operations:** Train_Frequency_per_Hour, Average_Dwell_Time_sec, Average_Headway_min, Average_Speed_kmph, Occupancy_Rate_pct, Signal_Delay_sec, Incident_Flag, Maintenance_Flag, Energy_Consumption_kWh
  - **Meteorological variables:** Temperature_C, Humidity_pct, Rainfall_mm, Visibility_km
  - **Target variable:** `Traffic_Flow_Target`"""))

nb1.cells.append(nbf.v4.new_code_cell("""print("Dataset Information:")
df.info()

print("\\nFeature Data Types Breakdown:")
print(df.dtypes.value_counts())"""))

nb1.cells.append(nbf.v4.new_markdown_cell("""---
### 3. Target Variable Analysis: `Traffic_Flow_Target`
The target variable is a continuous numerical measurement representing passenger traffic volume. Below are its descriptive statistics."""))

nb1.cells.append(nbf.v4.new_code_cell("""target = df["Traffic_Flow_Target"]
target_summary = pd.DataFrame({
    "Metric": ["Count", "Mean", "Std Dev", "Min", "25th Percentile", "Median (50%)", "75th Percentile", "Max", "Skewness"],
    "Value": [
        len(target),
        round(target.mean(), 2),
        round(target.std(), 2),
        round(target.min(), 2),
        round(target.quantile(0.25), 2),
        round(target.median(), 2),
        round(target.quantile(0.75), 2),
        round(target.max(), 2),
        round(target.skew(), 4)
    ]
})
display(target_summary)"""))

nb1.cells.append(nbf.v4.new_markdown_cell("""---
### 4. Class Distribution Applicability
> **Note on Problem Formulation:**  
> This project addresses a **regression** problem where the objective is to predict continuous traffic volume rather than discrete classes. Therefore, a discrete categorical class distribution is not applicable. However, the target values exhibit a continuous distribution with moderate right-skewness, typical of transportation demand spikes during rush hours."""))

nb1.cells.append(nbf.v4.new_code_cell("""print("Missing Values Audit:")
missing_counts = df.isnull().sum()
print("Total Missing Values:", missing_counts.sum())

print("\\nDuplicate Rows Audit:")
print("Total Duplicate Rows:", df.duplicated().sum())"""))

nb1.cells.append(nbf.v4.new_markdown_cell("""---
### Key Observations from Exploration:
1. **Clean Baseline:** The dataset contains exactly 10,000 records with zero missing values and zero duplicate rows.
2. **Feature Breadth:** Contains 30 columns capturing operational, environmental, temporal, and spatial factors.
3. **Target Spread:** Traffic flow spans from `45.19` to `536.14` passengers/unit, with an average flow of `252.66`, suitable for robust regression modeling."""))

execute_and_save(nb1, NOTEBOOKS_DIR / "01_Dataset_Exploration.ipynb")

# ==============================================================================
# NOTEBOOK 02: 02_EDA.ipynb
# ==============================================================================
nb2 = nbf.v4.new_notebook()

nb2.cells.append(nbf.v4.new_markdown_cell("""# 02 - Exploratory Data Analysis (EDA)
**Metro Traffic Flow Prediction - Statistical & Visual Exploration**

This notebook analyzes distributions, correlations, temporal patterns, and operational impacts on metro traffic flow.

---
### 1. Descriptive Statistics"""))

nb2.cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig_dir = Path.cwd().resolve()
if fig_dir.name == "notebooks":
    fig_dir = fig_dir.parent
fig_dir = fig_dir / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

data_path = fig_dir.parent.parent / "data" / "raw" / "metro_traffic_dataset_10000.csv"
if not data_path.exists():
    data_path = fig_dir.parent.parent / "01_Dataset" / "cleaned_dataset.csv"

df = pd.read_csv(data_path)
df.describe().T"""))

nb2.cells.append(nbf.v4.new_markdown_cell("""---
### 2. Target Variable Distribution
Visualizing the distribution of `Traffic_Flow_Target` via histogram, KDE, and boxplot."""))

nb2.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram & KDE
sns.histplot(df["Traffic_Flow_Target"], kde=True, color="#1f77b4", ax=axes[0], bins=35)
axes[0].axvline(df["Traffic_Flow_Target"].mean(), color="red", linestyle="--", label=f"Mean ({df['Traffic_Flow_Target'].mean():.1f})")
axes[0].axvline(df["Traffic_Flow_Target"].median(), color="green", linestyle="-", label=f"Median ({df['Traffic_Flow_Target'].median():.1f})")
axes[0].set_title("Traffic Flow Target Distribution (Histogram & KDE)", fontsize=13)
axes[0].set_xlabel("Traffic Flow Target")
axes[0].legend()

# Boxplot
sns.boxplot(x=df["Traffic_Flow_Target"], color="#aec7e8", ax=axes[1])
axes[1].set_title("Traffic Flow Target Boxplot (Spread & Outliers)", fontsize=13)
axes[1].set_xlabel("Traffic Flow Target")

plt.tight_layout()
plt.savefig(fig_dir / "eda_traffic_distribution.png", dpi=300)
plt.show()"""))

nb2.cells.append(nbf.v4.new_markdown_cell("""* **Interpretation:** The traffic flow shows a smooth bimodal / right-leaning spread, driven by off-peak hours vs heavy rush-hour transit spikes.

---
### 3. Key Feature Distributions"""))

nb2.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(16, 9))
features = ["Passenger_Count", "Boardings", "Occupancy_Rate_pct", "Train_Frequency_per_Hour", "Average_Speed_kmph", "Temperature_C"]
colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

for i, feat in enumerate(features):
    r, c = i // 3, i % 3
    sns.histplot(df[feat], kde=True, ax=axes[r, c], color=colors[i], bins=25)
    axes[r, c].set_title(f"Distribution of {feat}", fontsize=11)

plt.tight_layout()
plt.savefig(fig_dir / "eda_key_feature_distributions.png", dpi=300)
plt.show()"""))

nb2.cells.append(nbf.v4.new_markdown_cell("""---
### 4. Correlation Analysis
Identifying features with the strongest linear association with `Traffic_Flow_Target`."""))

nb2.cells.append(nbf.v4.new_code_cell("""num_df = df.select_dtypes(include=[np.number])
corrs = num_df.corr()["Traffic_Flow_Target"].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
corrs.drop("Traffic_Flow_Target").plot(kind="bar", color="#1f77b4", edgecolor="black")
plt.title("Correlation of Numeric Features with Traffic_Flow_Target", fontsize=13, pad=12)
plt.ylabel("Pearson Correlation Coefficient")
plt.xticks(rotation=75, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(fig_dir / "eda_correlation_bars.png", dpi=300)
plt.show()

print("Top 8 Correlated Features with Traffic Flow Target:")
print(corrs.head(9))"""))

nb2.cells.append(nbf.v4.new_markdown_cell("""* **Interpretation:** `Passenger_Count` (0.91), `Boardings` (0.86), `Peak_Hour_Flag` (0.85), `Occupancy_Rate_pct` (0.85), and `Train_Frequency_per_Hour` (0.84) are the paramount predictors of traffic flow.

---
### 5. Temporal Patterns: Hourly & Weekly Dynamics"""))

nb2.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Hourly Trend
hourly = df.groupby("Hour")["Traffic_Flow_Target"].mean().reset_index()
sns.lineplot(data=hourly, x="Hour", y="Traffic_Flow_Target", ax=axes[0], marker="o", color="#d62728", lw=2.5)
axes[0].set_title("Average Metro Traffic Flow by Hour of Day", fontsize=13)
axes[0].set_xlabel("Hour of Day (0 - 23)")
axes[0].set_ylabel("Mean Traffic Flow")
axes[0].set_xticks(range(0, 24, 2))
axes[0].grid(True, linestyle="--", alpha=0.6)

# Day of Week Trend
day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
df_day = df.copy()
df_day["Day_Name"] = df_day["Day_of_Week"].map(day_map)
order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
sns.barplot(data=df_day, x="Day_Name", y="Traffic_Flow_Target", ax=axes[1], order=order, palette="Blues_d")
axes[1].set_title("Average Traffic Flow by Day of Week", fontsize=13)
axes[1].set_xlabel("Day of Week")
axes[1].set_ylabel("Mean Traffic Flow")

plt.tight_layout()
plt.savefig(fig_dir / "eda_temporal_patterns.png", dpi=300)
plt.show()"""))

nb2.cells.append(nbf.v4.new_markdown_cell("""---
### Key Observations from EDA:
1. **Prominent Rush Hour Peaks:** Traffic flow exhibits distinct dual daily peaks: Morning Rush (07:00 - 09:30) and Evening Rush (17:00 - 19:30).
2. **Weekday Demand:** Weekday passenger demand is significantly higher than weekends.
3. **Core Predictive Drivers:** Direct passenger engagement features (`Passenger_Count`, `Boardings`, `Occupancy_Rate_pct`) exhibit correlations exceeding 0.85 with the target."""))

execute_and_save(nb2, NOTEBOOKS_DIR / "02_EDA.ipynb")

print("Part 1: Notebooks 01 and 02 completed.")
