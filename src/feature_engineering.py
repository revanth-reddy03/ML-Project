"""
Feature engineering and feature selection module.
"""
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific engineered features for metro traffic flow prediction."""
    df_feat = df.copy()
    
    # 1. Interaction between peak hour and passenger volume
    if "Peak_Hour_Flag" in df_feat.columns and "Passenger_Count" in df_feat.columns:
        df_feat["Rush_Hour_Multiplier"] = df_feat["Peak_Hour_Flag"] * df_feat["Passenger_Count"]
        
    # 2. Net passenger flow (boardings minus alightings)
    if "Boardings" in df_feat.columns and "Alightings" in df_feat.columns:
        df_feat["Net_Passenger_Flow"] = df_feat["Boardings"] - df_feat["Alightings"]
        df_feat["Total_Turnover"] = df_feat["Boardings"] + df_feat["Alightings"]
        
    # 3. Congestion pressure: occupancy divided by frequency
    if "Occupancy_Rate_pct" in df_feat.columns and "Train_Frequency_per_Hour" in df_feat.columns:
        df_feat["Congestion_Pressure"] = df_feat["Occupancy_Rate_pct"] / (df_feat["Train_Frequency_per_Hour"] + 1e-5)
        
    # 4. Weather severity index (rainfall compounded by low visibility)
    if "Rainfall_mm" in df_feat.columns and "Visibility_km" in df_feat.columns:
        df_feat["Weather_Severity_Index"] = df_feat["Rainfall_mm"] / (df_feat["Visibility_km"] + 0.1)
        
    return df_feat

def compute_feature_importances(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Calculate feature importances using Mutual Information and Random Forest."""
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    mi = mutual_info_regression(X, y, random_state=42)
    
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "RF_Importance": rf.feature_importances_,
        "Mutual_Info": mi,
        "Abs_Correlation": [abs(X[col].corr(y)) for col in X.columns]
    }).sort_values(by="RF_Importance", ascending=False).reset_index(drop=True)
    
    return importance_df
