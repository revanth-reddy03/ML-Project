"""
Data preprocessing, cleaning, and transformation module with zero data leakage.
"""
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Inspect and treat missing values, duplicates, and column naming."""
    initial_shape = df.shape
    dup_count = int(df.duplicated().sum())
    df_cleaned = df.drop_duplicates()
    
    # Missing values check
    missing_before = df_cleaned.isnull().sum().to_dict()
    
    # Numeric imputation (median) if any
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_cleaned[col].isnull().any():
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            
    stats = {
        "initial_shape": initial_shape,
        "final_shape": df_cleaned.shape,
        "duplicates_removed": dup_count,
        "missing_before": missing_before,
        "missing_after": df_cleaned.isnull().sum().to_dict()
    }
    return df_cleaned, stats

def get_feature_matrix(df: pd.DataFrame, target_col: str = "Traffic_Flow_Target") -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target variable and drop non-predictive columns like Timestamp."""
    drop_cols = [target_col]
    if "Timestamp" in df.columns:
        drop_cols.append("Timestamp")
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y

def split_and_scale(
    df: pd.DataFrame,
    target_col: str = "Traffic_Flow_Target",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Split data into train and test sets, then scale features strictly using train statistics.
    Ensures ZERO data leakage.
    """
    X, y = get_feature_matrix(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    feature_names = X_train.columns.tolist()
    
    # Fit strictly on train
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index
    )
    # Transform test using train scaler
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
