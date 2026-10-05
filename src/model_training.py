"""
Model training module for Baseline and Advanced ML Models.
"""
from typing import Dict, Any, Tuple
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score

def get_baseline_models() -> Dict[str, Any]:
    """Return dictionary of baseline regression models."""
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(alpha=1.0, random_state=42)
    }

def get_advanced_models() -> Dict[str, Any]:
    """Return dictionary of advanced tree and ensemble models."""
    rf = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    dt = DecisionTreeRegressor(max_depth=12, random_state=42)
    
    # Hybrid Stacking Ensemble combining RF and XGBoost with Ridge meta-estimator
    hybrid = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1))
        ],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1
    )
    
    return {
        "DecisionTree": dt,
        "RandomForest": rf,
        "XGBoost": xgb,
        "HybridEnsemble": hybrid
    }

def evaluate_cv(model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
    """Perform K-Fold Cross-Validation and return mean and std metrics."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    neg_mae = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=kf, n_jobs=-1)
    neg_rmse = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
    r2_scores = cross_val_score(model, X, y, scoring="r2", cv=kf, n_jobs=-1)
    
    return {
        "cv_mae_mean": float(-neg_mae.mean()),
        "cv_mae_std": float(neg_mae.std()),
        "cv_rmse_mean": float(-neg_rmse.mean()),
        "cv_rmse_std": float(neg_rmse.std()),
        "cv_r2_mean": float(r2_scores.mean()),
        "cv_r2_std": float(r2_scores.std())
    }

def save_model(model: Any, model_name: str, models_dir: Path = None):
    """Save trained model to disk."""
    if models_dir is None:
        from .data_loader import get_project_root
        models_dir = get_project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    filepath = models_dir / f"{model_name}.pkl"
    joblib.dump(model, filepath)
    return filepath
