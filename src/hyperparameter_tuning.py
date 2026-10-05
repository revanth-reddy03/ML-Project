"""
Hyperparameter tuning module using GridSearchCV and RandomizedSearchCV.
"""
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    random_state: int = 42
) -> Tuple[RandomForestRegressor, Dict[str, Any], pd.DataFrame]:
    """Run RandomizedSearchCV on RandomForestRegressor."""
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    results_df = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, search.best_params_, results_df

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    random_state: int = 42
) -> Tuple[XGBRegressor, Dict[str, Any], pd.DataFrame]:
    """Run RandomizedSearchCV on XGBRegressor."""
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }
    xgb = XGBRegressor(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    results_df = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, search.best_params_, results_df
