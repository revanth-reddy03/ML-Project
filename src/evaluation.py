"""
Model evaluation, residual analysis, and comparison visualization module.
"""
from typing import Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "MAE": round(float(mae), 3),
        "MSE": round(float(mse), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 4)
    }

def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Best Model",
    save_path: Path = None
):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.35, color="#1f77b4", edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal 45-Degree Line")
    plt.title(f"Actual vs Predicted Traffic Flow - {model_name}", fontsize=14, pad=12)
    plt.xlabel("Actual Traffic Flow", fontsize=12)
    plt.ylabel("Predicted Traffic Flow", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Best Model",
    save_path: Path = None
):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.35, color="#2ca02c")
    axes[0].axhline(0, color="red", linestyle="--", lw=2)
    axes[0].set_title(f"Residuals vs Predicted - {model_name}", fontsize=12)
    axes[0].set_xlabel("Predicted Values", fontsize=11)
    axes[0].set_ylabel("Residuals (Actual - Pred)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color="#ff7f0e", bins=30)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title(f"Residual Distribution - {model_name}", fontsize=12)
    axes[1].set_xlabel("Residual Value", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
