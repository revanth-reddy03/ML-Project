"""
Inference and demonstration module for Metro Traffic predictions.
"""
from typing import Dict, Any, Union
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

def load_inference_artifacts(models_dir: Path = None):
    """Load the trained best model and scaler."""
    if models_dir is None:
        from .data_loader import get_project_root
        models_dir = get_project_root() / "models"
        
    model_path = models_dir / "hybrid_model.pkl"
    if not model_path.exists():
        model_path = models_dir / "xgboost.pkl"
    if not model_path.exists():
        model_path = models_dir / "random_forest.pkl"
        
    model = joblib.load(model_path)
    
    scaler_path = models_dir / "preprocessor.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    
    return model, scaler

def predict_traffic_flow(
    input_data: Union[Dict[str, Any], pd.DataFrame],
    model: Any = None,
    scaler: Any = None,
    models_dir: Path = None
) -> np.ndarray:
    """
    Predict traffic flow given raw or scaled feature values.
    """
    if model is None:
        model, scaler = load_inference_artifacts(models_dir)
        
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()
        
    # Drop target or timestamp if present
    drop_cols = [c for c in ["Traffic_Flow_Target", "Timestamp"] if c in df_input.columns]
    if drop_cols:
        df_input = df_input.drop(columns=drop_cols)
        
    if scaler is not None:
        feature_cols = [c for c in df_input.columns if c in getattr(scaler, "feature_names_in_", df_input.columns)]
        df_scaled = pd.DataFrame(scaler.transform(df_input[feature_cols]), columns=feature_cols)
        predictions = model.predict(df_scaled)
    else:
        predictions = model.predict(df_input)
        
    return predictions
