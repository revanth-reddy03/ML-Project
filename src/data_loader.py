"""
Data loading and dataset description module for Metro Traffic ML Project.
"""
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    current = Path.cwd().resolve()
    # Check if we are inside 'notebooks' or 'src'
    if current.name in ["notebooks", "src"]:
        return current.parent
    return current

def load_raw_data(data_path: Path = None) -> pd.DataFrame:
    """Load the raw or cleaned Metro Traffic dataset."""
    if data_path is None:
        root = get_project_root()
        # Priority check
        candidates = [
            root / "data" / "raw" / "metro_traffic_dataset_10000.csv",
            root / "metro_traffic_dataset_10000.csv",
            root / "01_Dataset" / "cleaned_dataset.csv"
        ]
        for c in candidates:
            if c.exists():
                data_path = c
                break
        if data_path is None:
            raise FileNotFoundError("Could not locate metro_traffic_dataset_10000.csv")

    df = pd.read_csv(data_path)
    return df

def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract structural metadata and summary information from the dataset."""
    target_col = "Traffic_Flow_Target"
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    
    return {
        "num_records": len(df),
        "num_features": df.shape[1],
        "feature_names": df.columns.tolist(),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "target_variable": target_col,
        "target_present": target_col in df.columns,
        "target_stats": df[target_col].describe().to_dict() if target_col in df.columns else None,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum())
    }
