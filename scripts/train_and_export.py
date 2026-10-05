import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
import numpy as np
import src.data_loader as dl
import src.preprocessing as pp
import src.feature_engineering as fe
import src.evaluation as ev
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("Loading raw dataset...")
df = dl.load_raw_data()

# Clean and split
df_clean, clean_stats = pp.clean_data(df)
X_train, X_test, y_train, y_test, scaler = pp.split_and_scale(df_clean)
joblib.dump(scaler, "models/preprocessor.pkl")

# Save processed datasets
df_clean.to_csv("data/processed/processed_metro_traffic.csv", index=False)
X_train.assign(Traffic_Flow_Target=y_train).to_csv("data/processed/train_data.csv", index=False)
X_test.assign(Traffic_Flow_Target=y_test).to_csv("data/processed/test_data.csv", index=False)

models = {
    "DummyRegressor": DummyRegressor(strategy="mean"),
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(alpha=1.0, random_state=42),
    "DecisionTree": DecisionTreeRegressor(max_depth=12, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1),
    "HybridEnsemble": StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1))
        ],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1
    )
}

metrics_list = []
preds_dict = {"Actual": y_test.values}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    preds_dict[name] = y_pred
    m = ev.compute_regression_metrics(y_test, y_pred, model_name=name)
    metrics_list.append(m)
    joblib.dump(model, f"models/{name.lower()}.pkl")
    print(f"  {name} -> MAE: {m['MAE']}, RMSE: {m['RMSE']}, R2: {m['R2']}")

metrics_df = pd.DataFrame(metrics_list).sort_values(by="RMSE").reset_index(drop=True)
metrics_df.to_csv("outputs/tables/model_comparison_table.csv", index=False)
pd.DataFrame(preds_dict).to_csv("outputs/predictions/test_predictions.csv", index=False)

print("\nAll models trained and exported successfully.")
print("\nFinal Model Comparison Table:")
print(metrics_df.to_string(index=False))
