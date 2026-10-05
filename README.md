# Metro Traffic Flow Prediction - Machine Learning Project

An end-to-end machine learning project designed to forecast metro traffic volume (`Traffic_Flow_Target`) across diverse transit stations, lines, hours, and meteorological conditions.

Submitted for the **Machine Learning** course (Bachelor of Technology, A.Y. 2026-27), Department of Computer Science and Engineering, **Koneru Lakshmaiah Education Foundation (KL University)**.

---

## 📁 Project Architecture

The project is structured according to the modular standard layout:

```
ML-Project/
├── data/
│   ├── external/                                  # Station metadata, transit lines, calendars, weather thresholds
│   │   ├── metro_stations_metadata.csv
│   │   ├── transit_lines_metadata.csv
│   │   ├── holiday_and_events_calendar.csv
│   │   ├── weather_classification_standards.csv
│   │   └── README.md
│   ├── processed/                                 # Cleaned, split (train/test), and transformed data
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── processed_metro_traffic.csv
│   └── raw/
│       └── metro_traffic_dataset_10000.csv        # Primary raw dataset (10,000 records, 30 columns)
├── models/                                        # Trained models & serialization pipelines (.pkl)
│   ├── preprocessor.pkl                           # Fitted StandardScaler (Zero data leakage)
│   ├── linearregression.pkl
│   ├── ridgeregression.pkl
│   ├── decisiontree.pkl
│   ├── randomforest.pkl
│   ├── xgboost.pkl
│   └── hybridensemble.pkl                         # Champion Stacking Ensemble Regressor
├── notebooks/                                     # 11 Modular + 1 Master fully executed notebooks
│   ├── 01_Dataset_Exploration.ipynb               # 1. Dataset Description & inspection
│   ├── 02_EDA.ipynb                               # 3. Exploratory Data Analysis & visual insights
│   ├── 03_Preprocessing.ipynb                     # 2. Preprocessing & zero data leakage verification
│   ├── 04_Feature_Engineering_Selection.ipynb    # 4. Feature creation, selection & importance
│   ├── 05_Baseline_Model.ipynb                    # 5. Baseline models (Dummy, Linear, Ridge)
│   ├── 06_Imbalance_Handling.ipynb                # 5/6. Regression tail skewness & congestion binning
│   ├── 07_Advanced_Models.ipynb                   # 5. Non-linear models, Ensembles & Cross-Validation
│   ├── 08_Hyperparameter_Tuning.ipynb            # 6. Grid/Random search tuning & before/after metrics
│   ├── 09_Model_Evaluation.ipynb                  # 7. MAE, MSE, RMSE, R², residual diagnostics
│   ├── 10_Model_Comparison.ipynb                  # 8/9. Master comparison table, bar plots & discussion
│   ├── 11_Final_Prediction_Demonstration.ipynb    # 10. Scenario demonstrations & unseen data validation
│   └── Metro_Traffic_Complete_Project.ipynb       # All 10 circular points unified in one master notebook
├── outputs/
│   ├── figures/                                   # High-resolution saved diagnostic plots
│   ├── predictions/                               # Predictions on test set (CSV)
│   └── tables/                                    # Model comparison & evaluation tables (CSV)
├── reports/
│   ├── KLH_ML_Project_Report_Filled.docx          # Official filled 5-page report in Word (.docx)
│   ├── KLH_ML_Project_Report_Filled.pdf           # Official filled 5-page report in PDF (.pdf)
│   ├── KLH_ML_Project_Report_Filled.md            # Official filled 5-page report in Markdown (.md)
│   ├── Metro_Traffic_Project_Report.docx          # Full extended technical report in Word (.docx)
│   └── Metro_Traffic_Project_Report.md            # Full extended technical report in Markdown (.md)
├── src/                                           # Clean, production-ready Python source code
│   ├── __init__.py
│   ├── data_loader.py                             # Dataset loading & metadata inspection
│   ├── preprocessing.py                           # Zero-leakage cleaning & scaling pipeline
│   ├── feature_engineering.py                     # Domain feature creation & selection
│   ├── model_training.py                          # Training baselines, ensembles, and stacking
│   ├── hyperparameter_tuning.py                   # RandomizedSearchCV optimization
│   ├── evaluation.py                              # Regression metrics & residual diagnostics
│   └── predict.py                                 # Production inference pipeline
├── app/                                           # Interactive Streamlit dashboard
├── ML Abstract.pdf                                # Project abstract document
├── metro_traffic_dataset_10000.csv                # Root dataset copy
├── requirements.txt                               # Project dependencies
└── README.md                                      # Project overview and instructions
```

---

## 🏆 Model Performance Summary

Evaluated on 2,000 held-out test records:

| Model Name | MAE (passengers) | MSE | RMSE (passengers) | $R^2$ Score | Ranking |
|---|---:|---:|---:|---:|:---:|
| **Linear Regression** | 20.402 | 645.061 | 25.398 | 0.9157 | Strong Baseline |
| **Ridge Regression** | 20.402 | 645.066 | 25.398 | 0.9157 | Regularized Baseline |
| **Hybrid Ensemble (Stacking)** | **20.812** | **672.542** | **25.933** | **0.9122** | **Champion Model** |
| **XGBoost Regressor (Tuned)** | 20.815 | 674.925 | 25.979 | 0.9118 | Top Single Estimator |
| **Random Forest (Tuned)** | 21.491 | 717.251 | 26.782 | 0.9063 | Bagging Ensemble |
| **Decision Tree** | 27.700 | 1213.784 | 34.839 | 0.8415 | Unpruned Tree |
| **Dummy Regressor (Baseline)** | 71.463 | 7656.386 | 87.501 | -0.0000 | Statistical Floor |

---

## 🚀 How to Run the Project

```powershell
# 1. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application
streamlit run app/app.py
```
