# Gold Price Predictor: Project Pipeline & Documentation

This document provides a detailed explanation of the end-to-end pipeline for the Gold Price Predictor project, specifically designed for the Bangladesh market.

---

## 1. Project Overview
The goal of this project is to predict gold prices in Bangladesh by integrating multiple economic indicators:
- **Historical Gold Prices**: Daily price of traditional and 22K gold.
- **USD to BDT Exchange Rate**: Captures the impact of currency fluctuations.
- **Inflation Rates**: Reflects the decreasing purchasing power of currency.
- **Local Seasonality**: Accounts for "Wedding Season" demand shocks.

---

## 2. Data Preprocessing (`Pre-Process/`)
Raw data is processed through specialized scripts to ensure consistency and stationarity.
- **`preProcess_gold.py`**: Handles date standardization, **Cubic Spline Interpolation** for gaps, and calculates **Log Returns**.
- **`preProcess_rate.py`**: Cleans USD-BDT exchange rates and calculates daily returns.
- **`mergeDataset.py`**: Integrates all data, engineers features (Wedding Season, Lags, Rolling Stats), and saves the **`feature_scaler.pkl`**.

---

## 3. Training Pipeline (`training/`)
The project is designed to be **dynamic and modular**. Each model has its own dedicated training script.

### `lstm_trainer.py`
- **Architecture**: A 2-layer LSTM with Batch Normalization and Dropout.
- **Output**: Saves `lstm_model.pth` to the `Models/` directory.

### `xgboost_trainer.py`
- **Architecture**: A gradient-boosted tree model optimized for the flattened time-series data.
- **Output**: Saves `xgboost_model.joblib` to the `Models/` directory.

> **Adding New Models**: To add a new model, simply create a new script in the `training/` folder following the established pattern.

---

## 4. Dynamic Evaluation (`evaluation/`)
The evaluation phase is centralized to allow for direct model comparison.

### `evaluator.py`
- **Functionality**: Automatically detects trained models in the `Models/` folder.
- **Metrics**: Calculates RMSE, MAE, R2, and **Directional Accuracy**.
- **Outputs**: 
  - A comparative table printed to the console.
  - `comparison_plot.png`: A visual chart comparing model predictions against actual prices.

---

## 5. Inference (`predict_future.py`)
- **Action**: Uses the best-performing model (XGBoost by default) to predict the gold price for the next trading day.
- **Interpretability**: Provides a SHAP-based breakdown of which features influenced the specific prediction.
