# Gold Price Predictor

A data-driven machine learning project designed to predict historical and future gold prices in the Bangladesh market by integrating local and global economic indicators.

## Project Overview

The Gold Price Predictor uses a multi-model approach (LSTM, XGBoost, ARIMA, and Facebook Prophet) to analyze gold price trends. It incorporates several key factors that influence gold prices specifically in Bangladesh:

- Historical Gold Prices: Daily records for 18K, 21K, 22K, and Traditional gold.
- USD to BDT Exchange Rate: Fluctuations in currency value.
- Inflation Rates: Bangladesh-specific inflation data.
- Seasonality: Local demand peaks, such as the wedding season.

## Key Features

- Automated Data Preprocessing: Handles missing dates using Cubic Spline Interpolation and converts absolute prices to stationary Log Returns.
- Feature Engineering: Includes lag features, rolling statistics (mean, volatility), and a binary flag for the local wedding season (November to February).
- Modular Training Pipeline: Separate training scripts for each model type, allowing for easy expansion.
- Dynamic Evaluation: Centralized evaluation script that automatically detects trained models and compares them using RMSE, MAE, R2, and Directional Accuracy.
- SHAP Integration: Provides interpretability for future price predictions by explaining the influence of individual features.

## Project Structure

```text
D:\Gold Price Predictor\
├── Dataset\                          # Raw and processed CSV data
│   ├── prices.csv                    # Raw gold price data
│   ├── export-Bangladesh-Official.csv # USD-BDT exchange rate
│   └── pattern_INFLATION_2.csv       # Bangladesh inflation data
├── Pre-Process\                      # Data cleaning and feature engineering
│   ├── preProcess_gold.py            # Gold price normalization
│   ├── preProcess_rate.py            # Exchange rate normalization
│   ├── preProcess_inflation.py       # Inflation data processing
│   └── mergeDataset.py               # Dataset merging and final scaling
├── training\                         # Model training scripts
│   ├── lstm_trainer.py               # Deep learning LSTM model
│   ├── xgboost_trainer.py            # Gradient-boosted trees
│   ├── arima_trainer.py              # Statistical time-series model
│   └── prophet_trainer.py            # Facebook Prophet additive model
├── evaluation\                       # Model assessment and inference
│   ├── evaluator.py                  # Comparative model analysis
│   └── predict_future.py             # Future price inference with SHAP
├── Models\                           # Saved weights, scalers, and tensors
│   ├── lstm_model.pth                # PyTorch model weights
│   ├── xgboost_model.joblib          # Trained XGBoost model
│   ├── feature_scaler.pkl            # StandardScaler for normalization
│   └── *_train.npy, *_test.npy       # Pre-generated tensors for training
├── PIPELINE.md                       # Detailed execution sequence
├── DATASET_DETAILS.md                # In-depth feature documentation
└── requirements.txt                  # Python dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "Gold Price Predictor"
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Execution

To train the models and generate predictions, execute the scripts in the following order:

1. Preprocess raw data:
   ```bash
   python Pre-Process\preProcess_gold.py
   python Pre-Process\preProcess_rate.py
   python Pre-Process\preProcess_inflation.py
   python Pre-Process\mergeDataset.py
   ```

2. Generate training tensors:
   ```bash
   python tensor_generator.py
   ```

3. Train the models:
   ```bash
   python training\lstm_trainer.py
   python training\xgboost_trainer.py
   python training\arima_trainer.py
   python training\prophet_trainer.py
   ```

4. Evaluate and Compare:
   ```bash
   python evaluation\evaluator.py
   ```

5. Predict future prices:
   ```bash
   python evaluation\predict_future.py
   ```

## Models Used

### LSTM (Long Short-Term Memory)
- Architecture: 2-layer LSTM with 64 hidden units, Batch Normalization, and Dropout.
- Goal: Capture complex non-linear dependencies in time-series data.

### XGBoost Regressor
- Architecture: Gradient-boosted decision trees.
- Goal: Efficiently handle tabular features like rolling averages and lag values.

### ARIMA (Auto-Regressive Integrated Moving Average)
- Goal: Serve as a statistical baseline for time-series forecasting.

### Facebook Prophet
- Goal: Handle seasonality and external regressors (Inflation, Exchange Rates) effectively.

## Dataset Features

The final training dataset (`master_training_data.csv`) includes:
- Log_Returns: Daily percentage change in gold price (target).
- Rate_Log_Returns: Daily change in USD-BDT rate.
- Daily_Inflation_Scaled: Normalized inflation rate.
- Wedding_Season_Flag: Binary (1/0) for peak demand months.
- Lags (1, 2, 3 days): Short-term momentum captures.
- Rolling Statistics (7d, 14d): Medium-term trend indicators.

## Evaluation Metrics

The project evaluates models based on:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 Score (Coefficient of Determination)
- Directional Accuracy (Percentage of correctly predicted price movement directions)
