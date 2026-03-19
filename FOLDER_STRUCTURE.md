# Gold Price Predictor — Folder Structure

```
D:\Gold Price Predictor\
│
├── Dataset\                          # Raw source data (DO NOT DELETE)
│   ├── prices.csv                    # Raw gold prices (k18, k21, k22, traditional)
│   ├── export-Bangladesh-Official.csv # USD-BDT exchange rate data
│   └── pattern_INFLATION_2.csv       # Bangladesh inflation data
│
├── Pre-Process\                      # Data cleaning & feature engineering
│   ├── preProcess_gold.py            # Clean gold prices, cubic spline, log returns
│   ├── preProcess_rate.py            # Clean USD-BDT exchange rate
│   ├── preProcess_inflation.py       # Clean inflation data
│   └── mergeDataset.py               # Merge all data, feature engineering, StandardScaler
│
├── training\                         # Model training scripts
│   ├── lstm_trainer.py               # LSTM (64 hidden, 2 layers, BatchNorm, Dropout)
│   ├── xgboost_trainer.py            # XGBoost Regressor
│   ├── arima_trainer.py              # Auto ARIMA with walk-forward validation
│   └── prophet_trainer.py            # Facebook Prophet with external regressors
│
├── evaluation\                       # Model evaluation & prediction
│   ├── evaluator.py                  # Compare all models (LSTM, XGBoost, ARIMA, Prophet)
│   └── predict_future.py             # Predict future gold prices with SHAP explanation
│
├── Models\                           # Generated model files & tensors (auto-generated)
│   ├── X_train.npy, y_train.npy      # Training tensors
│   ├── X_val.npy, y_val.npy          # Validation tensors
│   ├── X_test.npy, y_test.npy        # Test tensors
│   ├── feature_scaler.pkl            # StandardScaler for feature normalization
│   ├── lstm_model.pth                # Trained LSTM weights
│   ├── xgboost_model.joblib          # Trained XGBoost model
│   ├── arima_model.pkl               # Trained ARIMA model
│   └── prophet_model.pkl             # Trained Prophet model
│
├── tensor_generator.py               # Creates windowed sequences & train/val/test split
├── DATASET_DETAILS.md                # Documentation of datasets
├── PIPELINE.md                       # Pipeline execution order
└── FOLDER_STRUCTURE.md               # This file
```

## Pipeline Execution Order

```
1. py Pre-Process\preProcess_gold.py
2. py Pre-Process\preProcess_rate.py
3. py Pre-Process\preProcess_inflation.py    (if needed)
4. py Pre-Process\mergeDataset.py
5. py tensor_generator.py
6. py training\lstm_trainer.py
7. py training\xgboost_trainer.py
8. py training\arima_trainer.py
9. py training\prophet_trainer.py
10. py evaluation\evaluator.py
```
