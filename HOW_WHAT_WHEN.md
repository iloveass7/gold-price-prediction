# Gold Price Predictor — How, What & When

This document provides a comprehensive narrative of the Gold Price Predictor project: **what** was built, **how** it was accomplished, and **when** each milestone occurred.

---

## What: Project Summary

The **Gold Price Predictor** is a machine learning pipeline designed to predict gold prices in the Bangladesh market. The project integrates multiple economic indicators — historical gold prices, USD-BDT exchange rates, Bangladesh inflation data, and local seasonality (wedding season) — to train and compare four different forecasting models:

- **LSTM** (Long Short-Term Memory neural network)
- **XGBoost** (Gradient-Boosted Decision Trees)
- **ARIMA** (Auto-Regressive Integrated Moving Average)
- **Facebook Prophet** (Additive time-series model)

The target variable is the **Log Return** of traditional gold prices, which represents the daily percentage change. By predicting log returns instead of absolute prices, the data is made **stationary**, which is a fundamental requirement for time-series forecasting.

### What Data Was Used

| Dataset | Source File | Description |
| :--- | :--- | :--- |
| Gold Prices | `prices.csv` | Daily prices for 18K, 21K, 22K, and Traditional gold (BDT/bhori) starting from 2007 |
| USD-BDT Rate | `export-Bangladesh-Official.csv` | Official Bangladesh Bank exchange rates (from Aug 2008, backfilled to 2007) |
| Inflation | `pattern_INFLATION_2.csv` | Annual Bangladesh inflation rates, interpolated to daily |

### What Features Were Engineered

The final training dataset (`master_training_data.csv`) contains **4 base features**:

| Feature | Description | Engineering Method |
| :--- | :--- | :--- |
| `Log_Returns` | Daily % change in gold price **(target)** | `ln(Price_t / Price_{t-1})` |
| `Rate_Log_Returns` | Daily % change in USD-BDT rate | `ln(Rate_t / Rate_{t-1})` |
| `Daily_Inflation_Scaled` | Normalized daily inflation rate | Annual → Daily via cubic spline, then `/100` |
| `Wedding_Season_Flag` | Binary indicator for high-demand months | `1` if month ∈ {Nov, Dec, Jan, Feb}, else `0` |

Additionally, **`Days_Since_Last_Trade`** is computed in preprocessing to encode time gaps (weekends, holidays) before non-trading days are removed.

### What Models Were Built

#### 1. LSTM (Deep Learning)
- **Architecture**: 2-layer LSTM with 64 hidden units
- **Regularization**: BatchNorm + Dropout (0.3)
- **Training**: AdamW optimizer, ReduceLROnPlateau scheduler, Early Stopping (patience=15)
- **Input**: 14-day sliding window sequences (14 × 4 features)

#### 2. XGBoost (Gradient Boosting)
- **Architecture**: 100 estimators, max depth 5, learning rate 0.1
- **Input**: Flattened 14-day windows (56-dimensional vectors)

#### 3. ARIMA (Statistical)
- **Architecture**: Auto-selected (p,d,q) via `pmdarima.auto_arima`
- **Note**: d=0 since log returns are already stationary
- **Evaluation**: Walk-forward validation (one-step-ahead predictions)

#### 4. Facebook Prophet (Additive)
- **Architecture**: Weekly + Yearly seasonality, additive mode
- **External Regressors**: Rate_Log_Returns, Daily_Inflation_Scaled, Wedding_Season_Flag
- **Changepoint Prior Scale**: 0.05

---

## How: Technical Methodology

### Step 1: Data Collection & Cleaning

**Gold Prices** (`preProcess_gold.py`):
1. Loaded raw `prices.csv` and parsed dates.
2. Split data at April 4, 2024 (transition point between sparse historical and dense modern data).
3. Resampled historical segment to daily frequency.
4. Applied **Cubic Spline Interpolation** to fill multi-month gaps in historical data.
5. Recombined with modern (daily) segment.
6. Calculated **Log Returns**: `ln(Price_today / Price_yesterday)`.
7. Computed `Days_Since_Last_Trade` to encode gap context.
8. **Removed non-trading days** (zero-return rows from weekends, holidays, unchanged prices).

**USD-BDT Exchange Rate** (`preProcess_rate.py`):
1. Loaded raw exchange rate CSV with date parsing.
2. Backfilled to January 2007 using the first known rate (~68.50 BDT/USD).
3. Resampled to daily, applied **Linear Interpolation** for gaps.
4. Calculated **Rate Log Returns**.

**Inflation** (`preProcess_inflation.py`):
1. Loaded annual inflation rates.
2. Created daily index and resampled via **Cubic Spline Interpolation**.
3. Scaled annual percentages to daily magnitude (`/100`).

### Step 2: Dataset Merging & Feature Engineering

**Merge** (`mergeDataset.py`):
1. Inner-joined all three cleaned datasets on date index.
2. Engineered the **Wedding Season Flag** (Nov–Feb = 1).
3. Selected 4 final base features.
4. Exported as `master_training_data.csv` — **unscaled** (scaling deferred to prevent data leakage).

### Step 3: Tensor Generation & Scaling

**Tensor Generator** (`tensor_generator.py`):
1. Performed **chronological split**: 70% train / 15% validation / 15% test.
2. Fitted `StandardScaler` on **training data only** (prevents data leakage — this was a critical fix).
3. Transformed all splits using the training scaler.
4. Created **sliding window sequences** with `LOOKBACK = 14` (14 trading days ≈ 2 weeks).
5. Saved numpy tensors (`X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`) and the scaler.

### Step 4: Model Training

Each model was trained independently via its own script in the `training/` directory:
- **LSTM**: Trained with early stopping on validation loss, learning rate decay, and batch training.
- **XGBoost**: Trained on flattened windows with validation-based early stopping.
- **ARIMA**: Auto-fitted optimal (p,d,q) on training series, validated via walk-forward.
- **Prophet**: Fitted on training portion with external regressors, validated on held-out data.

### Step 5: Evaluation

**Centralized Evaluator** (`evaluator.py`):
1. Loaded test tensors and all saved models.
2. Generated predictions from each model.
3. Computed metrics: **RMSE**, **MAE**, **R²**, and **Directional Accuracy**.
4. Included a **Naive Baseline** (always predict 0) for honest comparison.
5. Generated a `comparison_plot.png` overlaying all predictions vs actual.

### Step 6: Future Prediction with SHAP

**Predict Future** (`predict_future.py`):
1. Reconstructed the full feature pipeline from raw data.
2. Scaled using the saved training scaler.
3. Used XGBoost to predict the next day's log return.
4. Inverse-transformed back to BDT price.
5. Used **SHAP** (SHapley Additive exPlanations) to explain which features drove the prediction.

---

## How: Critical Design Decisions

### 1. Log Returns Instead of Absolute Prices
Absolute gold prices are non-stationary (trending upward over 18 years). Log returns `ln(P_t / P_{t-1})` make the series stationary, which is a prerequisite for ML models to learn meaningful patterns rather than just memorizing trends.

### 2. Removing Non-Trading Days (Zero-Return Filtering)
Originally, weekends and holidays were kept in the dataset as zero-return days. This caused **39.7% of all training samples to have a target of exactly 0.0**, which overwhelmed the actual price movement signal. Removing these days was the single most impactful improvement. See [RESULTS.md](RESULTS.md) for the before/after comparison.

### 3. Scaling After Split (Preventing Data Leakage)
Initially, `StandardScaler` was fitted on the **entire dataset** before splitting — meaning the scaler "saw" future data during training. This was fixed by fitting the scaler **only on training data**, then transforming validation and test sets with the same parameters.

### 4. LSTM Right-Sizing
The original LSTM had **512 hidden units × 4 layers ≈ 4.2M parameters** for only ~5,000 training samples (a 760:1 parameter-to-sample ratio). This was reduced to **64 hidden × 2 layers** with BatchNorm and Dropout to prevent severe overfitting.

### 5. Days_Since_Last_Trade Feature
When non-trading days are removed, the model loses context about time gaps. The `Days_Since_Last_Trade` feature preserves this information: a value of 1 means a normal trading day, 2–3 means a weekend gap, and 4+ means a holiday period.

---

## When: Project Timeline

| Date | Milestone | Git Commit |
| :--- | :--- | :--- |
| **2026-02-20** | Initial data preprocessing complete. Built tensor generator with sliding window approach. | `c6acc8f` — *"tensor done + sliding window + feature eng"* |
| **2026-02-22** | First model training run. LSTM and XGBoost trained; initial evaluation performed. Results showed **negative R² scores** — both models performed worse than predicting zero. Diagnosed root causes: zero contamination, oversized LSTM, data leakage from scaler, poor feature set. | `645b7ab` — *"Initial training"* |
| **2026-02-22 → 2026-03-12** | Major refactoring phase: removed zero-return days, right-sized LSTM (512→64 hidden, 4→2 layers), added BatchNorm/Dropout/EarlyStopping, fixed scaler leakage, added ARIMA and Prophet models, removed engineered lag/rolling features to use only base features for honest evaluation. | Development period |
| **2026-03-12** | ARIMA and Prophet models added. Zero-return filtering implemented. Final pipeline with 4 models operational. | `6b1a6ff` — *"arima + prophet + -0logs"* |
| **2026-04-03** | Project documentation created (this file + RESULTS.md). | Current |

---

## Pipeline Execution Order

```
1.  python Pre-Process/preProcess_gold.py         # Clean gold prices, interpolate, log returns, remove zeros
2.  python Pre-Process/preProcess_rate.py          # Clean USD-BDT rates, log returns
3.  python Pre-Process/preProcess_inflation.py     # Interpolate annual inflation to daily
4.  python Pre-Process/mergeDataset.py             # Merge all data, feature engineering, export unscaled
5.  python Pre-Process/tensor_generator.py         # Split → Scale (train only) → Window → Save tensors
6.  python training/lstm_trainer.py                # Train LSTM
7.  python training/xgboost_trainer.py             # Train XGBoost
8.  python training/arima_trainer.py               # Train ARIMA (auto parameter search)
9.  python training/prophet_trainer.py             # Train Prophet
10. python evaluation/evaluator.py                 # Compare all models
11. python evaluation/predict_future.py            # Predict next day's gold price + SHAP
```
