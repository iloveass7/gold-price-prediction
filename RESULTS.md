# Gold Price Predictor — Results & Findings

This document presents the experimental results of the Gold Price Predictor project, including the critical comparison between the **before** (with zero-return contamination) and **after** (with zeros removed) scenarios.

---

## Table of Contents

1. [Experiment 1: Before Zero Removal (Failed Baseline)](#experiment-1-before-zero-removal-failed-baseline)
2. [Experiment 2: After Zero Removal (Final Results)](#experiment-2-after-zero-removal-final-results)
3. [Why Removing Zeros Mattered](#why-removing-zeros-mattered)
4. [Model-by-Model Analysis](#model-by-model-analysis)
5. [Key Takeaways](#key-takeaways)

---

## Experiment 1: Before Zero Removal (Failed Baseline)

> **Date**: February 22, 2026
> **Commit**: `645b7ab` — *"Initial training"*
> **Configuration**: All interpolated days kept (including weekends, holidays), LSTM with 512 hidden units × 4 layers, no BatchNorm/Dropout, scaler fitted on full dataset (data leakage)

### The Problem

After cubic spline interpolation filled all missing dates, **39.7% of all `Log_Returns` were exactly 0.0** — these came from weekends, public holidays, and days where the gold price didn't change. The models were trained on a dataset where nearly 4 out of 10 samples had a target of zero.

#### Zero Contamination Statistics (Before Fix)
| Metric | Value |
| :--- | :--- |
| Total samples | ~6,238 |
| Samples with `Log_Returns = 0` | ~2,477 |
| **Percentage of zeros** | **39.7%** |
| Number of consecutive-zero streaks | 117 |
| Longest consecutive-zero streak | 31 days |

### Results (Before Zero Removal)

| Model | RMSE | MAE | R² | Dir. Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | ~0.010 | ~0.004 | **-0.65** | ~58% |
| **XGBoost** | ~0.010 | ~0.004 | **-0.65** | ~58% |
| **Naive (predict 0)** | ~0.008 | ~0.004 | **-0.04** | ~35% |

### Interpretation

- **Negative R² scores** mean the models performed **worse than simply predicting the mean** (which is approximately zero for log returns).
- Both LSTM and XGBoost had **higher RMSE and MAE than the Naive baseline** — meaning you'd literally be better off predicting "no change" every single day.
- The ~58% directional accuracy sounds reasonable, but it's **inflated** because the models learned to predict small values (near zero) most of the time — matching the direction of the 39.7% of true-zero days.
- The LSTM was also massively overfitted: **4.2 million parameters** trained on only ~5,500 samples (a 760:1 ratio).
- `StandardScaler` was fitted on the full dataset **before** the train/test split, introducing **data leakage** — the scaler "saw" test-period statistics during training.

> **Verdict**: Complete failure. The models learned to predict "approximately zero" because that was the statistically dominant pattern in the data. No actual market signal was captured.

---

## Experiment 2: After Zero Removal (Final Results)

> **Date**: March 12, 2026
> **Commit**: `6b1a6ff` — *"arima + prophet + -0logs"*
> **Configuration**: Non-trading days removed, LSTM right-sized to 64 hidden × 2 layers with BatchNorm + Dropout, scaler fitted on training data only, ARIMA and Prophet models added

### Changes Made

1. **Removed all zero-return rows** from the gold price data (weekends, holidays, unchanged prices)
2. **Added `Days_Since_Last_Trade`** feature to preserve time-gap context
3. **Right-sized the LSTM**: 512×4 → 64×2, added BatchNorm + Dropout (0.3)
4. **Fixed data leakage**: `StandardScaler` now fitted on training split only
5. **Added early stopping** (patience=15) and **ReduceLROnPlateau** scheduler
6. **Added ARIMA and Prophet** models for broader comparison
7. **Increased LOOKBACK** from 7 to 14 days (2 trading weeks)
8. Used **base features only** (removed engineered lags/rolling stats from `mergeDataset.py`)

### Results (After Zero Removal)

Based on the final model comparison on the test set:

| Model | RMSE | MAE | R² | Dir. Accuracy | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LSTM** | Best among ML models | Low | Positive | Higher | Tracks large movements well; occasional overshooting |
| **XGBoost** | Competitive | Low | Positive | Higher | Good at capturing turning points; more responsive |
| **ARIMA** | Moderate | Moderate | Near zero | ~50% | Converges to mean quickly; struggles with volatility |
| **Prophet** | Moderate | Moderate | Near zero | ~50% | Captures seasonality but weak on daily returns |
| **Naive** | Baseline | Baseline | ~0.0 | ~50% | Always predicts 0 (no change) |

### Visual Comparison

The comparison plot (`evaluation/comparison_plot.png`) shows all four models overlaid on the test set. Key observations:

- **LSTM and XGBoost** closely track the actual signal, especially during high-volatility periods (the large dip around sample 200 and the recovery around sample 250-300).
- **ARIMA and Prophet** essentially flatline near zero — they revert to the mean almost immediately and cannot capture the magnitude of price movements.
- The actual signal shows **significant regime changes** in the test period: a calm period, a dramatic crash, a sharp recovery, and then a stabilization phase.

---

## Why Removing Zeros Mattered

### The Core Issue: Signal-to-Noise Ratio

When 39.7% of your training target is exactly zero, the model's optimal strategy becomes **"predict zero (or near-zero) for everything"**. This is because:

1. **Loss function optimization**: With MSE loss, the model minimizes total error. Predicting zero for 40% of samples is "free" — it perfectly matches those targets. Any attempt to predict non-zero values on the other 60% is risky because it might also generate errors on the zero-target days.

2. **Gradient dilution**: During backpropagation, 40% of gradient updates push the model toward zero. This dilutes the meaningful gradients from actual price movements.

3. **Feature-target decorrelation**: On zero-return days, the features (exchange rate, inflation, etc.) still have non-zero values. This teaches the model that features are **uninformative** — a catastrophic lesson.

### Before vs After: Side-by-Side

| Aspect | Before (With Zeros) | After (Zeros Removed) |
| :--- | :--- | :--- |
| **Zero-return %** | 39.7% | ~0% |
| **Training samples** | ~6,238 | ~3,761 (trading days only) |
| **R² Score** | -0.65 (worse than baseline) | Positive (better than baseline) |
| **Models beat Naive?** | ❌ No — Naive was better | ✅ Yes — LSTM and XGBoost outperform |
| **Dir. Accuracy** | ~58% (inflated by zeros) | Honest metric on actual movements |
| **Signal quality** | Overwhelmed by noise | Clean trading-day signal |

### The Counter-Intuitive Part

Removing 40% of the data and training on **fewer samples** produced **better results**. This demonstrates that **data quality > data quantity** in machine learning. The 2,477 zero-return samples weren't just useless — they were actively harmful because they taught the model the wrong patterns.

---

## Model-by-Model Analysis

### LSTM (Long Short-Term Memory)

**Strengths**:
- Best at capturing non-linear relationships and complex temporal patterns
- Tracks large market movements (crashes and recoveries) more closely than other models
- BatchNorm and Dropout effectively prevent overfitting (after the architecture fix)

**Weaknesses**:
- Tendency to overshoot during extreme events (visible in the comparison plot around sample 250)
- Requires more data than statistical methods to generalize well
- Black-box nature makes it harder to interpret

**Architecture Evolution**:
| Parameter | Before | After |
| :--- | :--- | :--- |
| Hidden units | 512 | 64 |
| Layers | 4 | 2 |
| Parameters | ~4.2M | ~50K |
| Regularization | None | BatchNorm + Dropout(0.3) |
| Early stopping | No | Yes (patience=15) |
| LR scheduler | No | ReduceLROnPlateau |

### XGBoost (Gradient Boosted Trees)

**Strengths**:
- Most responsive to short-term turning points
- Naturally handles feature importance (no scaling needed internally)
- Fast training and inference

**Weaknesses**:
- Treats each flattened window as independent features (doesn't inherently model sequences)
- Can overfit on high-dimensional flattened input (14 × 4 = 56 features)

**Best Use Case**: Quick, interpretable predictions with SHAP explanation. The `predict_future.py` script uses XGBoost as the default prediction model specifically because of its SHAP compatibility.

### ARIMA (Auto-Regressive Integrated Moving Average)

**Strengths**:
- Theoretically sound for stationary time series
- Provides a solid statistical baseline
- Automatic parameter selection via `auto_arima`

**Weaknesses**:
- Univariate only — ignores exchange rate, inflation, and seasonality features
- Reverts to mean very quickly (predicts ~0 after a few steps)
- Walk-forward evaluation is computationally expensive
- Essentially matches the Naive baseline in performance

**Observation**: ARIMA's behavior of predicting near-zero is actually *theoretically correct* for a stationary series — the best prediction for a mean-reverting process is the mean (≈0 for log returns). This validates our preprocessing but shows ARIMA's limitation: it cannot capture the short-term patterns that LSTM and XGBoost exploit.

### Facebook Prophet

**Strengths**:
- Native support for external regressors and seasonality
- Handles irregular time series well
- Interpretable components (trend, seasonality, regressors)

**Weaknesses**:
- Designed for longer-horizon, smoother forecasting (not daily financial returns)
- Seasonality patterns in gold returns are weak
- Similar to ARIMA, reverts to near-zero predictions

**Observation**: Prophet was designed for business metrics (sales, web traffic) with strong weekly/yearly patterns. Daily gold log returns are too noisy and lack the stable seasonality that Prophet excels at modeling.

---

## Key Takeaways

### 1. Data Quality > Data Quantity
Removing 40% of the data (zero-return non-trading days) dramatically improved model performance. The remaining 60% contained the actual market signal. This is a textbook example of the "garbage in, garbage out" principle.

### 2. Directional Accuracy Is the Key Financial Metric
For trading applications, **whether you correctly predict the direction** (up vs down) matters more than the exact magnitude. A model with 70% directional accuracy can be profitable even with poor R² scores.

### 3. Simple Baselines Are Essential
The **Naive baseline** (always predict 0) exposed that both initial models were worthless. Without this baseline, the ~58% directional accuracy might have seemed acceptable. Always compare against trivial strategies.

### 4. Data Leakage Is Subtle but Devastating
Fitting `StandardScaler` on the full dataset before splitting is a common mistake. It means the scaler's mean and standard deviation incorporate information from the test period, giving the model an unfair advantage that doesn't exist in real-world deployment.

### 5. LSTM Architecture Must Match Data Scale
The original 4.2M-parameter LSTM was absurdly oversized for ~5,000 samples. The 50K-parameter version (64 hidden × 2 layers) performs better because it's forced to learn generalizable patterns instead of memorizing training noise.

### 6. Not All Models Suit All Problems
ARIMA and Prophet are excellent tools for many forecasting problems, but **daily financial return prediction** isn't their strength. LSTM and XGBoost outperform because they can model non-linear, multi-feature relationships that statistical models miss.

### 7. SHAP Provides Actionable Interpretability
The XGBoost + SHAP combination explains *why* a specific prediction was made, breaking down the contribution of each feature from each day in the lookback window. This is crucial for building trust in financial ML systems.

---

## Future Work

- **Add engineered features back carefully**: Lag features and rolling statistics showed strong correlations (7-day rolling mean had 0.78 correlation with next-day returns). These were removed for honest initial evaluation but should be reintroduced with proper evaluation.
- **Hyperparameter tuning**: Systematic grid search or Bayesian optimization for LSTM hidden size, XGBoost depth/estimators.
- **Ensemble methods**: Combine LSTM and XGBoost predictions (e.g., weighted average, stacking).
- **Walk-forward validation**: Implement expanding window backtesting for more realistic performance estimates.
- **Additional data sources**: Global gold price (XAUUSD), central bank interest rates, geopolitical risk indices.
