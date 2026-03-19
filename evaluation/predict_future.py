import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. Load Data =====================
# Load the UNSCALED master data to get raw feature values for iterative prediction
# We need to rebuild features from raw data, then scale only when feeding to model

# Load raw cleaned datasets
gold_df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\cleaned_daily_gold.csv',
                       parse_dates=['date'], index_col='date')
gold_df.index.name = 'Date'

usd_df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\cleaned_daily_usd_bdt.csv',
                       parse_dates=['Date'], index_col='Date')
inf_df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\cleaned_daily_inflation.csv',
                       parse_dates=[0], index_col=0)
inf_df.index.name = 'Date'

# Merge (same as mergeDataset.py)
merged = gold_df.join([usd_df, inf_df], how='inner')
merged = merged[merged.index >= gold_df.index.min()]
merged['Wedding_Season_Flag'] = merged.index.to_series().apply(
    lambda d: 1 if d.month in [11, 12, 1, 2] else 0)

# Add derived features (same as mergeDataset.py)
for lag in [1, 2, 3]:
    merged['Log_Returns_Lag{}'.format(lag)] = merged['Log_Returns'].shift(lag)
merged['Rolling_Mean_7d'] = merged['Log_Returns'].rolling(window=7).mean()
merged['Rolling_Mean_14d'] = merged['Log_Returns'].rolling(window=14).mean()
merged['Rolling_Volatility_7d'] = merged['Log_Returns'].rolling(window=7).std()
merged['Momentum_7d'] = merged['Log_Returns'].rolling(window=7).sum()
merged['Momentum_14d'] = merged['Log_Returns'].rolling(window=14).sum()

FEATURE_NAMES = [
    'Log_Returns', 'Rate_Log_Returns', 'Daily_Inflation_Scaled', 'Wedding_Season_Flag',
    'Log_Returns_Lag1', 'Log_Returns_Lag2', 'Log_Returns_Lag3',
    'Rolling_Mean_7d', 'Rolling_Mean_14d', 'Rolling_Volatility_7d',
    'Momentum_7d', 'Momentum_14d'
]

raw_df = merged[FEATURE_NAMES].dropna()

# Load the saved scaler
scaler = joblib.load(r'D:\Gold Price Predictor\Models\feature_scaler.pkl')

# Scale the data (same as mergeDataset.py does for training)
scaled_values = scaler.transform(raw_df)
scaled_df = pd.DataFrame(scaled_values, columns=FEATURE_NAMES, index=raw_df.index)

LOOKBACK = 14

print("=" * 65)
print("     GOLD PRICE PREDICTION FOR 2026-02-22 (XGBoost)")
print("=" * 65)

last_known_price = gold_df['traditional'].iloc[-1]
last_known_date = gold_df.index[-1]
k22_ratio = (gold_df['k22'] / gold_df['traditional']).iloc[-1]
print("Last known price: {} BDT (traditional) on {}".format(last_known_price, last_known_date.date()))

# ===================== 2. Train XGBoost on ALL scaled data =====================
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data.iloc[i : i + lookback].values)
        y.append(data.iloc[i + lookback, 0])
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(scaled_df, LOOKBACK)
X_all_flat = X_all.reshape(X_all.shape[0], -1)

xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_all_flat, y_all)
print("XGBoost trained on {} samples".format(len(X_all_flat)))

# ===================== 3. Single-step Prediction =====================
# Use the last LOOKBACK rows of SCALED data as input
window_scaled = scaled_df.iloc[-LOOKBACK:].values.copy()
input_flat = window_scaled.reshape(1, -1)

# Predict (output is scaled log return)
pred_scaled = xgb_model.predict(input_flat)[0]

# Inverse transform: we need to convert scaled prediction back to real log return
# The scaler's first column (index 0) corresponds to Log_Returns
log_return_mean = scaler.mean_[0]
log_return_std = scaler.scale_[0]
pred_real_log_return = pred_scaled * log_return_std + log_return_mean

# Convert to price
predicted_price = last_known_price * np.exp(pred_real_log_return)
k22_price = predicted_price * k22_ratio
pct = (predicted_price - last_known_price) / last_known_price * 100

print("\n" + "=" * 65)
print("  PREDICTED GOLD PRICE FOR 2026-02-22:")
print("  Traditional: {:,.2f} BDT per bhori".format(predicted_price))
print("  22 Karat:    {:,.2f} BDT per bhori".format(k22_price))
print("  Predicted log return: {:.6f} (real scale)".format(pred_real_log_return))
print("  Change: {:+.2f}% from {} ({:,.0f} BDT)".format(pct, last_known_date.date(), last_known_price))
print("=" * 65)

# ===================== 4. SHAP Explanation =====================
flat_names = []
for d in range(1, LOOKBACK + 1):
    for f in FEATURE_NAMES:
        flat_names.append("D{}_{}".format(d, f))

explainer = shap.TreeExplainer(xgb_model)
sv = explainer.shap_values(input_flat)

shap_df = pd.DataFrame({
    'feat': flat_names,
    'shap': sv[0],
})
shap_df['abs'] = shap_df['shap'].abs()
shap_df = shap_df.sort_values('abs', ascending=False)

print("\n--- WHAT CAUSED THIS PREDICTION? ---\n")
print("Top 10 driving features:")
print("{:<6} {:<35} {:>12} {:>10}".format('Rank', 'Feature', 'SHAP Value', 'Direction'))
print("-" * 65)
rank = 1
for _, r in shap_df.head(10).iterrows():
    d = "UP" if r['shap'] > 0 else "DOWN"
    print("{:<6} {:<35} {:>+12.6f} {:>10}".format(rank, r['feat'], r['shap'], d))
    rank += 1

print("\n--- AGGREGATED BY FEATURE TYPE ---\n")
for feat in FEATURE_NAMES:
    mask = shap_df['feat'].str.contains(feat)
    total = shap_df.loc[mask, 'shap'].sum()
    d = "PUSHES UP" if total > 0 else "PUSHES DOWN"
    print("  {:<30} {:>+10.6f}  {}".format(feat, total, d))
