import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the Master Dataset (RAW, unscaled)
df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\master_training_data.csv', parse_dates=['Date'], index_col='Date')

# 2. Define Window Parameters
LOOKBACK = 14  # Number of past days to look at (2 trading weeks)
FEATURES = len(df.columns)  # 12 features

print(f"Features ({FEATURES}): {df.columns.tolist()}")

# 3. Chronological Split FIRST (before scaling, to prevent leakage)
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size + val_size]
test_df = df.iloc[train_size + val_size:]

print(f"\nRaw split sizes:")
print(f"  Train: {len(train_df)} ({train_df.index.min().date()} to {train_df.index.max().date()})")
print(f"  Val:   {len(val_df)} ({val_df.index.min().date()} to {val_df.index.max().date()})")
print(f"  Test:  {len(test_df)} ({test_df.index.min().date()} to {test_df.index.max().date()})")

# 4. Fit StandardScaler on TRAINING DATA ONLY (prevents data leakage)
scaler = StandardScaler()
scaler.fit(train_df)  # Only learns mean/std from training period

# Transform all splits using the training scaler
train_scaled = pd.DataFrame(scaler.transform(train_df), columns=df.columns, index=train_df.index)
val_scaled = pd.DataFrame(scaler.transform(val_df), columns=df.columns, index=val_df.index)
test_scaled = pd.DataFrame(scaler.transform(test_df), columns=df.columns, index=test_df.index)

# Save scaler (fitted on training data only)
scaler_path = r'D:\Gold Price Predictor\Models\feature_scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"\nScaler saved (fitted on training data only): {scaler_path}")

# 5. Create windowed sequences from each split separately
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data.iloc[i : i + lookback].values)
        y.append(data.iloc[i + lookback, 0])  # Target: next day's scaled Log_Return
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOKBACK)
X_val, y_val = create_sequences(val_scaled, LOOKBACK)
X_test, y_test = create_sequences(test_scaled, LOOKBACK)

print(f"\nWindowing Complete.")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

# 6. Save tensors
np.save(r'D:\Gold Price Predictor\Models\X_train.npy', X_train)
np.save(r'D:\Gold Price Predictor\Models\y_train.npy', y_train)
np.save(r'D:\Gold Price Predictor\Models\X_val.npy', X_val)
np.save(r'D:\Gold Price Predictor\Models\y_val.npy', y_val)
np.save(r'D:\Gold Price Predictor\Models\X_test.npy', X_test)
np.save(r'D:\Gold Price Predictor\Models\y_test.npy', y_test)

print("\nTensors saved: X_train, y_train, X_val, y_val, X_test, y_test")