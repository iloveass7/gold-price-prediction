import pandas as pd
import numpy as np

# 1. Load the Master Dataset
df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\master_training_data.csv', parse_dates=['Date'], index_col='Date')

# 2. Define Window Parameters [cite: 76, 99]
LOOKBACK = 7  # Number of past days to look at
FEATURES = 4   # Gold Log Returns, Rate Log Returns, Inflation, Wedding Flag

def create_sequences(data, lookback):
    X, y = [], []
    # We loop through the data to create overlapping 30-day windows
    for i in range(len(data) - lookback):
        # The input: 30 days of data
        X.append(data.iloc[i : i + lookback].values)
        # The target: The very next day's Gold Log Return [cite: 51]
        y.append(data.iloc[i + lookback, 0])
    return np.array(X), np.array(y)

# 3. Generate Tensors
# Convert the dataframe to a numpy array for processing
data_array = df.values
X, y = create_sequences(df, LOOKBACK)

# 4. Chronological Split (Train/Val/Test)
# To avoid data leakage, we do NOT shuffle; we split by time
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Windowing Complete.")
print(f"Input Shape (X_train): {X_train.shape}")  # Should be [Samples, 30, 4]
print(f"Target Shape (y_train): {y_train.shape}")

np.save(r'D:\Gold Price Predictor\Models\X_train.npy', X_train)
np.save(r'D:\Gold Price Predictor\Models\X_test.npy', X_test)
np.save(r'D:\Gold Price Predictor\Models\y_train.npy', y_train)
np.save(r'D:\Gold Price Predictor\Models\y_test.npy', y_test)

print("Tensors saved to disk: X_train.npy, X_test.npy, y_train.npy, y_test.npy")