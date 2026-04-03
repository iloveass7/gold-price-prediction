import numpy as np
import joblib
from xgboost import XGBRegressor
import os

def train_xgboost(n_estimators=100, max_depth=5, learning_rate=0.1):
    # Load and Flatten Tensors
    X_train = np.load(r'D:\Gold Price Predictor\Models\X_train.npy')
    y_train = np.load(r'D:\Gold Price Predictor\Models\y_train.npy')
    X_val = np.load(r'D:\Gold Price Predictor\Models\X_val.npy')
    y_val = np.load(r'D:\Gold Price Predictor\Models\y_val.npy')

    # XGBoost expects 2D inputs
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    print("Starting XGBoost training...")
    model = XGBRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=learning_rate,
        random_state=42
    )
    
    model.fit(
        X_train_flat, y_train,
        eval_set=[(X_val_flat, y_val)],
        verbose=False
    )
    
    model_path = r'D:\Gold Price Predictor\Models\xgboost_model.joblib'
    joblib.dump(model, model_path)
    print(f"XGBoost Model saved to {model_path}")

if __name__ == '__main__':
    train_xgboost()
