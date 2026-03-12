import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_arima():
    """
    Train an ARIMA model on gold price log returns.
    
    ARIMA works differently from LSTM/XGBoost:
    - It operates on a single time series (not windowed multi-feature tensors)
    - It uses auto-regression (AR), differencing (I), and moving averages (MA)
    - We use pmdarima's auto_arima to find the best (p,d,q) parameters
    """
    from pmdarima import auto_arima

    # Load the master training data
    df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\master_training_data.csv',
                     parse_dates=['Date'], index_col='Date')
    
    # ARIMA uses the raw Log_Returns series (column 0)
    log_returns = df['Log_Returns']
    
    # Chronological split (70/15/15 to match other models)
    train_size = int(len(log_returns) * 0.70)
    val_size = int(len(log_returns) * 0.15)
    
    train_series = log_returns.iloc[:train_size]
    val_series = log_returns.iloc[train_size:train_size + val_size]
    test_series = log_returns.iloc[train_size + val_size:]
    
    print(f"Train: {len(train_series)}, Val: {len(val_series)}, Test: {len(test_series)}")
    
    # Auto ARIMA: automatically searches for best (p,d,q) parameters
    print("Running Auto ARIMA (this may take a few minutes)...")
    model = auto_arima(
        train_series,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=0,              # Log returns are already stationary (differenced)
        seasonal=False,    # We handle seasonality via features, not ARIMA
        stepwise=True,     # Faster search using stepwise algorithm
        suppress_warnings=True,
        trace=True         # Print search progress
    )
    
    print(f"\nBest ARIMA order: {model.order}")
    print(model.summary())
    
    # Validate on validation set using walk-forward prediction
    print("\nValidating with walk-forward prediction...")
    val_preds = []
    history = list(train_series.values)
    
    for i in range(len(val_series)):
        # Refit on expanding window and predict 1 step ahead
        fc = model.predict(n_periods=1)
        val_preds.append(fc[0])
        # Add actual observation to history and update model
        model.update([val_series.iloc[i]])
    
    val_preds = np.array(val_preds)
    val_actual = val_series.values
    
    # Directional accuracy on validation
    da = np.mean(np.sign(val_actual) == np.sign(val_preds)) * 100
    mae = np.mean(np.abs(val_actual - val_preds))
    print(f"Validation Dir. Accuracy: {da:.2f}%")
    print(f"Validation MAE: {mae:.6f}")
    
    # Save model
    model_path = r'D:\Gold Price Predictor\Models\arima_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nARIMA model saved to {model_path}")

if __name__ == '__main__':
    train_arima()
