import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_prophet():
    """
    Train a Prophet model on gold price log returns.
    
    Prophet works differently from LSTM/XGBoost:
    - It expects a DataFrame with 'ds' (datestamp) and 'y' (target) columns
    - It automatically handles trend, seasonality, and holidays
    - We add external regressors for the base features
    """
    from prophet import Prophet

    # Load the raw master training data (unscaled)
    df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\master_training_data.csv',
                     parse_dates=['Date'], index_col='Date')
    
    # Prepare Prophet format
    prophet_df = pd.DataFrame({
        'ds': df.index,
        'y': df['Log_Returns'].values,
    })
    
    # Add external regressors (base features only)
    extra_regressors = ['Rate_Log_Returns', 'Daily_Inflation_Scaled', 'Wedding_Season_Flag']
    
    for reg in extra_regressors:
        prophet_df[reg] = df[reg].values
    
    # Chronological split (70/15/15)
    train_size = int(len(prophet_df) * 0.70)
    val_size = int(len(prophet_df) * 0.15)
    
    train_df = prophet_df.iloc[:train_size]
    val_df = prophet_df.iloc[train_size:train_size + val_size]
    test_df = prophet_df.iloc[train_size + val_size:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize Prophet model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode='additive'
    )
    
    # Register external regressors
    for reg in extra_regressors:
        model.add_regressor(reg)
    
    # Fit on training data
    print("Training Prophet model...")
    model.fit(train_df)
    
    # Validate
    print("Validating...")
    val_forecast = model.predict(val_df)
    val_preds = val_forecast['yhat'].values
    val_actual = val_df['y'].values
    
    da = np.mean(np.sign(val_actual) == np.sign(val_preds)) * 100
    mae = np.mean(np.abs(val_actual - val_preds))
    print(f"Validation Dir. Accuracy: {da:.2f}%")
    print(f"Validation MAE: {mae:.6f}")
    
    # Save model
    model_path = r'D:\Gold Price Predictor\Models\prophet_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nProphet model saved to {model_path}")

if __name__ == '__main__':
    train_prophet()
