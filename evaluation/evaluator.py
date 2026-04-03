import numpy as np
import torch
import joblib
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path so we can import from training/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.lstm_trainer import GoldPriceLSTM

def get_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def directional_accuracy(actual, pred):
    actual_dir = np.sign(actual)
    pred_dir = np.sign(pred)
    return np.mean(actual_dir == pred_dir) * 100

def run_evaluation():
    # 1. Load Test Data
    X_test = np.load(r'D:\Gold Price Predictor\Models\X_test.npy')
    y_test = np.load(r'D:\Gold Price Predictor\Models\y_test.npy')
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    # 2. Evaluate LSTM
    lstm_path = r'D:\Gold Price Predictor\Models\lstm_model.pth'
    if os.path.exists(lstm_path):
        print("Evaluating LSTM...")
        model = GoldPriceLSTM(input_size=X_test.shape[2]).to(device)
        model.load_state_dict(torch.load(lstm_path, map_location=device))
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test).to(device)).cpu().numpy().flatten()
            results['LSTM'] = preds

    # 3. Evaluate XGBoost
    xgb_path = r'D:\Gold Price Predictor\Models\xgboost_model.joblib'
    if os.path.exists(xgb_path):
        print("Evaluating XGBoost...")
        model = joblib.load(xgb_path)
        preds = model.predict(X_test_flat)
        results['XGBoost'] = preds

    # 4. Evaluate ARIMA
    arima_path = r'D:\Gold Price Predictor\Models\arima_model.pkl'
    if os.path.exists(arima_path):
        print("Evaluating ARIMA (walk-forward on test set)...")
        arima_model = joblib.load(arima_path)
        
        # Load full data to get test period for walk-forward
        df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\master_training_data.csv',
                         parse_dates=['Date'], index_col='Date')
        log_returns = df['Log_Returns']
        
        train_size = int(len(log_returns) * 0.70)
        val_size = int(len(log_returns) * 0.15)
        test_series = log_returns.iloc[train_size + val_size:]
        
        arima_preds = []
        for i in range(len(test_series)):
            fc = arima_model.predict(n_periods=1)
            arima_preds.append(fc[0])
            arima_model.update([test_series.iloc[i]])
        
        results['ARIMA'] = np.array(arima_preds)
        # Use ARIMA's own test set (same period but may differ in length from tensor test)
        # We'll align by using min length
        min_len = min(len(y_test), len(arima_preds))
        results['ARIMA'] = np.array(arima_preds[:min_len])

    # 5. Evaluate Prophet
    prophet_path = r'D:\Gold Price Predictor\Models\prophet_model.pkl'
    if os.path.exists(prophet_path):
        print("Evaluating Prophet...")
        prophet_model = joblib.load(prophet_path)
        
        df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\master_training_data.csv',
                         parse_dates=['Date'], index_col='Date')
        
        train_size = int(len(df) * 0.70)
        val_size = int(len(df) * 0.15)
        test_raw = df.iloc[train_size + val_size:]
        
        extra_regressors = ['Rate_Log_Returns', 'Daily_Inflation_Scaled', 'Wedding_Season_Flag']
        
        test_prophet = pd.DataFrame({'ds': test_raw.index})
        for reg in extra_regressors:
            test_prophet[reg] = test_raw[reg].values
        
        forecast = prophet_model.predict(test_prophet)
        prophet_preds = forecast['yhat'].values
        min_len = min(len(y_test), len(prophet_preds))
        results['Prophet'] = prophet_preds[:min_len]

    # 6. Naive Baseline
    results['Naive'] = np.zeros_like(y_test)

    # 7. Print Comparison Table
    print("\n" + "=" * 60)
    print("              DYNAMIC MODEL EVALUATION")
    print("=" * 60)
    print(f"{'Model':<12} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'Dir. Acc.':>10}")
    print("-" * 60)
    
    for model_name, preds in results.items():
        # Align lengths (ARIMA/Prophet may have different test size)
        min_len = min(len(y_test), len(preds))
        rmse, mae, r2 = get_metrics(y_test[:min_len], preds[:min_len])
        da = directional_accuracy(y_test[:min_len], preds[:min_len])
        print(f"{model_name:<12} {rmse:>10.6f} {mae:>10.6f} {r2:>10.4f} {da:>9.2f}%")
    print("=" * 60)

    # 8. Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(y_test, label='Actual', color='black', alpha=0.5, linewidth=1)
    colors = {'LSTM': '#e74c3c', 'XGBoost': '#3498db', 'ARIMA': '#2ecc71', 'Prophet': '#9b59b6'}
    for model_name, preds in results.items():
        if model_name != 'Naive':
            min_len = min(len(y_test), len(preds))
            plt.plot(preds[:min_len], label=model_name, color=colors.get(model_name, 'gray'), alpha=0.7)
    plt.title("All Models Comparison on Test Set", fontsize=14, fontweight='bold')
    plt.xlabel("Test Sample Index")
    plt.ylabel("Scaled Log Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(r'D:\Gold Price Predictor\evaluation\comparison_plot.png', dpi=150)
    print("\nComparison plot saved to evaluation/comparison_plot.png")

if __name__ == '__main__':
    run_evaluation()
