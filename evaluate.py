import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap

# Import the model class directly from train_lstm.py
from train_lstm import GoldPriceLSTM

# ===================== 1. Load Data =====================
X_train, X_test = np.load(r'D:\Gold Price Predictor\Models\X_train.npy'), np.load(r'D:\Gold Price Predictor\Models\X_test.npy')
y_train, y_test = np.load(r'D:\Gold Price Predictor\Models\y_train.npy'), np.load(r'D:\Gold Price Predictor\Models\y_test.npy')

# ===================== 2. Feature Names =====================
LOOKBACK = 30
FEATURE_NAMES = ['Gold Log Return', 'Rate Log Return', 'Inflation', 'Wedding Flag']

# For flattened features (XGBoost / SHAP), create Day_X_FeatureName labels
flat_feature_names = []
for day in range(1, LOOKBACK + 1):
    for feat in FEATURE_NAMES:
        flat_feature_names.append(f'Day {day} - {feat}')

# ===================== 3. LSTM Predictions =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GoldPriceLSTM().to(device)
model.load_state_dict(torch.load(r'D:\Gold Price Predictor\Models\Mogold_lstm_model.pth', map_location=device))
model.eval()
with torch.no_grad():
    lstm_preds = model(torch.FloatTensor(X_test).to(device)).cpu().numpy().flatten()

# ===================== 4. Baselines =====================
# Naive Baseline: predict 0 (no change in log returns)
naive_preds = np.zeros_like(y_test)

# XGBoost Baseline
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train_flat, y_train)
xgb_preds = xgb_model.predict(X_test_flat)

# ===================== 5. Metrics =====================
def get_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def directional_accuracy(actual, pred):
    """What % of the time does the model correctly predict the direction (up/down)?"""
    actual_dir = np.sign(actual)
    pred_dir = np.sign(pred)
    return np.mean(actual_dir == pred_dir) * 100

lstm_rmse, lstm_mae, lstm_r2 = get_metrics(y_test, lstm_preds)
xgb_rmse, xgb_mae, xgb_r2 = get_metrics(y_test, xgb_preds)
naive_rmse, naive_mae, naive_r2 = get_metrics(y_test, naive_preds)

lstm_da = directional_accuracy(y_test, lstm_preds)
xgb_da = directional_accuracy(y_test, xgb_preds)
naive_da = directional_accuracy(y_test, naive_preds)

print("=" * 60)
print("              MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"{'Model':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Dir. Acc.':>10}")
print("-" * 60)
print(f"{'LSTM':<12} {lstm_rmse:>10.6f} {lstm_mae:>10.6f} {lstm_r2:>10.4f} {lstm_da:>9.2f}%")
print(f"{'XGBoost':<12} {xgb_rmse:>10.6f} {xgb_mae:>10.6f} {xgb_r2:>10.4f} {xgb_da:>9.2f}%")
print(f"{'Naive':<12} {naive_rmse:>10.6f} {naive_mae:>10.6f} {naive_r2:>10.4f} {naive_da:>9.2f}%")
print("=" * 60)

# ===================== 6. Visualizations =====================
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Gold Price LSTM - Model Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)

# --- Plot 1: Actual vs Predicted (Time Series) ---
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(y_test, label='Actual', color='#2ecc71', linewidth=1.5, alpha=0.8)
ax1.plot(lstm_preds, label='LSTM Predicted', color='#e74c3c', linewidth=1.2, alpha=0.8)
ax1.plot(xgb_preds, label='XGBoost Predicted', color='#3498db', linewidth=1.0, alpha=0.6, linestyle='--')
ax1.set_title('Actual vs Predicted Gold Log Returns', fontsize=13, fontweight='bold')
ax1.set_xlabel('Test Sample Index')
ax1.set_ylabel('Gold Log Return')
ax1.legend(fontsize=9)

# --- Plot 2: Prediction Residuals (LSTM) ---
ax2 = fig.add_subplot(2, 2, 2)
residuals = y_test - lstm_preds
ax2.scatter(range(len(residuals)), residuals, alpha=0.5, s=10, color='#9b59b6')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax2.set_title('LSTM Prediction Residuals', fontsize=13, fontweight='bold')
ax2.set_xlabel('Test Sample Index')
ax2.set_ylabel('Residual (Actual - Predicted)')

# --- Plot 3: Error Distribution ---
ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(residuals, bins=40, color='#1abc9c', edgecolor='black', alpha=0.75, label='LSTM Residuals')
xgb_residuals = y_test - xgb_preds
ax3.hist(xgb_residuals, bins=40, color='#e67e22', edgecolor='black', alpha=0.5, label='XGBoost Residuals')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax3.set_title('Prediction Error Distribution', fontsize=13, fontweight='bold')
ax3.set_xlabel('Error (Actual - Predicted)')
ax3.set_ylabel('Frequency')
ax3.legend(fontsize=9)

# --- Plot 4: Model Comparison Bar Chart (RMSE, MAE, Directional Accuracy) ---
ax4 = fig.add_subplot(2, 2, 4)
models = ['LSTM', 'XGBoost', 'Naive']
x_pos = np.arange(len(models))
width = 0.25

rmse_vals = [lstm_rmse, xgb_rmse, naive_rmse]
mae_vals = [lstm_mae, xgb_mae, naive_mae]
da_vals = [lstm_da / 100, xgb_da / 100, naive_da / 100]  # Normalize for scale

bars1 = ax4.bar(x_pos - width, rmse_vals, width, label='RMSE', color='#e74c3c', alpha=0.8)
bars2 = ax4.bar(x_pos, mae_vals, width, label='MAE', color='#3498db', alpha=0.8)
bars3 = ax4.bar(x_pos + width, da_vals, width, label='Dir. Accuracy (×0.01)', color='#2ecc71', alpha=0.8)

ax4.set_title('Model Comparison', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models)
ax4.legend(fontsize=9)
ax4.set_ylabel('Score')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('evaluation_dashboard.png', dpi=150, bbox_inches='tight')
print("\nDashboard saved as evaluation_dashboard.png")
plt.show()

# ===================== 7. SHAP Interpretability =====================
print("\nGenerating SHAP analysis...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_flat)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_flat, feature_names=flat_feature_names, max_display=20, show=False)
plt.title('SHAP Feature Importance (Top 20 Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_analysis.png', dpi=150, bbox_inches='tight')
print("SHAP plot saved as shap_analysis.png")
plt.show()

print("\nEvaluation complete!")