import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Model Architecture ---
class GoldPriceLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2):
        super(GoldPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]           # Use the last time step
        out = self.batch_norm(out)     # Normalize activations
        out = self.dropout(out)        # Regularization
        return self.fc(out)

# --- 2. Training Logic ---
def train_lstm(epochs=200, batch_size=32, lr=0.001, patience=15):
    # Load the Saved Tensors
    X_train = np.load(r'D:\Gold Price Predictor\Models\X_train.npy')
    y_train = np.load(r'D:\Gold Price Predictor\Models\y_train.npy')
    X_val = np.load(r'D:\Gold Price Predictor\Models\X_val.npy')
    y_val = np.load(r'D:\Gold Price Predictor\Models\y_val.npy')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    
    model = GoldPriceLSTM(input_size=X_train.shape[2]).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Starting LSTM training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]  Train Loss: {train_loss/len(train_loader):.6f}  Val Loss: {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model_path = r'D:\Gold Price Predictor\Models\lstm_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"LSTM Model saved to {model_path}")

if __name__ == '__main__':
    train_lstm()
