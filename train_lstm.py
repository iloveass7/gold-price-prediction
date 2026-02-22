import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 1. Load the Saved Tensors
X_train = np.load(r'D:\Gold Price Predictor\Models\X_train.npy')
y_train = np.load(r'D:\Gold Price Predictor\Models\y_train.npy')
X_test = np.load(r'D:\Gold Price Predictor\Models\X_test.npy')
y_test = np.load(r'D:\Gold Price Predictor\Models\y_test.npy')

# Convert to PyTorch Tensors and move to GPU (3060 Ti)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

# Create DataLoader for batch processing
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)

# 2. Define the Stacked LSTM Architecture [cite: 51, 99]
class GoldPriceLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=512, num_layers=4):
        super(GoldPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Use the last time step for prediction

if __name__ == '__main__':
    model = GoldPriceLSTM().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 3. Training Loop 
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation Check every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                val_loss = criterion(test_outputs, y_test_t)
                # Calculate MAE as per project objectives 
                mae = torch.mean(torch.abs(test_outputs - y_test_t))
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, MAE: {mae.item():.6f}')

    # 4. Save the Model Weights
    torch.save(model.state_dict(), r'D:\Gold Price Predictor\Models\Mogold_lstm_model.pth')
    print("Model trained and saved as gold_lstm_model.pth")