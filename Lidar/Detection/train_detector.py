import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # To save the scaler for real-time use
import os

# --- 1. DATA PREPARATION ---
DATA_DIR = "dataset"
MODEL_SAVE_PATH = "models/lstm_detector.pt"
SCALER_PATH = "scaler.pkl"
os.makedirs("models", exist_ok=True)

print("Loading SHAP dataset...")
X_clean = np.load(f"{DATA_DIR}/clean_X.npy")
y_clean = np.load(f"{DATA_DIR}/clean_y.npy")
X_attacked = np.load(f"{DATA_DIR}/attacked_X.npy")
y_attacked = np.load(f"{DATA_DIR}/attacked_y.npy")

# Combine datasets
X = np.concatenate([X_clean, X_attacked])
y = np.concatenate([y_clean, y_attacked])

# Functional Concern: Standardization
# We flatten to (Samples*10, 10) to scale across all time-steps, then reshape back
X_flat = X.reshape(-1, 10)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat).reshape(-1, 10, 10)

# Performance: Save the scaler! Essential for real-time inference
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")

# Split into Training (80%) and Validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create PyTorch DataLoaders
train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# --- 2. LSTM ARCHITECTURE (The Detector) ---
class AdversarialDetector(nn.Module):
    def __init__(self):
        super(AdversarialDetector, self).__init__()
        # 10 features, 64 memory units
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1) # Binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Time, Features)
        _, (hn, _) = self.lstm(x)
        # We take the final hidden state (hn) representing the 10-step window
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return self.sigmoid(out)

# if __name__ == "__main__":
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     DATA_DIR = os.path.join(BASE_DIR, "dataset")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = AdversarialDetector().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(detector.parameters(), lr=0.001)

# --- 3. TRAINING LOOP ---
print(f"Starting training on {len(X_train)} samples...")
for epoch in range(30):
    detector.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = detector(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/30] - Avg Loss: {total_loss/len(train_loader):.4f}")

# --- 4. VALIDATION & PERFORMANCE CHECK ---
detector.eval()
with torch.no_grad():
    val_X_t = torch.FloatTensor(X_val).to(device)
    val_y_t = torch.FloatTensor(y_val).to(device)
    preds = detector(val_X_t).squeeze()
    
    # Binary threshold at 0.5
    correct_predictions = (preds > 0.5).float() == val_y_t
    acc = correct_predictions.float().mean().item() 
    print(f"\nFINAL DETECTION ACCURACY: {acc*100:.2f}%")

# --- 5. SAVE MODEL ---
torch.save(detector.state_dict(), MODEL_SAVE_PATH)
print(f"Detector saved to {MODEL_SAVE_PATH}")
