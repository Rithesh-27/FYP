import os
import sys
import torch
import torch.nn as nn
import numpy as np
import joblib
import shap
from collections import deque
from stable_baselines3 import DDPG
import time

# --- 1. DEFINE ARCHITECTURE DIRECTLY (Prevents Import Errors) ---
class AdversarialDetector(nn.Module):
    def __init__(self):
        super(AdversarialDetector, self).__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.sigmoid(self.fc(out))

# --- 2. PATH SETUP ---
# Get the absolute path to the 'Lidar' directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure the Lidar folder is in sys.path so it can find ddpg_lidar and bim_attack
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ddpg_lidar import AirSimDroneEnv
from bim_attack import bim_attack_sb3_ddpg

# Paths to your saved files
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "20260113-141301_ddpg_airsim_drone_callback", " 20260113-141301_ddpg_airsim_drone")
DETECTOR_WEIGHTS = os.path.join(BASE_DIR, "Detection", "models", "lstm_detector.pt")
SCALER_PATH = os.path.join(BASE_DIR, "Detection", "scaler.pkl")

# --- 3. LOADING & INITIALIZATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Detector
print("Loading Security Shield (LSTM)...")
detector = AdversarialDetector().to(DEVICE)
detector.load_state_dict(torch.load(DETECTOR_WEIGHTS))
detector.eval()

# Load Scaler
scaler = joblib.load(SCALER_PATH)

# Load DRL Agent
print("Loading DDPG Navigator...")
env = AirSimDroneEnv()
agent = DDPG.load(MODEL_PATH, env=env)

# Initialize SHAP
print("Initializing SHAP Explainer...")
background_data = torch.FloatTensor(np.random.rand(100, 5)).to(DEVICE)
explainer = shap.DeepExplainer(agent.policy.actor, background_data)

# Rolling memory for the last 10 steps
shap_window = deque(maxlen=10)

def get_shield_prob(obs):
    """Processes current observation through SHAP and LSTM"""
    obs_t = torch.FloatTensor(obs).to(DEVICE)
    # Get 10-D signature (5 for Velocity, 5 for Yaw)
    shaps = explainer.shap_values(obs_t)
    combined_sig = np.concatenate([shaps[0].flatten(), shaps[1].flatten()])
    
    shap_window.append(combined_sig)
    if len(shap_window) < 10:
        return 0.0
    
    # Scale and Predict
    window_data = np.array(shap_window)
    # Flatten to scale, then back to (1, 10, 10) for LSTM
    window_scaled = scaler.transform(window_data.reshape(-1, 10)).reshape(1, 10, 10)
    window_t = torch.FloatTensor(window_scaled).to(DEVICE)
    
    with torch.no_grad():
        prob = detector(window_t).item()
    return prob

# --- 4. EXECUTION LOOP ---
try:
    for ep in range(3):
        obs = env.reset()
        done = False
        shap_window.clear()
        print(f"\n>>> Starting Shielded Mission {ep+1}")
        
        while not done:
            # A. THE ATTACK (Simulating the BIM attack)
            obs_attacked, _ = bim_attack_sb3_ddpg(agent, obs, epsilon=0.02)
            
            # B. THE DEFENSE
            prob = get_shield_prob(obs_attacked)
            
            # C. MITIGATION
            if prob > 0.6: # Threshold for detection
                print(f"[!] ALERT: Attack Probability {prob:.2f} | Mitigation: IGNORE LIDAR")
                # Mitigation strategy: Nullify the Lidar inputs (the first 3 features)
                # This forces the drone to rely on Goal Dist and Relative Yaw only.
                safe_obs = obs_attacked.copy()
                safe_obs[0][0:3] = 1.0 # Set front/left/right distances to 'max/clear'
                action, _ = agent.predict(safe_obs, deterministic=True)
            else:
                action, _ = agent.predict(obs_attacked, deterministic=True)
            
            # D. STEP
            obs, reward, done, info = env.step(action)
            
            if done:
                res = "SUCCESS" if info.get('is_success') else "CRASHED"
                print(f"Episode Finished: {res}")

except KeyboardInterrupt:
    print("Evaluation stopped.")
finally:
    env.close()