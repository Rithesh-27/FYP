import torch
import numpy as np
import joblib
import shap
import os
from collections import deque
from stable_baselines3 import DDPG
from ddpg_lidar import AirSimDroneEnv
from bim_attack import bim_attack_sb3_ddpg

# --- SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "20260113-141301_ddpg_airsim_drone_callback", " 20260113-141301_ddpg_airsim_drone")
DETECTOR_PATH = os.path.join(BASE_DIR, "Detection", "models", "lstm_detector.pt")
SCALER_PATH = os.path.join(BASE_DIR, "Detection", "scaler.pkl")

# Re-define architecture
class AdversarialDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(10, 64, batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.sigmoid(self.fc(self.dropout(hn[-1])))

# Load models
scaler = joblib.load(SCALER_PATH)
detector = AdversarialDetector().to(DEVICE)
detector.load_state_dict(torch.load(DETECTOR_PATH))
detector.eval()
env = AirSimDroneEnv()
agent = DDPG.load(MODEL_PATH, env=env)
explainer = shap.DeepExplainer(agent.policy.actor, torch.FloatTensor(np.random.rand(100, 5)).to(DEVICE))

# --- ANALYTICS LOOP ---
detection_times = []
num_test_episodes = 5

print(f"Starting Performance Analytics over {num_test_episodes} episodes...")

for ep in range(num_test_episodes):
    obs = env.reset()
    shap_window = deque(maxlen=10)
    attack_step_start = 0 # In this test, attack starts at step 0
    detected_at_step = None
    
    for step in range(50): # We only need the first 50 steps to measure latency
        # 1. Apply Attack
        obs_attacked, _ = bim_attack_sb3_ddpg(agent, obs, epsilon=0.02)
        
        # 2. SHAP + LSTM
        obs_t = torch.FloatTensor(obs_attacked).to(DEVICE)
        shaps = explainer.shap_values(obs_t)
        combined_sig = np.concatenate([shaps[0].flatten(), shaps[1].flatten()])
        shap_window.append(combined_sig)
        
        if len(shap_window) == 10:
            window_scaled = scaler.transform(np.array(shap_window).reshape(-1, 10)).reshape(1, 10, 10)
            with torch.no_grad():
                prob = detector(torch.FloatTensor(window_scaled).to(DEVICE)).item()
            
            if prob > 0.7 and detected_at_step is None:
                detected_at_step = step
                break
        
        # 3. Step
        action, _ = agent.predict(obs_attacked, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done: break

    if detected_at_step is not None:
        latency = detected_at_step - attack_step_start
        detection_times.append(latency)
        print(f"Episode {ep+1}: Attack Detected at step {detected_at_step} (Latency: {latency} steps)")
    else:
        print(f"Episode {ep+1}: Attack NOT detected!")

# --- FINAL RESULTS ---
avg_latency = np.mean(detection_times)
# Assuming 1 step in AirSim = 0.1 seconds (from your self.dt)
time_seconds = avg_latency * 0.1 

print("\n" + "="*30)
print("FINAL PERFORMANCE METRICS")
print("="*30)
print(f"Average Detection Latency: {avg_latency:.1f} steps")
print(f"Average Real-World Time:    {time_seconds:.2f} seconds")
print(f"Detection Success Rate:     {(len(detection_times)/num_test_episodes)*100}%")
print("="*30)