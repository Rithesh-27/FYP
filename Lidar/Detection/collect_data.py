import sys
import os

# 1. Get the absolute path to the 'Lidar' directory
# Since this script is in 'Lidar/Detection', we go up one level
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LIDAR_DIR = os.path.dirname(CURRENT_DIR)

# 2. Add 'Lidar' to the Python path
if LIDAR_DIR not in sys.path:
    sys.path.append(LIDAR_DIR)

print(f"1. Script Started in: {CURRENT_DIR}")
print(f"2. Root Lidar Directory added: {LIDAR_DIR}")

import torch
import shap
import numpy as np
import tqdm
import time
from stable_baselines3 import DDPG

# Now these imports will find the files and the dummy setup_path.py
from ddpg_lidar import AirSimDroneEnv
from bim_attack import bim_attack_sb3_ddpg

print("3. Imports successful. Connecting to AirSim...")
# ... rest of the script

# --- 1. CONFIGURATION (Functional & Performance) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"..\models\20260113-141301_ddpg_airsim_drone_callback\ 20260113-141301_ddpg_airsim_drone.zip"
SAVE_DIR = "Detection/dataset"
WINDOW_SIZE = 10  # Temporal window for LSTM
EPISODES_PER_CLASS = 20 # 20 Clean, 20 Attacked
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. INITIALIZE ENVIRONMENT & MODEL ---
env = AirSimDroneEnv()
model = DDPG.load(MODEL_PATH, env=env)
actor = model.policy.actor.to(DEVICE)
actor.eval()

# --- 3. SHAP WRAPPER (Double Signature: Velocity + Yaw) ---
# We explain the whole action vector [v_x, yaw_rate]
def predict_fn(obs_tensor):
    obs_tensor = torch.FloatTensor(obs_tensor).to(DEVICE)
    return actor(obs_tensor)

# Performance: Use a representative background (100 samples)
print("Creating SHAP background baseline...")
background_data = torch.FloatTensor(np.random.rand(100, 5)).to(DEVICE)
explainer = shap.DeepExplainer(actor, background_data)

def collect_shaps_for_class(is_attacked=False):
    type_str = "attacked" if is_attacked else "clean"
    all_windows = []
    
    for ep in tqdm.tqdm(range(EPISODES_PER_CLASS), desc=f"Running {type_str} episodes"):
        obs = env.reset()
        done = False
        episode_raw_shaps = []

        while not done:
            # A. HANDLE ATTACK
            current_obs = obs
            if is_attacked:
                # Performance Note: BIM is also heavy. This step is the latency bottleneck.
                current_obs, _ = bim_attack_sb3_ddpg(model, obs, epsilon=0.02)
            
            # B. CALCULATE SHAP (The 'Reasoning' Vector)
            obs_input = torch.FloatTensor(current_obs).to(DEVICE)
            
            # Non-functional Concern: Catch potential Explainer errors
            try:
                # shap_values is a list of 2 arrays: [SHAP_Vel, SHAP_Yaw]
                shaps = explainer.shap_values(obs_input) 
                
                # Combine: [5 vel features] + [5 yaw features] = 10-D signature
                combined_sig = np.concatenate([shaps[0].flatten(), shaps[1].flatten()])
                episode_raw_shaps.append(combined_sig)
            except Exception as e:
                print(f"SHAP Error: {e}")
                break

            # C. EXECUTE STEP
            action, _ = model.predict(current_obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        # D. WINDOWING LOGIC (Functional: Transform episode into Packets)
        # We turn the whole flight into overlapping 10-step windows
        if len(episode_raw_shaps) >= WINDOW_SIZE:
            for i in range(len(episode_raw_shaps) - WINDOW_SIZE + 1):
                window = episode_raw_shaps[i : i + WINDOW_SIZE]
                all_windows.append(window)

    # E. SAVE DATA
    final_data = np.array(all_windows)
    np.save(f"{SAVE_DIR}/{type_str}_X.npy", final_data)
    # Create labels (0 for clean, 1 for attacked)
    labels = np.full(len(final_data), (1 if is_attacked else 0))
    np.save(f"{SAVE_DIR}/{type_str}_y.npy", labels)

# --- 4. EXECUTION ---
start_time = time.time()

collect_shaps_for_class(is_attacked=False)
collect_shaps_for_class(is_attacked=True)

total_time = (time.time() - start_time) / 60
print(f"Data Collection Complete in {total_time:.2f} minutes.")
print(f"Files saved to {SAVE_DIR}")