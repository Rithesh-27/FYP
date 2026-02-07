from eval_ddpg_lidar import AirSimDroneEnv
from stable_baselines3 import DDPG
from datetime import datetime
import os

from navigation.logger import init_logger
from navigation.metrics import compute_metrics
from navigation.normal_navigation import run_normal_navigation
from navigation.attacked_navigation import run_attacked_navigation
from navigation.visualize import plot_comparison
from navigation.visualize_anomaly import plot_anomaly_score
from defense.shield import SecurityShield
from navigation.shielded_navigation import run_shielded_navigation
import torch

env = AirSimDroneEnv()

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
NORMAL_LOG = f"Logs/normal_{RUN_ID}.csv"
ATTACK_LOG = f"Logs/attack_{RUN_ID}.csv"
script_path = os.path.abspath(__file__)
base_dir = os.path.dirname(script_path)
project_dir = os.path.dirname(base_dir)

init_logger(NORMAL_LOG)
init_logger(ATTACK_LOG)

# ---------------- NORMAL NAVIGATION ----------------
print("\n🚀 Running nominal navigation demo...\n")
normal_success = run_normal_navigation(env, NORMAL_LOG)

normal_metrics = {
    "success_rate": 100.0 if normal_success else 0.0,
    "avg_reward": 0.0
}

# ---------------- ATTACKED NAVIGATION ----------------
print("\n🚨 Running attacked RL navigation...\n")

model = DDPG.load(project_dir + "/models/20260113-141301_ddpg_airsim_drone_callback/ 20260113-141301_ddpg_airsim_drone.zip", env=env)

rewards, collisions, successes = run_attacked_navigation(
    env, model, num_episodes=1, log_file=ATTACK_LOG
)

attacked_metrics = compute_metrics(rewards, collisions, successes)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD AGENT ----------------
agent = DDPG.load(project_dir + "/models/20260113-141301_ddpg_airsim_drone_callback/ 20260113-141301_ddpg_airsim_drone.zip", env=env)

# ---------------- LOAD SHIELD ----------------
shield = SecurityShield(
    agent=agent,
    detector_path= project_dir + "/Lidar/Detection/models/lstm_detector.pt",
    scaler_path= project_dir + "/Lidar/Detection/scaler.pkl",
    device=DEVICE
)

# ---------------- SHIELDED NAVIGATION ----------------
print("\n🛡️ Running SHIELDED navigation (Attack + Detection + Mitigation)\n")

(   shield_rewards,
    shield_collisions,
    shield_successes,
    anomaly_scores,
    detection_flags
) = run_shielded_navigation(
    env, agent, shield,
    log_file="Logs/shielded.csv",
    episodes=5
)

plot_anomaly_score(anomaly_scores, detection_flags, threshold=0.6, episode_idx=0)

shield_metrics = compute_metrics(
    shield_rewards,
    shield_collisions,
    shield_successes
)


# ---------------- VISUALIZATION ----------------
plot_comparison(normal_metrics, attacked_metrics)

print("\n📊 FINAL COMPARISON")
print("Nominal:", normal_metrics)
print("Attacked:", attacked_metrics)
print("Shielded:", shield_metrics)

