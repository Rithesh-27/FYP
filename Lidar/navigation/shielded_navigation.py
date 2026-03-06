from bim_attack import bim_attack_sb3_ddpg
import numpy as np
import matplotlib.pyplot as plt
import os
from navigation.normal_navigation import run_normal_navigation
os.makedirs("shap_values", exist_ok=True)
os.makedirs("detection_graphs", exist_ok=True)

def plot_shap_values(shap_vals, step):
    """
    Visualize SHAP values for actor outputs at a timestep.
    """
    shap_arr = np.concatenate([
        shap_vals[0].flatten(),
        shap_vals[1].flatten()
    ])

    plt.figure(figsize=(10, 4))
    plt.plot(shap_arr)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.title(f"SHAP Values (Step {step})")
    plt.xlabel("Feature Index")
    plt.ylabel("SHAP Contribution")
    plt.tight_layout()

    plt.savefig(f"shap_values/shap_step{step}.png", dpi=200)
    plt.close()

def plot_anomaly_scores(anomaly_scores, threshold, ep):
    plt.figure(figsize=(10, 4))

    plt.plot(anomaly_scores, label="Anomaly Score")
    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Detection Threshold"
    )

    plt.xlabel("Timestep")
    plt.ylabel("Anomaly Probability")
    plt.title(f"Attack Detection Signal – Episode {ep+1}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"detection_graphs/episode{ep+1}_anomaly.png",dpi=200)
    plt.close()


def run_shielded_navigation(env, agent, shield, log_file, episodes=5, threshold=0.6, attack_start=20):
    rewards, collisions, successes = [], [], []
    all_anomaly_scores = []   # per episode
    all_detection_flags = []  # per episode

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        crashed = False

        anomaly_scores = []
        detection_flags = []
        detection_announced = False

        print(f"\n🛡️ Shielded Episode {ep+1}")

        step_idx = 0
        while not done:
            # 1. Attack
            obs_attacked = obs
            if step_idx >= attack_start:
                obs_attacked, _ = bim_attack_sb3_ddpg(agent, obs, epsilon=0.02)

            # 2. Detection
            prob, shap_vals = shield.detect(obs_attacked,return_shap=True)
            anomaly_scores.append(prob)

            detected = prob > threshold
            detection_flags.append(detected)

            # ---- CLEAR INDICATOR (ONCE) ----
            if detected and step_idx % 5 == 0 and not detection_announced:
                print("\n" + "="*55)
                print(f"🚨 ADVERSARIAL ATTACK DETECTED at step {step_idx}")
                print(f"📈 Anomaly score: {prob:.3f} (threshold = {threshold})")
                print("🛡️ Mitigation ENABLED")
                print("="*55 + "\n")
                plot_shap_values(shap_vals, step_idx)
                detection_announced = True

                success = run_normal_navigation(env, log_file,start_from_current_state=True)

                crashed = not success

                break

            # 3. Mitigation
            if detected:
                obs_safe = shield.mitigate(obs_attacked)
                action, _ = agent.predict(obs_safe, deterministic=True)
            else:
                action, _ = agent.predict(obs_attacked, deterministic=True)

            # 4. Environment step
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if info.get("is_crash", False):
                crashed = True

            step_idx += 1

        rewards.append(total_reward)
        collisions.append(1 if crashed else 0)
        successes.append(not crashed)

        all_anomaly_scores.append(anomaly_scores)
        all_detection_flags.append(detection_flags)

        status = "SUCCESS" if not crashed else "CRASH"
        plot_anomaly_scores(anomaly_scores, threshold, ep)
        print(f"Episode {ep+1} finished → {status}")

    return (
        rewards,
        collisions,
        successes,
        all_anomaly_scores,
        all_detection_flags
    )
