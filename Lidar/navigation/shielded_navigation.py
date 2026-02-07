from bim_attack import bim_attack_sb3_ddpg
import numpy as np

def run_shielded_navigation(env, agent, shield, log_file, episodes=5, threshold=0.6):
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
            obs_attacked, _ = bim_attack_sb3_ddpg(agent, obs, epsilon=0.02)

            # 2. Detection
            prob = shield.detect(obs_attacked)
            anomaly_scores.append(prob)

            detected = prob > threshold
            detection_flags.append(detected)

            # ---- CLEAR INDICATOR (ONCE) ----
            if detected and not detection_announced:
                print("\n" + "="*55)
                print(f"🚨 ADVERSARIAL ATTACK DETECTED at step {step_idx}")
                print(f"📈 Anomaly score: {prob:.3f} (threshold = {threshold})")
                print("🛡️ Mitigation ENABLED")
                print("="*55 + "\n")
                detection_announced = True

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
        print(f"Episode {ep+1} finished → {status}")

    return (
        rewards,
        collisions,
        successes,
        all_anomaly_scores,
        all_detection_flags
    )
