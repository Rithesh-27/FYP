from fgsm_attack import fgsm_attack_sb3_ddpg
import numpy as np
import matplotlib.pyplot as plt
import os

save_dir = "attack_graphs"
os.makedirs(save_dir, exist_ok=True)

def get_lidar_depth_vector(client, num_rays=64, max_range=50.0):
    """
    Converts AirSim LiDAR point cloud into a 1D depth vector.
    """

    lidar = client.getLidarData()

    if len(lidar.point_cloud) < 3:
        return np.ones(num_rays) * max_range

    pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)

    # Compute angles and distances
    xy = pts[:, :2]
    distances = np.linalg.norm(xy, axis=1)
    angles = np.arctan2(xy[:,1], xy[:,0])  # [-pi, pi]

    # Bin angles into rays
    bins = np.linspace(-np.pi, np.pi, num_rays + 1)
    depth = np.ones(num_rays) * max_range

    for i in range(num_rays):
        mask = (angles >= bins[i]) & (angles < bins[i+1])
        if np.any(mask):
            depth[i] = np.min(distances[mask])

    return depth

def apply_lidar_fgsm(depth, epsilon=0.05):
    """
    FGSM-style signed perturbation on LiDAR depth.
    """
    sign_noise = np.sign(np.random.randn(*depth.shape))
    perturbation = epsilon * sign_noise
    attacked = np.clip(depth + perturbation, 0.0, None)
    return attacked, perturbation

def visualize_real_lidar_attack(
    client,
    step,
    num_rays=64,
    epsilon=0.05,
):
    clean = get_lidar_depth_vector(client, num_rays=num_rays)
    attacked, perturbation = apply_lidar_fgsm(clean, epsilon)

    x = np.arange(num_rays)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3), sharey=True)

    # Clean
    axes[0].vlines(x, ymin=0, ymax=clean, linewidth=1)
    axes[0].set_title("Clean LiDAR")

    # Perturbation
    axes[1].vlines(x, ymin=0, ymax=perturbation, linewidth=1)
    axes[1].set_title("Perturbation")

    # Attacked
    axes[2].vlines(x, ymin=0, ymax=attacked, linewidth=1)
    axes[2].set_title("Attacked LiDAR")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(0.335, 0.5, "+", fontsize=28, ha="center", va="center")
    fig.text(0.665, 0.5, "→", fontsize=28, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"timestep{step+1}.png"))
    plt.close()

def run_attacked_navigation(env, model, num_episodes, log_file, attack_start=10):
    rewards, collisions, successes = [], [], []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        collided = 0
        success = True
        step = 0

        while not done:
            original_obs = obs.copy()
            if step >= attack_start:
                obs, perturbation = fgsm_attack_sb3_ddpg(model, obs, epsilon=0.02)

            if ep == 0 and step % 10 == 0:
                visualize_real_lidar_attack(env.drone, step=step)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward

            if info.get("is_crash", False):
                collided += 1
                success = False
            
            step += 1

        rewards.append(total_reward)
        collisions.append(collided)
        successes.append(success)

    return rewards, collisions, successes
