import numpy as np

def compute_metrics(episode_rewards, collisions, successes):
    return {
        "avg_reward": np.mean(episode_rewards),
        "success_rate": np.mean(successes) * 100,
        "collision_rate": np.mean(collisions),
    }
