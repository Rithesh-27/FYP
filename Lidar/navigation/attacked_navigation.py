from fgsm_attack import fgsm_attack_sb3_ddpg

def run_attacked_navigation(env, model, num_episodes, log_file):
    rewards, collisions, successes = [], [], []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        collided = 0
        success = True

        while not done:
            obs, _ = fgsm_attack_sb3_ddpg(model, obs, epsilon=0.02)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward

            if info.get("is_crash", False):
                collided += 1
                success = False

        rewards.append(total_reward)
        collisions.append(collided)
        successes.append(success)

    return rewards, collisions, successes
