import torch
import numpy as np

def bim_attack_sb3_ddpg(
    model,
    obs,
    epsilon=0.02,     # total perturbation budget
    alpha=0.005,      # step size
    num_steps=5,      # number of BIM iterations
    clip_min=None,
    clip_max=None,
):
    """
    BIM (Iterative FGSM) attack for SB3 DDPG using critic gradients
    """

    device = model.device
    policy = model.policy
    actor = policy.actor
    critic = policy.critic

    actor.eval()
    critic.eval()

    # Original observation
    obs_orig = torch.tensor(
        obs,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    # Initialize adversarial observation
    obs_adv = obs_orig.clone().detach()

    for _ in range(num_steps):

        obs_adv.requires_grad_(True)

        # Forward
        action = actor(obs_adv)
        q_vals = critic(obs_adv, action)

        # SB3 critic returns tuple (Q1, Q2)
        q_val = q_vals[0] if isinstance(q_vals, tuple) else q_vals

        # Loss: minimize Q-value
        loss = -q_val.mean()

        actor.zero_grad()
        critic.zero_grad()
        loss.backward()

        # BIM update
        grad = obs_adv.grad
        obs_adv = obs_adv + alpha * grad.sign()

        # Project back to epsilon-ball
        delta = torch.clamp(
            obs_adv - obs_orig,
            min=-epsilon,
            max=epsilon
        )
        obs_adv = obs_orig + delta

        # Optional observation bounds
        if clip_min is not None and clip_max is not None:
            obs_adv = torch.clamp(obs_adv, clip_min, clip_max)

        obs_adv = obs_adv.detach()

    return (
        obs_adv.cpu().numpy()[0],
        (obs_adv - obs_orig).cpu().numpy()[0]
    )
