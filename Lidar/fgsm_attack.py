import torch

def fgsm_attack_sb3_ddpg(
    model,
    obs,
    epsilon=0.01
):
    """
    FGSM attack using CRITIC loss (robust for SB3 DDPG)
    """

    device = model.device
    policy = model.policy
    actor = policy.actor
    critic = policy.critic

    actor.eval()
    critic.eval()

    # Observation tensor
    obs_t = torch.tensor(
        obs,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    # Enable gradient
    obs_adv = obs_t.clone().detach()
    obs_adv.requires_grad_(True)

    # Forward pass
    action = actor(obs_adv)
    q_vals = critic(obs_adv, action)

    # q_vals is a tuple: (q1, q2)
    if isinstance(q_vals, tuple):
        q_val = q_vals[0]   # use Q1
    else:
        q_val = q_vals
        
    # LOSS: minimize Q-value
    loss = -q_val.mean()

    # Backprop
    actor.zero_grad()
    critic.zero_grad()
    loss.backward()

    # FGSM step
    grad = obs_adv.grad

    if grad is None:
        raise RuntimeError("Gradient is None — attack failed")

    perturbation = epsilon * grad.sign()
    obs_adv = obs_adv + perturbation

    return (
        obs_adv.detach().cpu().numpy()[0],
        perturbation.detach().cpu().numpy()[0]
    )