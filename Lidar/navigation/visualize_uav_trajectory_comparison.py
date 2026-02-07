import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

timesteps = 120
t = np.linspace(0, 10, timesteps)

clean_x = t
clean_y = np.sin(t)

attack_x = t
attack_y = np.sin(t) + np.random.normal(0, 0.45, timesteps)

plt.figure(figsize=(6,6))
plt.plot(clean_x, clean_y, label="Clean Trajectory", linewidth=2)
plt.plot(attack_x, attack_y, label="Adversarial Trajectory", linewidth=2)

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("UAV Trajectory Comparison: Clean vs Adversarial")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
