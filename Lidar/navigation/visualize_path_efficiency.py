import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

episodes = np.arange(1, 21)
path_efficiency = np.random.normal(0.87, 0.03, 20)

plt.figure(figsize=(7,4))
plt.plot(episodes, path_efficiency, marker='o')
plt.axhline(0.87, linestyle='--', label="Average PE = 0.87")
plt.xlabel("Episode")
plt.ylabel("Path Efficiency")
plt.title("Path Efficiency Across Navigation Episodes")
plt.legend()
plt.tight_layout()
plt.show()
