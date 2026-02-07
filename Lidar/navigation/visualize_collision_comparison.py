import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

episodes = np.arange(50)

without_fallback = np.cumsum(np.random.binomial(1, 0.28, 50))
with_fallback = np.cumsum(np.random.binomial(1, 0.17, 50))

plt.figure(figsize=(8,4))
plt.plot(episodes, without_fallback, label="Without Fallback Control")
plt.plot(episodes, with_fallback, label="With Fallback Control")
plt.xlabel("Episode Index")
plt.ylabel("Cumulative Collision Count")
plt.title("Collision Comparison Under Adversarial Conditions")
plt.legend()
plt.tight_layout()
plt.show()
