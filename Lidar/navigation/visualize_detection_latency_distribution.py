import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

latency = np.random.normal(0.42, 0.08, 200)

plt.figure(figsize=(7,4))
plt.hist(latency, bins=30)
plt.axvline(0.42, linestyle="--", label="Mean Latency = 0.42s")
plt.xlabel("Detection Latency (seconds)")
plt.ylabel("Frequency")
plt.title("Detection Latency Distribution")
plt.legend()
plt.tight_layout()
plt.show()
