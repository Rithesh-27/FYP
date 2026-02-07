import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

timesteps = np.arange(60)

clean_drift = np.random.normal(0.02, 0.005, 60)
attack_drift = np.random.normal(0.15, 0.03, 60)

plt.figure(figsize=(8,4))
plt.plot(timesteps, clean_drift, label="Clean")
plt.plot(timesteps, attack_drift, label="Adversarial")
plt.xlabel("Time Step")
plt.ylabel("Mean SHAP Drift")
plt.title("Temporal Drift in SHAP Explanations")
plt.legend()
plt.tight_layout()
plt.show()
