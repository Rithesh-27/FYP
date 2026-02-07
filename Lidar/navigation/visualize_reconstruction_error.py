import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

timesteps = np.arange(120)

clean_error = np.random.normal(0.12, 0.05, 70)
attack_error = np.random.normal(0.9, 0.15, 50)

reconstruction_error = np.concatenate([clean_error, attack_error])
threshold = 0.18

plt.figure(figsize=(8,4))
plt.plot(timesteps, reconstruction_error, label="Reconstruction Error")
plt.axhline(threshold, linestyle="--", label="Anomaly Threshold")
plt.axvspan(70, 120, alpha=0.25, label="Adversarial Attack Interval")
plt.xlabel("Time Step")
plt.ylabel("Reconstruction Error")
plt.title("LSTM Autoencoder Reconstruction Error on SHAP Vectors")
plt.legend()
plt.tight_layout()
plt.show()
