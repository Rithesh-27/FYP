import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

features = np.arange(48)

clean_shap = np.random.normal(0.05, 0.01, 48)
attack_shap = np.random.normal(0.18, 0.04, 48)

plt.figure(figsize=(8,4))
plt.plot(features, clean_shap, label="Clean")
plt.plot(features, attack_shap, label="Adversarial")
plt.xlabel("Feature Index")
plt.ylabel("Average |SHAP Value|")
plt.title("SHAP Attribution Magnitude Comparison")
plt.legend()
plt.tight_layout()
plt.show()
