import matplotlib.pyplot as plt

metrics = ["TPR", "F1 Score"]
values = [89.1, 86.0]

plt.figure(figsize=(6,4))
plt.bar(metrics, values)
plt.ylabel("Percentage (%)")
plt.title("Adversarial Detection Performance")
plt.ylim(0,100)
plt.tight_layout()
plt.show()
