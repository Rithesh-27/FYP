import matplotlib.pyplot as plt

labels = ["Clean", "Adversarial"]
success_rates = [92.4, 61.8]

plt.figure(figsize=(6,4))
plt.bar(labels, success_rates)
plt.ylabel("Success Rate (%)")
plt.title("Navigation Success Rate Under Clean and Adversarial Conditions")
plt.ylim(0,100)
plt.tight_layout()
plt.show()
