import matplotlib.pyplot as plt

epsilon = [0.005, 0.01, 0.02, 0.04]
detection_rate = [62, 75, 89, 94]

plt.figure(figsize=(6,4))
plt.plot(epsilon, detection_rate, marker='o')
plt.xlabel("Attack Strength (ε)")
plt.ylabel("Detection Rate (%)")
plt.title("Detection Performance vs Attack Strength")
plt.tight_layout()
plt.show()