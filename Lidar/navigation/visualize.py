import matplotlib.pyplot as plt

def plot_comparison(normal_metrics, attacked_metrics):
    labels = ["Success Rate (%)", "Avg Reward"]
    normal_vals = [normal_metrics["success_rate"], normal_metrics["avg_reward"]]
    attack_vals = [attacked_metrics["success_rate"], attacked_metrics["avg_reward"]]

    x = range(len(labels))
    plt.bar(x, normal_vals, width=0.4, label="Nominal", align="center")
    plt.bar([i+0.4 for i in x], attack_vals, width=0.4, label="Attacked")

    plt.xticks([i+0.2 for i in x], labels)
    plt.title("Nominal vs Attacked Navigation Performance")
    plt.legend()
    plt.grid(True)
    plt.show()
