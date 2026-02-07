import matplotlib.pyplot as plt
import numpy as np

def plot_anomaly_score(anomaly_scores, detection_flags, threshold=0.6, episode_idx=0):
    scores = np.array(anomaly_scores[episode_idx])
    detected = np.array(detection_flags[episode_idx])
    steps = np.arange(len(scores))

    plt.figure(figsize=(8, 4))

    # Anomaly score curve
    plt.plot(steps, scores, "-r", linewidth=2, label="Anomaly / Reconstruction Score")

    # Threshold
    plt.axhline(
        y=threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Detection Threshold"
    )

    # Highlight detected regions
    plt.fill_between(
        steps,
        scores,
        threshold,
        where=detected,
        color="red",
        alpha=0.25,
        label="Attack Detected"
    )

    plt.xlabel("Timestep")
    plt.ylabel("Anomaly Score")
    plt.title("Adversarial Detection via Temporal SHAP Signatures")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
