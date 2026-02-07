import torch
import torch.nn as nn
import numpy as np
import joblib
import shap
from collections import deque

# -----------------------------
# LSTM Detector Architecture
# -----------------------------
class AdversarialDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.sigmoid(self.fc(out))


# -----------------------------
# Shield Wrapper
# -----------------------------
class SecurityShield:
    def __init__(self, agent, detector_path, scaler_path, device):
        self.device = device
        self.agent = agent

        # Load detector
        self.detector = AdversarialDetector().to(device)
        self.detector.load_state_dict(torch.load(detector_path, map_location=device))
        self.detector.eval()

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # SHAP Explainer
        background = torch.rand(100, 5).to(device)
        self.explainer = shap.DeepExplainer(agent.policy.actor, background)

        self.window = deque(maxlen=10)

    def detect(self, obs):
        obs_t = torch.FloatTensor(obs).to(self.device)

        # SHAP values
        shap_vals = self.explainer.shap_values(obs_t)
        signature = np.concatenate([
            shap_vals[0].flatten(),
            shap_vals[1].flatten()
        ])

        self.window.append(signature)

        if len(self.window) < 10:
            return 0.0

        window = np.array(self.window)
        window_scaled = self.scaler.transform(
            window.reshape(-1, 10)
        ).reshape(1, 10, 10)

        with torch.no_grad():
            prob = self.detector(
                torch.FloatTensor(window_scaled).to(self.device)
            ).item()

        return prob

    def mitigate(self, obs):
        """
        Simple mitigation: mask LiDAR
        """
        safe_obs = obs.copy()
        safe_obs[0][0:3] = 1.0
        return safe_obs
