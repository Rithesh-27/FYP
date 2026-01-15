from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
import numpy as np
import torch as th
import time
import os

from ddpg_lidar import AirSimDroneEnv

env = AirSimDroneEnv()

shape = env.action_space.shape[-1]

noise_sigma = 0.1 * np.ones(shape)   # reduced for fine-tuning
action_noise = NormalActionNoise(
    mean=np.zeros(shape),
    sigma=noise_sigma
)

policy_kwargs = dict(
    activation_fn=th.nn.Tanh,
    net_arch=[64, 32, 16]
)

log_dir = "tmp/"
eval_log_dir = "./eval_logs/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(eval_log_dir, exist_ok=True)

class SaveModelCallback(BaseCallback):

    def __init__(self, check_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = os.path.join(
            "./models/",
            time.strftime("%Y%m%d-%H%M%S") + "_ddpg_airsim_drone_callback"
        )

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print("==============================================================")
            print(f"Saving model checkpoint to {self.save_path}")
            self.model.save(self.save_path)
        return True

MODEL_PATH = "./models/20260113-141301_ddpg_airsim_drone_callback/ 20260113-141301_ddpg_airsim_drone.zip" 

model = DDPG.load(
    MODEL_PATH,
    env=env,
    tensorboard_log=log_dir,
    verbose=2
)

model.train_freq = TrainFreq(500, TrainFrequencyUnit.STEP)
model.action_noise = action_noise

model.learn(
    total_timesteps=50000,          
    callback=SaveModelCallback(check_freq=5000),
    reset_num_timesteps=False,      
    progress_bar=True,
    log_interval=50
)

final_name = os.path.join(
    "./models/",
    time.strftime("%Y%m%d-%H%M%S") + "_ddpg_airsim_drone"
)
model.save(final_name)