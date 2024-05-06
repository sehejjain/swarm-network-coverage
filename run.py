'''run'''
import logging, time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from env import NetworkEnv
from utils import EpisodicRewardLogger

logging.getLogger().setLevel(logging.INFO)



n_envs = 10
n = 8
x_min = 0
x_max = 100
y_min = 0
y_max = 100
step = 1
radius = 20
render_mode = "rgb_array"
max_ep_len = 5000
bonus_threshold = 0.6
bonus_reward = 100
overlap_threshold = 0.1


def get_env(train=True):
    """Return env"""
    return NetworkEnv(
        n,
        (x_min, x_max),
        (y_min, y_max),
        step=step,
        radius=radius,
        render_mode=render_mode,
        max_ep_len=max_ep_len,
        bonus_threshold=bonus_threshold,
        bonus_reward=bonus_reward,
        overlap_threshold=overlap_threshold,
        train=train,
    )


env = make_vec_env(get_env, n_envs=n_envs)

checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path="./logs/ppo4/")

eval_env = Monitor(get_env(train=False))

eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/ppo4/best_model",
                             log_path="./logs/ppo4/results", eval_freq=50_000)

callback = CallbackList([checkpoint_callback, eval_callback, EpisodicRewardLogger()])


model1 = PPO("MultiInputPolicy", env, verbose=0, tensorboard_log="tensorboard_logs/", learning_rate=1e-5)

logging.info("Start training")
logging.info("Device %a", model1.device)

# model1.load("logs/best_model/best_model")

start=time.time()
model1.learn(
    total_timesteps=10_000_000,
    callback=callback,
    tb_log_name="ppo4",
    progress_bar=True,
    reset_num_timesteps=False,
)
print("Time taken: ", time.time()-start)

