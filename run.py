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
print("hello")



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
bonus_threshold = 0.9
bonus_reward = 1000
overlap_threshold = 0.1


def get_env():
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
    )


env = make_vec_env(get_env, n_envs=n_envs)

checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path="./logs/")

eval_env = Monitor(get_env())

eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/best_model",
                             log_path="./logs/results", eval_freq=100_000)

callback = CallbackList([checkpoint_callback, eval_callback, EpisodicRewardLogger()])


model1 = PPO("MultiInputPolicy", env, verbose=0, tensorboard_log="tensorboard_logs/")

logging.info("Start training")
logging.info("Device %a", model1.device)
start=time.time()
model1.learn(
    total_timesteps=10_000,
    callback=callback,
    tb_log_name="ppo1",
    progress_bar=True,
    reset_num_timesteps=False,
)
print("Time taken: ", time.time()-start)