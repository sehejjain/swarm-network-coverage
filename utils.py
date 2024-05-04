"""utils"""

from stable_baselines3.common.callbacks import BaseCallback


class EpisodicRewardLogger(BaseCallback):
    """Callback"""

    def __init__(self, verbose=0):
        super(EpisodicRewardLogger, self).__init__(verbose)
        self.episodic_rewards = []
        self.ep_len = []
        self.infos = []

    def _on_step(self) -> bool:
        # print(self.locals)
        self.infos.append(self.locals)
        info = self.locals["infos"][0]
        # print(info)
        if "episode" in info:
            self.episodic_rewards.append(info["episode"]["r"])
            self.ep_len.append(info["episode"]["l"])
        return True
        # # Retrieve the latest ep_rew_mean from the logger
        # latest_ep_rew_mean = self.logger.get_log_dict().get('rollout/ep_rew_mean', None)
        # latest_ep_len_mean = self.logger.get_log_dict().get('rollout/ep_len_mean', None)
        # if latest_ep_rew_mean is not None:
        #     self.episodic_rewards.append(latest_ep_rew_mean)
        # return True
