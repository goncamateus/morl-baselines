import numpy as np

from gymnasium.spaces import Box
from rsoccer_gym.ssl.ssl_hw_challenge.pass_endurance import SSLPassEnduranceEnv


class PassEnduranceStratEnv(SSLPassEnduranceEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.reward_dim = 2
        self.reward_space = Box(low=-1, high=1, shape=(self.reward_dim,))
        self.cumulative_reward_info = {
            "reward_Pass": 0,
            "reward_Angle": 0,
            "Original_reward": 0,
        }
        self.is_holding = True

    def reset(self, *, seed=None, options=None):
        self.cumulative_reward_info = {
            "reward_Pass": 0,
            "reward_Angle": 0,
            "Original_reward": 0,
        }
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.has_shoot = action[1] > 0 and self.frame.robots_blue[0].infrared
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, self.cumulative_reward_info

    def _angle_reward(self):
        if self.has_shoot:
            my_angle = self.frame.robots_blue[0].theta
            receiver_angle = self.frame.robots_blue[1].theta
            angle_diff = abs(180 - abs(my_angle - receiver_angle))
            if angle_diff > 1.5:
                return -1
            else:
                return 1
        return 0

    def _calculate_reward_and_done(self):
        reward = np.zeros(2, dtype=np.float32)
        done = False
        if self.has_shoot and self.is_holding:
            self.is_holding = False
        if self.is_holding:
            self.holding_steps += 1
            if self.holding_steps > 40:
                done = True
                reward += np.array([0.0, -1.0])
        if self.frame.robots_blue[1].infrared:
            reward += np.array([0.0, 1.0])
            done = True
            self.cumulative_reward_info["reward_Pass"] += 1
        elif not done and not self.is_holding:
            rw_angle = self._angle_reward()
            reward += np.array([rw_angle, 0.0])
            self.cumulative_reward_info["reward_Angle"] += rw_angle

        self.cumulative_reward_info["Original_reward"] += reward.sum()
        return reward, done
