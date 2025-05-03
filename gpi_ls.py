import numpy as np
import mo_gymnasium as mo_gym

import envs

from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPILSContinuousAction as GPILS,
)
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

GAMMA = 0.98


def main():
    env = mo_gym.make("mo-HalfCheetah-v4")
    env = MORecordEpisodeStatistics(
        env, gamma=GAMMA
    )  # wrapper for recording statistics

    eval_env = mo_gym.make("mo-HalfCheetah-v4")  # environment used for evaluation

    # Your code here:
    agent = GPILS(
        env,
        gradient_updates=10,
        min_priority=0.1,
        batch_size=128,
        buffer_size=int(4e5),
        dynamics_rollout_starts=1000,
        dynamics_rollout_len=5,
        dynamics_rollout_freq=250,
        dynamics_rollout_batch_size=50000,
        dynamics_train_freq=250,
        dynamics_buffer_size=200000,
        dynamics_real_ratio=0.1,
        dynamics_min_uncertainty=2.0,
        per=True,
        project_name="DyLam",
        log=True,
    )

    agent.train(total_timesteps=200000, eval_env=eval_env, ref_point=np.array([-1, -1]))

    agent


if __name__ == "__main__":
    main()
