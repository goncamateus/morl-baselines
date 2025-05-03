import numpy as np
import mo_gymnasium as mo_gym

import envs

from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

GAMMA = 0.98


def main():
    eval_env = mo_gym.make("mo-HalfCheetah-v4")  # environment used for evaluation

    # Your code here:
    agent = PGMORL(
        "mo-HalfCheetah-v4",
        origin=np.array([-1.0, -1.0]),
        project_name="MORL-Baselines",
        log=True,
        update_epochs=20,
    )

    agent.train(
        total_timesteps=5000000,
        eval_env=eval_env,
        ref_point=np.array([-1.0, -1.0]),
    )


if __name__ == "__main__":
    main()
