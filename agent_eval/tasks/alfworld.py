import os
from typing import Iterable, Tuple

import alfworld
import alfworld.agents.environment as envs
import yaml

from agent_eval.paths import ALFWORLD_DATA_DIR
from agent_eval.tasks.base import Task


_SPLIT_TO_ALFWORLD = {
    "train": ("train", 3321),
    "dev": ("eval_in_distribution", 140),
    "test": ("eval_out_of_distribution", 134),
}


class AlfWorldTask(Task):
    """One ALFWorld task."""

    task_name = "alfworld"

    def __init__(
        self,
        game_file: str,
        env: envs.AlfredTWEnv,
        obs: str,
        split: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.game_file = game_file
        self.observation = obs
        self.split = split
        self.env = env

    @classmethod
    def load_tasks(
        cls,
        split: str,
        part_num: int = 1,
        part_idx: int = -1,
    ) -> Tuple[Iterable[Task], int]:
        if split not in _SPLIT_TO_ALFWORLD:
            raise ValueError(f"Unknown ALFWorld split: {split}")
        if part_num <= 0:
            raise ValueError("part_num must be positive")
        if part_num > 1 and not 0 <= part_idx < part_num:
            raise ValueError("part_idx must satisfy 0 <= part_idx < part_num")

        os.environ["ALFWORLD_DATA"] = str(ALFWORLD_DATA_DIR)

        config_path = ALFWORLD_DATA_DIR / "base_config.yaml"
        with config_path.open() as f:
            config = yaml.safe_load(f)

        alfworld_split, n_tasks = _SPLIT_TO_ALFWORLD[split]

        env_cls = getattr(alfworld.agents.environment, config["env"]["type"])
        env = env_cls(config, train_eval=alfworld_split)
        if not isinstance(env, alfworld.agents.environment.AlfredTWEnv):
            raise TypeError("Only AlfredTWEnv is supported by this MVP")

        env = env.init_env(batch_size=1)

        if part_num > 1:
            part_sizes = [n_tasks // part_num] * part_num
            part_sizes[-1] += n_tasks % part_num
            env.skip(sum(part_sizes[:part_idx]))
            n_tasks = part_sizes[part_idx]

        def generator():
            for local_idx in range(n_tasks):
                obs, info = env.reset()
                task_obs = "\n".join(obs[0].split("\n\n")[1:])
                game_file = info["extra.gamefile"][0]

                yield cls(
                    task_id=local_idx,
                    game_file=game_file,
                    env=env,
                    obs=task_obs,
                    split=split,
                )

        return generator(), n_tasks
