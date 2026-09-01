import json
from typing import Iterable, Tuple

from agent_eval.paths import SCIWORLD_DATA_DIR
from agent_eval.tasks.base import Task


class SciWorldTask(Task):
    """One ScienceWorld task variation."""

    task_name = "sciworld"

    def __init__(
        self,
        sub_task_name: str,
        variation_idx: int,
        split: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sub_task_name = sub_task_name
        self.variation_idx = variation_idx
        self.split = split

    @classmethod
    def load_tasks(
        cls,
        split: str,
        part_num: int = 1,
        part_idx: int = -1,
    ) -> Tuple[Iterable[Task], int]:
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown ScienceWorld split: {split}")
        if part_num <= 0:
            raise ValueError("part_num must be positive")
        if part_num > 1 and not 0 <= part_idx < part_num:
            raise ValueError("part_idx must satisfy 0 <= part_idx < part_num")

        with (SCIWORLD_DATA_DIR / f"{split}_indices.json").open() as f:
            task_indices = json.load(f)
        with (SCIWORLD_DATA_DIR / "taskname2id.json").open() as f:
            taskname2id = json.load(f)

        if part_num > 1:
            start = len(task_indices) * part_idx // part_num
            end = len(task_indices) * (part_idx + 1) // part_num
            task_indices = task_indices[start:end]

        def generator():
            for task_name, variation_idx in task_indices:
                yield cls(
                    task_id=f"{taskname2id[task_name]}_{variation_idx}",
                    sub_task_name=task_name,
                    variation_idx=variation_idx,
                    split=split,
                )

        return generator(), len(task_indices)
