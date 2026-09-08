import json
from typing import Iterable, Tuple

from agent_eval.paths import WEBSHOP_DATA_DIR
from agent_eval.tasks.base import Task


class WebShopTask(Task):
    """One WebShop shopping session."""

    task_name = "webshop"

    def __init__(
        self,
        session_id: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.session_id = session_id

    @classmethod
    def load_tasks(cls, split: str, part_num: int = 1, part_idx: int = -1) -> Tuple[Iterable[Task], int]:
        if part_num <= 0:
            raise ValueError("part_num must be positive")

        if part_num > 1 and not 0 <= part_idx < part_num:
            raise ValueError("part_idx must satisfy 0 <= part_idx < part_num")

        if split == "train_1k":
            path = WEBSHOP_DATA_DIR / "train_indices.json"
            with path.open() as f:
                indices = json.load(f)
            indices = indices[:1000]

        elif split in {"train", "train_2k"}:
            path = WEBSHOP_DATA_DIR / "train_indices.json"
            with path.open() as f:
                indices = json.load(f)

        elif split == "test":
            path = WEBSHOP_DATA_DIR / "test_indices.json"
            with path.open() as f:
                indices = json.load(f)

        elif split == "test_500":
            path = WEBSHOP_DATA_DIR / "test_indices_500.json"
            with path.open() as f:
                indices = json.load(f)

        else:
            raise ValueError(f"Unknown WebShop split: {split}")

        # Same slicing convention as the current SciWorld adapter.
        if part_num > 1:
            start = len(indices) * part_idx // part_num
            end = len(indices) * (part_idx + 1) // part_num
            indices = indices[start:end]

        def generator():
            for session_id in indices:
                yield cls(task_id=session_id, session_id=session_id)

        return generator(), len(indices)
