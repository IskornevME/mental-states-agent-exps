from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple


class Task(ABC):
    """Base class for one benchmark task instance."""

    task_name = "base"

    def __init__(self, task_id: Any = None, **_: Any) -> None:
        self.task_id = task_id

    @classmethod
    @abstractmethod
    def load_tasks(
        cls,
        split: str,
        part_num: int = 1,
        part_idx: int = -1,
    ) -> Tuple[Iterable["Task"], int]:
        raise NotImplementedError
