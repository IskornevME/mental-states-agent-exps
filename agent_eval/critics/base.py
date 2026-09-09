from abc import ABC, abstractmethod
from typing import Dict, List


class BaseCritic(ABC):
    """Base interface for candidate-scoring critics."""

    @abstractmethod
    def score_candidate(
        self,
        actor_messages: List[Dict[str, str]],
        raw_action: str,
    ) -> float:
        """Return a scalar score for one state-action candidate."""
        raise NotImplementedError

    def close(self) -> None:
        """Release critic resources if needed."""