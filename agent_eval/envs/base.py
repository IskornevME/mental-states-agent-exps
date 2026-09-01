from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from agent_eval.state import State


class BaseEnv(ABC):
    """Common interface for text environments used by LLM agents."""

    def __init__(
        self,
        max_steps: int = 40,
        include_admissible_actions: bool = True,
        history_length: int = 50,
        **_,
    ) -> None:
        self.max_steps = int(max_steps)
        self.include_admissible_actions = bool(include_admissible_actions)
        self.history_length = max(int(history_length), 0)
        self.state = State()

    def get_task_text(self) -> str:
        return ""

    def get_current_observation(self) -> str:
        return ""

    def get_inventory(self) -> str:
        return ""

    def get_admissible_commands(self) -> List[str]:
        return []

    @abstractmethod
    def build_agent_messages(self) -> List[Dict[str, str]]:
        raise NotImplementedError

    def build_react_actor_messages(self, history_length: int | None = None):
        """Temporary compatibility alias for current QLASS code."""
        if history_length is not None:
            old_value = self.history_length
            self.history_length = max(int(history_length), 0)
            try:
                return self.build_agent_messages()
            finally:
                self.history_length = old_value
        return self.build_agent_messages()

    @abstractmethod
    def step(self, llm_output: str) -> Tuple[str, State]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Tuple[str, State]:
        raise NotImplementedError
