from abc import ABC, abstractmethod
from typing import Any, List, Mapping


class BaseAgent(ABC):
    """Minimal interface implemented by all actor backends.

    The environment is responsible for constructing benchmark-specific messages. The agent only receives chat messages and generates one response.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    @abstractmethod
    def act(self, messages: List[dict]) -> str:
        """Generate a single actor response."""
        raise NotImplementedError

    def __call__(self, messages: List[dict]) -> str:
        """Convenience alias compatible with the old QLASS calling style."""
        return self.act(messages)

    def close(self) -> None:
        """Release optional backend resources.

        Most agents do not need explicit cleanup, so the default implementation intentionally does nothing.
        """