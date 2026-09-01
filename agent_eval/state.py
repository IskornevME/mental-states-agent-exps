from copy import deepcopy
from typing import Any, Dict, List, Optional


class State:
    """Episode state shared by all environments."""

    def __init__(
        self,
        reward: Optional[float] = None,
        finished: bool = False,
        success: bool = False,
        terminate_reason: Optional[str] = None,
    ) -> None:
        self.history: List[Dict[str, Any]] = []
        self.reward = reward
        self.finished = finished
        self.success = success
        self.terminate_reason = terminate_reason
        self.error: Optional[str] = None
        self.steps = 0

    @property
    def empty(self) -> bool:
        return not self.history

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "steps": self.steps,
                "reward": self.reward,
                "finished": self.finished,
                "success": self.success,
                "terminate_reason": self.terminate_reason,
                "error": self.error,
            },
            "conversations": deepcopy(self.history),
        }

    @classmethod
    def load_json(cls, data: Dict[str, Any]) -> "State":
        state = cls()
        state.history = deepcopy(data["conversations"])

        meta = data["meta"]
        state.steps = meta["steps"]
        state.reward = meta["reward"]
        state.finished = meta["finished"]
        state.success = meta["success"]
        state.terminate_reason = meta["terminate_reason"]
        state.error = meta.get("error")
        return state
