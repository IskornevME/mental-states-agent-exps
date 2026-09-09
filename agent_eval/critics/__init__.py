from agent_eval.critics.base import BaseCritic
from agent_eval.critics.qnet import QwenQNetCritic


# In the future 'llm_judge' may be added
CRITIC_REGISTRY = {
    "qwen_qnet": QwenQNetCritic,
}


__all__ = [
    "BaseCritic",
    "QwenQNetCritic",
    "CRITIC_REGISTRY",
]