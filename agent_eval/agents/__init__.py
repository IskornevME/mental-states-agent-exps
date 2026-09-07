from agent_eval.agents.base import BaseAgent
from agent_eval.agents.human import HumanAgent
from agent_eval.agents.sglang import SGLangChatAgent


AGENT_REGISTRY = {
    "sglang_chat": SGLangChatAgent,
    "human": HumanAgent,
}


__all__ = [
    "BaseAgent",
    "HumanAgent",
    "SGLangChatAgent",
    "AGENT_REGISTRY",
]