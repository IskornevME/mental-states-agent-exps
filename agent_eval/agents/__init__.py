from agent_eval.agents.base import BaseAgent
from agent_eval.agents.sglang import SGLangChatAgent


AGENT_REGISTRY = {
    "sglang_chat": SGLangChatAgent,
}


__all__ = [
    "BaseAgent",
    "SGLangChatAgent",
    "AGENT_REGISTRY",
]