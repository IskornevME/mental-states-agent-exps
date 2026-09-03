import re
from typing import Iterable


def parse_react_action(llm_output: str) -> str:
    """Parse an action from the shared ReAct protocol.

    Preferred format:
        <think>...</think>
        <action>...</action>

    The legacy "Action: ..." format is temporarily supported to keep compatibility with old QLASS trajectories.
    """
    text = str(llm_output).strip()

    # Current Qwen / ReAct format.
    match = re.search(
        r"<action>\s*(.*?)\s*</action>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        action = match.group(1).strip()
        if action:
            return action

    # Legacy QLASS format.
    match = re.search(
        r"Action:\s*(.*)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match and match.group(1).strip():
        return match.group(1).strip()

    raise ValueError(
        "Could not parse action. Expected <action>...</action>."
    )


def format_admissible_actions(actions: Iterable[str]) -> str:
    """Format admissible actions exactly as in the QLASS ReAct prompt.

    Example:
        'go to cabinet 1'
         'open cabinet 1'
    """
    clean_actions = [str(action).strip() for action in actions if str(action).strip()]

    return "\n ".join(
        f"'{action}'" for action in clean_actions
    )
