import re
from typing import Iterable


def parse_react_action(llm_output: str) -> str:
    """Parse the shared ReAct action protocol."""
    text = str(llm_output).strip()

    match = re.search(
        r"<action>\s*(.*?)\s*</action>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        action = match.group(1).strip()
        if action:
            return action

    # Temporary fallback for old QLASS trajectories/prompts.
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
    actions = [str(action).strip() for action in actions if str(action).strip()]
    return "\n".join(f"- {action}" for action in actions)
