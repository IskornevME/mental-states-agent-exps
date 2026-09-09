import sys
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_eval.envs.react import parse_react_action

DATA_PATH = REPO_ROOT / "data" / "expert" / "alfworld_sft.json"


def parse_thought(output: str) -> str:
    """
    Достает reasoning из legacy QLASS-формата:

        Thought: ...
        Action: ...
    """
    match = re.search(
        r"Thought:\s*(.*?)\s*\n\s*Action:",
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return match.group(1).strip() if match else ""


def to_react_format(output: str) -> str:
    """
    Преобразует старый expert output в текущий формат Qwen/ReAct:

        Thought: ...
        Action: ...

    ->

        <think>...</think>
        <action>...</action>
    """
    thought = parse_thought(output)
    action = parse_react_action(output)

    return (
        f"<think>{thought}</think>\n"
        f"<action>{action}</action>"
    )


def get_expert_steps(trajectory: dict) -> list[dict]:
    """
    Преобразует одну expert trajectory в последовательность decision steps.

    Первые два сообщения:
        human -> инструкция
        gpt   -> OK

    Поэтому сама trajectory начинается с conversations[2:].

    Далее формат всегда:
        human -> state / observation
        gpt   -> expert Thought + Action
    """
    messages = trajectory["conversations"][2:]

    steps = []
    previous_actions = []

    for i in range(0, len(messages), 2):
        state_message = messages[i]
        expert_message = messages[i + 1]

        state = state_message["value"].strip()
        expert_output = expert_message["value"].strip()

        # После первого шага observation содержит служебный prefix.
        if steps and state.lower().startswith("observation:"):
            state = state[len("Observation:") :].strip()

        action = parse_react_action(expert_output)

        steps.append(
            {
                "step_id": len(steps),

                # State ПЕРЕД expert action.
                "state": state,

                # Все expert actions, которые привели нас к этому state.
                "previous_actions": previous_actions.copy(),

                "thought": parse_thought(expert_output),
                "action": action,

                # Удобно для будущего env.step(...).
                "react_output": to_react_format(expert_output),
            }
        )

        previous_actions.append(action)

    return steps


def main():
    # Оригинальный train_sft.py QLASS читает данные обычным json.load()
    with DATA_PATH.open("r", encoding="utf-8") as f:
        trajectories = json.load(f)

    print(f"Loaded trajectories: {len(trajectories)}")

    # Для примера берем первую trajectory.
    trajectory = trajectories[0]
    steps = get_expert_steps(trajectory)

    print(f"Trajectory id: {trajectory['id']}")
    print(f"Game file: {trajectory['game_file']}")
    print(f"Number of expert steps: {len(steps)}")

    # Покажем первые три decision steps.
    for step in steps[:3]:
        print("\n" + "=" * 80)
        print(f"STEP {step['step_id']}")

        print("\nSTATE:")
        print(step["state"])

        print("\nPREVIOUS EXPERT ACTIONS:")
        print(step["previous_actions"])

        print("\nEXPERT THOUGHT:")
        print(step["thought"])

        print("\nEXPERT ACTION:")
        print(step["action"])

        print("\nCURRENT REACT FORMAT:")
        print(step["react_output"])


if __name__ == "__main__":
    main()