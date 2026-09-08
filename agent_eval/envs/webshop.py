import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

from agent_eval.envs.base import BaseEnv
from agent_eval.envs.react import parse_react_action
from agent_eval.paths import REPO_ROOT
from agent_eval.state import State
from agent_eval.tasks.webshop import WebShopTask


def _repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


class WebShopEnv(BaseEnv):
    """WebShop wrapper preserving the QLASS interaction protocol."""

    def __init__(
        self,
        task: WebShopTask,
        env,
        instruction_path: str,
        icl_path: str,
        num_icl_examples: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.task = task
        self.env = env
        self.session_id = task.session_id

        self.num_icl_examples = max(int(num_icl_examples), 0)

        with _repo_path(instruction_path).open(encoding="utf-8") as f:
            self.instruction = f.read().strip()

        with _repo_path(icl_path).open(encoding="utf-8") as f:
            self.raw_icl = json.load(f)

        self.current_observation = ""
        self.current_task_text = ""

    def get_task_text(self) -> str:
        return self.current_task_text

    def get_current_observation(self) -> str:
        return self.current_observation

    def get_admissible_commands(self) -> List[str]:
        """Return WebShop actions for logging/future critic use.

        Search accepts arbitrary keywords. Click actions are constrained to the clickables on the current page.
        """
        available = self.env.get_available_actions()

        actions: List[str] = []

        if available.get("has_search_bar", False):
            actions.append("search[<keywords>]")

        actions.extend(f"click[{value}]" for value in available.get("clickables", []))

        return actions

    def build_agent_messages(self) -> List[Dict[str, str]]:
        """Return the same conversation structure used by QLASS."""
        messages = copy.deepcopy(self.state.history)

        # The released QLASS wrapper does NOT append a separate available
        # actions block. Keep it optional for future experiments.
        if self.include_admissible_actions and messages:
            actions = self.get_admissible_commands()

            if actions and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\nAvailable Actions:\n" + "\n".join(actions)

        return messages

    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append(
            {
                "role": "assistant",
                "content": llm_output,
            }
        )

        try:
            action = parse_react_action(llm_output)

        except ValueError as exc:
            observation = (
                "Observation: Invalid format. Your response must contain "
                "an action inside <action>...</action> tags."
            )

            self.state.error = str(exc)
            self.state.history.append(
                {
                    "role": "user",
                    "content": observation,
                }
            )

            self.current_observation = observation
            self.state.steps += 1
            self.state.reward = 0.0

            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"

            return observation, self.state

        try:
            observation, reward, done, _ = self.env.step(action=action)

        except AssertionError as exc:
            self.state.error = str(exc)
            observation = "Invalid action!"
            reward = 0.0
            done = False

        self.current_observation = str(observation).strip()

        observation_message = f"Observation:\n{self.current_observation}"

        self.state.history.append(
            {
                "role": "user",
                "content": observation_message,
            }
        )

        self.state.steps += 1

        # Preserve QLASS ordering: a terminal purchase at the last allowed
        # step still keeps its environment reward.
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = 0.0

        if done:
            self.state.finished = True

            # Preserve original QLASS semantics:
            # done means the shopping episode ended with Buy Now.
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = float(reward)

        return observation_message, self.state

    def reset(self) -> Tuple[str, State]:
        self.state = State()

        # The QLASS task id is directly the WebShop session id.
        self.env.reset(self.session_id)

        self.current_observation = str(self.env.observation).strip()

        self.current_task_text = str(self.env.instruction_text).strip()

        # Reproduce QLASS prompt_with_icl(..., icl_format="conversation"):
        #
        # user:      instruction
        # assistant: OK
        # user:      ICL observation
        # assistant: ICL response
        # ...
        # user:      current WebShop task
        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": self.instruction,
            },
            {
                "role": "assistant",
                "content": "OK",
            },
        ]

        for example in self.raw_icl[: self.num_icl_examples]:
            messages.extend(copy.deepcopy(example))

        messages.append(
            {
                "role": "user",
                "content": self.current_observation,
            }
        )

        self.state.history = messages

        return self.current_observation, self.state
