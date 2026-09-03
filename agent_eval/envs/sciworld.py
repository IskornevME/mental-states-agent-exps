import json
from typing import Dict, List, Tuple

from scienceworld import ScienceWorldEnv

from agent_eval.envs.base import BaseEnv
from agent_eval.envs.react import format_admissible_actions, parse_react_action
from agent_eval.paths import SCIWORLD_DATA_DIR
from agent_eval.state import State
from agent_eval.tasks import SciWorldTask


class SciWorldEnv(BaseEnv):
    """ScienceWorld wrapper using the same ReAct contract as ALFWorld."""

    def __init__(
        self,
        task: SciWorldTask,
        env: ScienceWorldEnv,
        **kwargs,
    ) -> None:
        configured_max_steps = kwargs.get("max_steps")
        super().__init__(**kwargs)

        self.task = task
        self.env = env
        self._configured_max_steps = (
            int(configured_max_steps)
            if configured_max_steps is not None
            else None
        )

        with (SCIWORLD_DATA_DIR / "max_steps.json").open() as f:
            self.max_steps_dict = json.load(f)

        self.current_task_text = ""
        self.current_observation = ""
        self.current_inventory = ""
        self.current_admissible_commands: List[str] = []
        self.react_history: List[Tuple[str, str]] = []

    def get_task_text(self) -> str:
        return self.current_task_text

    def get_current_observation(self) -> str:
        return self.current_observation

    def get_inventory(self) -> str:
        return self.current_inventory

    def get_admissible_commands(self) -> List[str]:
        return list(self.current_admissible_commands)

    def build_agent_messages(self) -> List[Dict[str, str]]:
        recent_history = (
            self.react_history[-self.history_length :]
            if self.history_length > 0
            else []
        )

        parts = [
            "You are an expert agent operating in the ScienceWorld environment.",
            f"Your task is to: {self.get_task_text()}",
        ]

        if recent_history:
            first_step = len(self.react_history) - len(recent_history) + 1
            history_lines = []
            for offset, (observation, action) in enumerate(recent_history):
                step = first_step + offset
                history_lines.append(
                    f"[Observation {step}: '{observation}', Action {step}: '{action}']"
                )
            parts.append(
                f"Prior to this step, you have already taken "
                f"{len(self.react_history)} step(s)."
            )
            parts.append(
                "Below are the most recent observations and actions:\n"
                + "\n".join(history_lines)
            )

        parts.append(
            f"You are now at step {len(self.react_history) + 1} and your "
            f"current observation is: {self.get_current_observation()}"
        )

        if self.current_inventory:
            parts.append(
                f"Your current inventory is: {self.current_inventory}"
            )

        if self.include_admissible_actions:
            parts.append(
                "Your admissible actions of the current situation are:\n"
                + format_admissible_actions(self.get_admissible_commands())
            )

        parts.append(
            "Now it's your turn to take an action.\n"
            "First reason step-by-step about the current situation. "
            "The reasoning MUST be enclosed within <think>...</think> tags.\n"
            "Then choose one action and present it within "
            "<action>...</action> tags."
        )

        return [{"role": "user", "content": "\n\n".join(parts).strip()}]

    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append(
            {"role": "assistant", "content": llm_output}
        )
        observation_before = self.current_observation

        try:
            action = parse_react_action(llm_output)
        except ValueError as exc:
            observation = (
                "Observation: Invalid format. Your response must contain an "
                "action inside <action>...</action> tags."
            )
            self.state.error = str(exc)
            self.state.history.append(
                {"role": "user", "content": observation}
            )
            self.react_history.append(
                (observation_before, "__invalid_action__")
            )
            self.current_observation = observation
            self.state.steps += 1

            if self.state.reward is None:
                self.state.reward = 0.0

            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"

            return observation, self.state

        try:
            observation, _, done, info = self.env.step(action)
            reward = float(info["raw_score"])

            self.current_observation = str(observation).strip()
            self.current_inventory = str(info.get("inv", "")).strip()
            self.current_admissible_commands = list(
                self.env.get_valid_action_object_combinations()
            )
            self.react_history.append((observation_before, action))

            if self.state.reward is None or reward > self.state.reward:
                self.state.reward = reward

        except AssertionError as exc:
            self.state.error = str(exc)
            observation = "Invalid action!"
            done = False
            info = {}
            self.current_observation = observation
            self.react_history.append((observation_before, action))

        observation_message = f"Observation: {observation}"
        self.state.history.append(
            {"role": "user", "content": observation_message}
        )
        self.state.steps += 1

        if done:
            task_completed = bool(info.get("task_completed", False))
            self.state.finished = True
            self.state.success = task_completed

            if task_completed:
                self.state.terminate_reason = "success"
            elif info.get("terminated_by_step_limit", False):
                self.state.terminate_reason = "internal_env_step_limit"
            elif info.get("terminated_by_negative_score", False):
                self.state.terminate_reason = "negative_score"
            else:
                self.state.terminate_reason = "environment_done_unknown"

        elif self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"

        return observation_message, self.state

    def reset(self) -> Tuple[str, State]:
        self.state = State()
        self.react_history = []

        task_default_max_steps = int(
            self.max_steps_dict[self.task.sub_task_name]
        )
        self.max_steps = (
            self._configured_max_steps
            if self._configured_max_steps is not None
            else task_default_max_steps
        )

        self.env.load(
            self.task.sub_task_name,
            self.task.variation_idx,
            simplificationStr="easy",
            generateGoldPath=False,
        )
        observation, info = self.env.reset()

        self.current_task_text = str(info["taskDesc"]).strip()
        self.current_observation = str(observation).strip()
        self.current_inventory = str(info.get("inv", "")).strip()
        self.current_admissible_commands = list(
            self.env.get_valid_action_object_combinations()
        )

        return self.current_observation, self.state
