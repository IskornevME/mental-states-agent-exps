from pathlib import Path
from typing import Dict, List, Tuple

from agent_eval.envs.base import BaseEnv
from agent_eval.envs.react import format_admissible_actions, parse_react_action
from agent_eval.paths import REPO_ROOT
from agent_eval.state import State
from agent_eval.tasks.webshop import WebShopTask


WEBSHOP_REACT_TEMPLATE_NO_HIS = """
You are an expert agent operating in the WebShop environment.
Your task is to: {task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

You can use search[<keywords>] when search is available.
For click actions, choose one of the listed click[...] actions.

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an action for the current step and present it within <action> </action> tags.
"""


WEBSHOP_REACT_TEMPLATE = """
You are an expert agent operating in the WebShop environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

You can use search[<keywords>] when search is available.
For click actions, choose one of the listed click[...] actions.

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an action for the current step and present it within <action> </action> tags.
"""


class WebShopEnv(BaseEnv):
    """WebShop wrapper using the shared single-message ReAct protocol."""

    def __init__(
        self,
        task: WebShopTask,
        env,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.task = task
        self.env = env
        self.session_id = task.session_id

        self.current_observation = ""
        self.current_task_text = ""

        self.react_history: List[Tuple[str, str]] = []

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
        """Build one self-contained WebShop ReAct prompt."""
        history_length = max(int(self.history_length), 0)

        admissible_text = format_admissible_actions(self.get_admissible_commands())

        if not self.react_history:
            prompt = WEBSHOP_REACT_TEMPLATE_NO_HIS.format(
                task_description=self.get_task_text(),
                current_observation=self.get_current_observation(),
                admissible_actions=admissible_text,
            )

        else:
            recent_history = (
                self.react_history[-history_length:]
                if history_length > 0 else []
            )

            first_step = len(self.react_history) - len(recent_history) + 1

            history_lines = []

            for offset, (observation, action) in enumerate(recent_history):
                step_num = first_step + offset

                history_lines.append(
                    f"[Observation {step_num}: '{observation}', "
                    f"Action {step_num}: '{action}']"
                )

            prompt = WEBSHOP_REACT_TEMPLATE.format(
                task_description=self.get_task_text(),
                step_count=len(self.react_history),
                history_length=len(recent_history),
                action_history="\n".join(history_lines),
                current_step=len(self.react_history) + 1,
                current_observation=self.get_current_observation(),
                admissible_actions=admissible_text,
            )

        # Keep the option configurable, although the reference WebShop setup now includes available actions by default.
        if not self.include_admissible_actions:
            admissible_line = (
                "Your admissible actions of the current situation are: "
                f"[{admissible_text}].\n"
            )
            prompt = prompt.replace(admissible_line, "")

        return [
            {
                "role": "user",
                "content": prompt.strip(),
            }
        ]

    def step(self, llm_output: str) -> Tuple[str, State]:
        observation_before = self.get_current_observation()

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
            self.react_history.append((observation_before, "__invalid_action__"))
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

        self.react_history.append((observation_before, action))

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
        self.react_history = []

        # The QLASS task id is directly the WebShop session id.
        self.env.reset(self.session_id)

        self.current_observation = str(self.env.observation).strip()

        self.current_task_text = str(self.env.instruction_text).strip()

        return self.current_observation, self.state
