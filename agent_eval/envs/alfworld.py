from typing import Dict, List, Tuple

import textworld
import yaml
from alfworld.agents.environment.alfred_tw_env import (
    AlfredDemangler,
    AlfredExpert,
    AlfredInfos,
)

from agent_eval.envs.base import BaseEnv
from agent_eval.envs.react import format_admissible_actions, parse_react_action
from agent_eval.paths import ALFWORLD_DATA_DIR
from agent_eval.state import State
from agent_eval.tasks import AlfWorldTask


def _process_observation(observation: str) -> str:
    if observation.startswith("You arrive at loc "):
        observation = observation[observation.find(". ") + 2 :]
    return observation


class AlfWorldEnv(BaseEnv):
    """ALFWorld wrapper preserving the working QLASS ReAct behavior."""

    def __init__(self, task: AlfWorldTask, **kwargs) -> None:
        super().__init__(**kwargs)
        self.task = task
        self.env = self._load_single_task_env(task.game_file)

        self.current_admissible_commands: List[str] = []
        self.react_history: List[Tuple[str, str]] = []
        self.react_current_observation = task.observation

    def _load_single_task_env(self, game_file: str):
        with (ALFWORLD_DATA_DIR / "base_config.yaml").open() as f:
            config = yaml.safe_load(f)

        domain_randomization = config["env"]["domain_randomization"]
        if self.task.split != "train":
            domain_randomization = False

        wrappers = [
            AlfredDemangler(shuffle=domain_randomization),
            AlfredInfos,
        ]
        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["gamefile"],
        )

        training_method = config["general"]["training_method"]
        if training_method == "dqn":
            max_env_steps = config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_env_steps = config["dagger"]["training"]["max_nb_steps_per_episode"]
            if self.task.split == "train":
                wrappers.append(AlfredExpert(config["env"]["expert_type"]))
                request_infos.extras.append("expert_plan")
        else:
            raise ValueError(f"Unsupported ALFWorld training method: {training_method}")

        env_id = textworld.gym.register_games(
            [game_file],
            request_infos,
            batch_size=1,
            asynchronous=True,
            max_episode_steps=max_env_steps,
            wrappers=wrappers,
        )
        return textworld.gym.make(env_id)

    def get_task_text(self) -> str:
        marker = "Your task is to:"
        if marker in self.task.observation:
            return self.task.observation.split(marker, 1)[1].strip()
        return self.task.observation.strip()

    def get_current_observation(self) -> str:
        return str(self.react_current_observation or "").strip()

    def get_admissible_commands(self) -> List[str]:
        return [
            action
            for action in self.current_admissible_commands
            if action != "help"
        ]

    def build_agent_messages(self) -> List[Dict[str, str]]:
        recent_history = (
            self.react_history[-self.history_length :]
            if self.history_length > 0
            else []
        )

        parts = [
            "You are an expert agent operating in the ALFRED Embodied Environment.",
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

    def _conduct_action(self, action: str):
        observation, _, done, info = self.env.step([action])

        self.current_admissible_commands = list(
            (info.get("admissible_commands") or [[]])[0]
        )
        clean_observation = _process_observation(observation[0])
        reward = info["won"][0]
        return clean_observation, reward, done[0]

    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append(
            {"role": "assistant", "content": llm_output}
        )
        observation_before = self.get_current_observation()

        try:
            action = parse_react_action(llm_output)
            observation, reward, done = self._conduct_action(action)
            self.react_history.append((observation_before, action))
            self.react_current_observation = observation
        except Exception as exc:
            self.state.success = False
            self.state.reward = 0.0
            self.state.error = str(exc)

            observation = (
                "Observation: Error Input. Your response must contain an "
                "action inside <action>...</action> tags."
            )
            self.react_history.append(
                (observation_before, "__invalid_action__")
            )
            self.react_current_observation = observation

            self.state.history.append(
                {"role": "user", "content": observation}
            )
            self.state.steps += 1

            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.terminate_reason = "max_steps"

            return observation, self.state

        observation_message = f"Observation: {observation}"
        self.state.history.append(
            {"role": "user", "content": observation_message}
        )
        self.state.steps += 1

        if done:
            self.state.finished = True
            self.state.success = bool(reward)
            self.state.reward = float(reward)
            self.state.terminate_reason = (
                "success" if self.state.success else "env_done"
            )
        elif self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.reward = float(reward)
            self.state.terminate_reason = "max_steps"

        return observation_message, self.state

    def reset(self) -> Tuple[str, State]:
        self.state = State()
        self.state.error = self.task.game_file

        self.react_history = []
        self.react_current_observation = self.task.observation

        _, info = self.env.reset()
        self.current_admissible_commands = list(
            (info.get("admissible_commands") or [[]])[0]
        )

        return self.task.observation, self.state
