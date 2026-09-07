from typing import Any, List, Mapping

from agent_eval.agents.base import BaseAgent


class HumanAgent(BaseAgent):
    """Interactive CLI actor for manual benchmark exploration.

    The agent receives exactly the same chat messages as an LLM actor, displays them in the terminal,
    and asks the user to provide only the environment action.

    The action is automatically wrapped into the shared ReAct format so that the rest of the pipeline does not need any human-specific logic.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)

        self.think_placeholder = str(
            config.get(
                "think_placeholder",
                "Human action selected manually; no explicit reasoning provided.",
            )
        ).strip()

    def act(self, messages: List[dict]) -> str:
        """Display the actor input and request one action from the user."""
        if not messages:
            raise ValueError("HumanAgent received an empty message list")

        print("\n" + "=" * 100)
        print("HUMAN ACTOR INPUT")
        print("=" * 100)

        # Print messages without modifying their content.
        # The role markers below are UI-only and are not part of the prompt.
        for message in messages:
            role = str(message.get("role", "unknown")).upper()
            content = str(message.get("content", ""))

            print(f"\n[{role}]")
            print(content)

        print("\n" + "-" * 100)
        print(
            "Enter only the environment action. "
            "<think>/<action> tags will be added automatically."
        )

        while True:
            try:
                action = input("ACTION> ").strip()
            except EOFError as exc:
                raise RuntimeError("Human input stream was closed while waiting for an action") from exc

            if action:
                break

            print("Action cannot be empty. Please try again.")

        # Keep exactly the same downstream ReAct contract as for LLM actors.
        raw_output = (
            f"<think>{self.think_placeholder}</think>\n"
            f"<action>{action}</action>"
        )

        print("\n[SUBMITTED]")
        print(raw_output)

        return raw_output
