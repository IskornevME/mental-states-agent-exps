import argparse
import copy
import json
import logging
import re
import sys
import yaml

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_eval.agents import AGENT_REGISTRY
from agent_eval.compat import apply_sciworld_step_patch
from agent_eval.envs import AlfWorldEnv, SciWorldEnv, WebShopEnv
from agent_eval.envs.react import parse_react_action
from agent_eval.paths import SCIENCEWORLD_JAR
from agent_eval.tasks import AlfWorldTask, SciWorldTask, WebShopTask

logger = logging.getLogger("critic_data")

class InvalidExpertTrajectoryError(RuntimeError):
    """Raised when an expert trajectory is not a successful terminal trajectory."""
    pass


# -----------------------------------------------------------------------------
# Benchmark registries
# -----------------------------------------------------------------------------

TASK_REGISTRY = {
    "alfworld": AlfWorldTask,
    "sciworld": SciWorldTask,
    "webshop": WebShopTask,
}

ENV_REGISTRY = {
    "alfworld": AlfWorldEnv,
    "sciworld": SciWorldEnv,
    "webshop": WebShopEnv,
}


# =============================================================================
# Generic config/path helpers
# =============================================================================


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load and validate a YAML mapping."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")

    return data


def repo_path(path_value: str) -> Path:
    """Resolve repository-relative paths."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


# =============================================================================
# Benchmark setup
# =============================================================================


def prepare_benchmark_runtime(
    benchmark: str,
    env_config: Dict[str, Any],
):
    """
    Create optional shared runtime objects required by a benchmark.

    ALFWorld:
        every task already carries its TextWorld environment.

    ScienceWorld:
        one Java-backed ScienceWorldEnv is reused by this worker.

    WebShop:
        one WebAgentTextEnv is reused by this worker.
    """
    if benchmark == "webshop":
        from eval.webshop.web_agent_site.envs import WebAgentTextEnv

        return WebAgentTextEnv(
            observation_mode="text",
            human_goals=True,
        )

    if benchmark == "sciworld":
        if not SCIENCEWORLD_JAR.exists():
            raise FileNotFoundError(
                f"ScienceWorld JAR was not found: {SCIENCEWORLD_JAR}. "
                "Copy it from the QLASS repository first."
            )

        from scienceworld import ScienceWorldEnv

        # The current SciWorld wrapper relies on this compatibility patch.
        apply_sciworld_step_patch()

        internal_step_limit = int(
            env_config.get("internal_env_step_limit", 200)
        )

        return ScienceWorldEnv(
            "",
            serverPath=str(SCIENCEWORLD_JAR),
            envStepLimit=internal_step_limit,
        )

    if benchmark == "alfworld":
        return None

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def build_env(
    benchmark: str,
    task,
    env_config: Dict[str, Any],
    runtime,
):
    """Instantiate the current repository wrapper for one task."""
    env_cls = ENV_REGISTRY[benchmark]

    # These fields configure dataset loading/runtime creation and must not be
    # passed to BaseEnv.
    wrapper_config = {
        key: value
        for key, value in env_config.items() if key
        not in {
            "name",
            "split",
            "part_num",
            "part_idx",
            "internal_env_step_limit",
        }
    }

    if benchmark in {"sciworld", "webshop"}:
        return env_cls(
            task=task,
            env=runtime,
            **wrapper_config,
        )

    return env_cls(
        task=task,
        **wrapper_config,
    )


# =============================================================================
# Matching benchmark tasks with expert trajectories
# =============================================================================


def normalize_alfworld_game_file(value: str) -> str:
    """
    Normalize current ALFWorld game_file and expert game_file to one key.

    Current environment paths can look like:
        .../json_2.1.1/train/<task>/game.tw-pddl

    Expert data stores:
        data/json_2.1.1/train/<task>

    Both become simply:
        <task>
    """
    value = str(value).replace("\\", "/")

    marker = "json_2.1.1/train/"

    if marker in value:
        value = value.split(marker, 1)[1]

    if value.endswith("/game.tw-pddl"):
        value = value[: -len("/game.tw-pddl")]

    return value.strip("/")


def task_key(benchmark: str, task) -> str:
    """
    Return a stable task identifier used for expert matching.

    ALFWorld task_id cannot be used because it is only a local integer within
    the currently loaded slice. We therefore match it through game_file.
    """
    if benchmark == "alfworld":
        return normalize_alfworld_game_file(task.game_file)

    if benchmark == "sciworld":
        return str(task.task_id)

    if benchmark == "webshop":
        return str(task.session_id)

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def expert_key(
    benchmark: str,
    record: Dict[str, Any],
) -> str:
    """Return the key of one expert trajectory."""
    if benchmark == "alfworld":
        if "game_file" not in record:
            raise KeyError(
                "ALFWorld expert record does not contain 'game_file'."
            )

        return normalize_alfworld_game_file(record["game_file"])

    if "id" not in record:
        raise KeyError(
            f"{benchmark} expert record does not contain 'id'."
        )

    return str(record["id"])


def load_expert_index(
    path: Path,
    benchmark: str,
) -> Dict[str, Dict[str, Any]]:
    """Load expert trajectories and index them by benchmark task id."""
    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in expert file: {path}")

    result: Dict[str, Dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            raise TypeError(
                f"Expert records must be JSON objects, got {type(record)}"
            )

        key = expert_key(benchmark, record)

        if key in result:
            raise ValueError(f"Duplicate expert trajectory key: {key!r}")

        result[key] = record

    return result


# =============================================================================
# Legacy expert trajectory -> current ReAct trajectory
# =============================================================================


def extract_expert_outputs(record: Dict[str, Any]) -> List[str]:
    """
    Extract assistant actions from a legacy QLASS trajectory.

    Old expert files normally start with:

        human -> general instruction
        gpt   -> OK

    The technical "OK" message is not an environment action and is removed.
    """
    conversations = record.get("conversations")

    if not isinstance(conversations, list):
        raise ValueError(
            "Expert trajectory must contain a list in 'conversations'."
        )

    outputs: List[str] = []

    for message in conversations:
        if not isinstance(message, dict):
            continue

        role = message.get("from", message.get("role"))

        if role not in {"gpt", "assistant"}:
            continue

        content = str(
            message.get("value", message.get("content", ""))
        ).strip()

        if not content:
            continue

        if content == "OK":
            continue

        outputs.append(content)

    if not outputs:
        raise ValueError("Expert trajectory does not contain any assistant actions.")

    return outputs


def to_react(
    raw_output: str,
) -> str:
    """
    Convert legacy QLASS output:
        Thought: ...
        Action: ...

    to the current repository format:
        <think>...</think>
        <action>...</action>

    If the expert output is already in the current format, keep it unchanged.
    """
    raw_output = str(raw_output).strip()

    if re.search(
        r"<action>\s*.*?\s*</action>",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        return raw_output

    action = parse_react_action(raw_output)

    thought_match = re.search(
        r"Thought:\s*(.*?)\s*Action:",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if thought_match:
        thought = thought_match.group(1).strip()

        return (
            f"<think>{thought}</think>\n"
            f"<action>{action}</action>"
        )

    return (
        f"<action>{action}</action>"
    )


# =============================================================================
# Canonical critic state
# =============================================================================


def canonical_actor_messages(
    env,
) -> List[Dict[str, str]]:
    """
    Return the exact state representation used by the current actor.

    All supported benchmarks now use the same high-level contract:

        [
            {
                "role": "user",
                "content": "<complete ReAct state>"
            }
        ]

    We intentionally enforce this here. It protects the collected critic
    dataset from silently drifting away from critic inference format.
    """
    messages = env.build_agent_messages()

    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError(
            "Critic data collection requires exactly one user message "
            f"from env.build_agent_messages(). Got: {messages!r}"
        )

    return messages


# =============================================================================
# Tree representation
# =============================================================================


def make_node(
    critic_state: Optional[str],
    critic_action: Optional[str],
    reward: float,
    finished: bool,
    success: bool,
    source: str,
) -> Dict[str, Any]:
    """
    Create one JSON-serializable tree node.

    No custom TreeNode class or pickle is required. This makes collected trees
    portable across code revisions and easy to inspect manually.
    """
    return {
        "critic_state": critic_state,
        "critic_action": critic_action,
        "reward": float(reward),
        "finished": bool(finished),
        "success": bool(success),
        "source": source,
        "children": [],
    }


# =============================================================================
# Environment replay
# =============================================================================


def replay_prefix(env, executed_outputs: List[str]):
    """
    Reset the environment and replay the selected raw ReAct outputs.

    Search-tree nodes store actions rather than serialized environment objects.
    Therefore any state can be reconstructed deterministically from its action
    prefix.

    This is the same principle already used by QNet inference in the current
    repository.
    """
    _, state = env.reset()

    for raw_output in executed_outputs:
        _, state = env.step(raw_output)

        if state.finished:
            break

    return state


# =============================================================================
# Sampling alternative actions
# =============================================================================


def build_diverse_sampling_messages(
    canonical_messages: List[Dict[str, str]],
    previous_outputs: List[str],
) -> List[Dict[str, str]]:
    """
    Add the QLASS-style "give another response" instruction.

    IMPORTANT:
    this modified prompt is used ONLY to generate diverse branches.

    It must never be stored as critic_state. The critic is trained on the
    canonical environment prompt, because that is what it sees at inference.
    """
    messages = copy.deepcopy(canonical_messages)

    if not previous_outputs:
        return messages

    if len(previous_outputs) == 1:
        suffix = (
            "\n\nPlease provide another reasonable response different "
            "from the previous answer:\n"
            f"{previous_outputs[0]}"
        )

    else:
        lines = ["\n\nYou have already given the following answers:"]

        for idx, output in enumerate(previous_outputs, start=1):
            lines.append(f"Answer {idx}: {output}")

        lines.append(
            "Please provide another reasonable response "
            "different from the previous answers."
        )

        suffix = "\n".join(lines)

    messages[0]["content"] += suffix

    return messages


def sample_candidate(
    agent,
    env,
    previous_outputs: List[str],
) -> Tuple[str, str]:
    """
    Sample one alternative actor response.

    Returns:
        critic_state:
            canonical pre-action prompt WITHOUT exploration instructions.

        raw_output:
            model output generated with the temporary diversity instruction.
    """
    canonical_messages = canonical_actor_messages(env)

    sampling_messages = build_diverse_sampling_messages(
        canonical_messages,
        previous_outputs,
    )

    raw_output = agent.act(sampling_messages)

    critic_state = str(canonical_messages[0]["content"]).strip()

    return (
        critic_state,
        str(raw_output).strip(),
    )


# =============================================================================
# Policy rollout
# =============================================================================


def rollout_to_terminal(
    agent,
    env,
    parent_node: Dict[str, Any],
    prefix: List[str],
) -> Tuple[
    List[Tuple[Dict[str, Any], List[str]]],
    Any,
]:
    """
    Roll the actor policy out until the benchmark reaches a terminal state.

    Note:
    MAX_DEPTH does NOT truncate a rollout.

    This matches the original QLASS collector. MAX_DEPTH controls which tree
    nodes can later be selected for another expansion; the rollout itself is
    allowed to continue until normal benchmark termination/MAX_STEPS.
    """
    created_nodes: List[
        Tuple[Dict[str, Any], List[str]]
    ] = []

    current_parent = parent_node
    current_prefix = list(prefix)

    state = env.state

    while not state.finished:
        messages = canonical_actor_messages(env)

        raw_output = str(agent.act(messages)).strip()

        _, state = env.step(raw_output)

        child = make_node(
            critic_state=str(messages[0]["content"]).strip(),
            critic_action=raw_output,
            reward=float(state.reward or 0.0),
            finished=state.finished,
            success=state.success,
            source="rollout",
        )

        current_parent["children"].append(child)

        current_prefix.append(raw_output)

        created_nodes.append(
            (
                child,
                list(current_prefix),
            )
        )

        current_parent = child

    return (
        created_nodes,
        state,
    )


# =============================================================================
# Expert trajectory replay
# =============================================================================


def collect_expert_steps(
    env,
    expert_record: Dict[str, Any],
) -> Tuple[
    List[Dict[str, str]], float, bool, bool,
]:
    """
    Replay an expert trajectory under the CURRENT ReAct prompt.

    The legacy state messages stored in *_sft.json are deliberately ignored.
    Before every expert action we ask the current environment wrapper to build
    its current canonical prompt. This gives exactly the critic_state format
    that the new actor/critic pipeline uses.
    """
    expert_outputs = [
        to_react(output)
        for output in extract_expert_outputs(expert_record)
    ]

    _, state = env.reset()

    expert_steps: List[Dict[str, str]] = []

    for raw_output in expert_outputs:
        if state.finished:
            break

        messages = canonical_actor_messages(env)

        expert_steps.append(
            {
                "critic_state": str(messages[0]["content"]).strip(),
                "critic_action": raw_output,
            }
        )

        _, state = env.step(raw_output)

    if not expert_steps:
        raise ValueError("Expert trajectory produced zero executable steps.")

    if state.finished:
        final_reward = float(state.reward or 0.0)

        final_success = bool(state.success)

    else:
        # Preserve the old QLASS fallback semantics for a malformed/incomplete
        # expert trajectory. WebShop expert data has an explicit reward;
        # ALFWorld/SciWorld normally fall back to 0.
        final_reward = float(expert_record.get("reward", 0.0) or 0.0)
        final_success = False

        logger.warning(
            "Expert actions ended before environment termination. "
            "expert_steps=%d env_steps=%d max_steps=%d "
            "last_action=%r current_observation=%r admissible_actions=%r",
            len(expert_steps),
            state.steps,
            env.max_steps,
            expert_steps[-1]["critic_action"] if expert_steps else None,
            env.get_current_observation(),
            env.get_admissible_commands(),
        )

    return (
        expert_steps,
        final_reward,
        final_success,
        bool(state.finished),
    )


# =============================================================================
# Stage 1: actor tree exploration
# =============================================================================


def explore_actor_tree(
    agent,
    env,
    root: Dict[str, Any],
    max_depth: int,
    min_prune_depth: int,
    samples_per_depth: int,
    positive_reward_threshold: float,
) -> None:
    """
    Perform the actor-driven tree exploration from QLASS.

    Algorithm:
      1. Start from root.
      2. At each queued state, sample up to SAMPLES_PER_DEPTH alternatives.
      3. Execute every new alternative.
      4. If it is non-terminal, rollout actor policy to terminal.
      5. Also queue the new action node itself for deeper exploration.
      6. If the rollout finishes with sufficiently positive reward, selected
         rollout nodes after MIN_PRUNE_DEPTH are also queued for exploration.

    Despite the historical "MCTS" name, this is not UCT-MCTS: there are no
    visit counts or UCB scores. We deliberately preserve QLASS behavior.
    """
    node_queue = deque(
        [
            (root, [], 1,)
        ]
    )

    while node_queue:
        (parent_node, prefix, depth) = node_queue.popleft()

        # A node coming from a rollout may already have one child: the next
        # action in that rollout. QLASS counts such existing children toward
        # SAMPLES_PER_DEPTH.
        existing_outputs = [
            str(child["critic_action"])
            for child in parent_node["children"]
            if child.get("critic_action") is not None
        ]

        num_new_samples = max(samples_per_depth - len(existing_outputs), 0)

        # Contains both already existing children and newly attempted outputs.
        # It is used only for the diversity-generation prompt.
        sampled_outputs = list(existing_outputs)

        for _ in range(num_new_samples):
            # Every alternative must start from the same tree state.
            state = replay_prefix(env, prefix)

            if state.finished:
                break

            critic_state, raw_output = sample_candidate(
                agent=agent,
                env=env,
                previous_outputs=sampled_outputs,
            )

            # QLASS performs a fixed number of sampling attempts. If the actor
            # returns an exact duplicate, we skip that tree child instead of
            # adding a second identical branch.
            is_duplicate = raw_output in sampled_outputs

            sampled_outputs.append(raw_output)

            if is_duplicate:
                logger.debug(
                    "Skipping exact duplicate candidate at depth=%d", depth,
                )

                continue

            _, new_state = env.step(raw_output)

            child = make_node(
                critic_state=critic_state,
                critic_action=raw_output,
                reward=float(new_state.reward or 0.0),
                finished=new_state.finished,
                success=new_state.success,
                source="exploration",
            )

            parent_node["children"].append(child)

            child_prefix = prefix + [raw_output]

            if new_state.finished:
                continue

            # QLASS uses depth=1 for expansion of the root state.
            new_depth = depth + 1

            # The sampled action itself may later be expanded.
            if new_depth <= max_depth:
                node_queue.append((child, child_prefix, new_depth))

            # Independently of MAX_DEPTH, complete this branch to terminal.
            (
                rollout_nodes,
                terminal_state,
            ) = rollout_to_terminal(
                agent=agent,
                env=env,
                parent_node=child,
                prefix=child_prefix,
            )

            terminal_reward = float(terminal_state.reward or 0.0)

            # Preserve the original QLASS criterion:
            #     reward > 0.01
            # It is exposed as a parameter because other benchmarks can have
            # non-binary reward scales.
            if terminal_reward <= positive_reward_threshold:
                continue

            # Nodes along a successful/positive rollout can themselves become
            # expansion points deeper in the tree.
            rollout_depth = new_depth + 1

            for (
                rollout_node,
                rollout_prefix,
            ) in rollout_nodes:
                if rollout_depth > max_depth:
                    break

                if rollout_depth > min_prune_depth:
                    node_queue.append(
                        (rollout_node, rollout_prefix, rollout_depth)
                    )

                rollout_depth += 1


# =============================================================================
# Stage 2: expert path + brother rollouts
# =============================================================================


def add_expert_path(
    agent,
    env,
    root: Dict[str, Any],
    expert_steps: List[Dict[str, str]],
    expert_reward: float,
    expert_success: bool,
    max_depth: int,
) -> None:
    """
    Add the expert trajectory to the already explored tree.

    Starting from the SECOND expert decision, QLASS also samples one
    alternative ("brother") action from each expert state up to MAX_DEPTH and
    rolls this alternative out to terminal.

    Result:

        expert state
          ├── actor brother -> rollout -> terminal
          └── expert action -> next expert state
    """
    current_parent = root

    for expert_idx, expert_step in enumerate(expert_steps):
        # expert_idx == 0 is the first expert action.
        #
        # Old QLASS starts brother rollouts from the second expert decision.
        if (expert_idx > 0 and expert_idx + 1 <= max_depth):
            expert_prefix = [
                step["critic_action"] for step in expert_steps[:expert_idx]
            ]

            brother_state = replay_prefix(env, expert_prefix)

            if not brother_state.finished:
                messages = canonical_actor_messages(env)

                brother_output = str(agent.act(messages)).strip()

                _, brother_state = env.step(brother_output)

                brother_node = make_node(
                    critic_state=str(messages[0]["content"]).strip(),
                    critic_action=brother_output,
                    reward=float(brother_state.reward or 0.0),
                    finished=brother_state.finished,
                    success=brother_state.success,
                    source="expert_brother",
                )

                current_parent["children"].append(brother_node)

                if not brother_state.finished:
                    rollout_to_terminal(
                        agent=agent,
                        env=env,
                        parent_node=brother_node,
                        prefix=expert_prefix + [brother_output],
                    )

        is_last = expert_idx == len(expert_steps) - 1

        # Legacy QLASS stores the final expert reward on every expert node.
        #
        # build_dataset.py later resets all internal-node rewards to zero, so
        # only the terminal expert leaf retains this reward for vanilla Q.
        expert_node = make_node(
            critic_state=expert_step["critic_state"],
            critic_action=expert_step["critic_action"],
            reward=expert_reward,
            finished=is_last,
            success=expert_success if is_last else False,
            source="expert",
        )

        current_parent["children"].append(expert_node)

        current_parent = expert_node


# =============================================================================
# Complete search for one task
# =============================================================================


def collect_task_tree(
    agent,
    env,
    expert_record: Dict[str, Any],
    max_depth: int,
    min_prune_depth: int,
    samples_per_depth: int,
    positive_reward_threshold: float,
) -> Tuple[
    Dict[str, Any], Dict[str, Any],
]:
    """Collect one complete QLASS-style expert-anchored tree."""

    # First replay the legacy expert trajectory with the current environment
    # and current ReAct prompts.
    (
        expert_steps,
        expert_reward,
        expert_success,
        expert_finished,
    ) = collect_expert_steps(
        env,
        expert_record,
    )

    # The expert trajectory is the anchor of the whole search procedure.
    # If it does not reproduce a successful terminal solution in the current
    # environment, do not spend compute exploring this task and do not include it in critic training data.
    if not expert_finished or not expert_success:
        if not expert_finished:
            reason = "expert actions ended before environment termination"
        else:
            reason = "expert trajectory reached a terminal state without success"

        raise InvalidExpertTrajectoryError(
            f"{reason}; "
            f"steps={len(expert_steps)}, "
            f"reward={expert_reward}"
        )

    root = make_node(
        critic_state=None,
        critic_action=None,
        reward=0.0,
        finished=False,
        success=False,
        source="root",
    )

    # ------------------------------------------------------------------
    # Stage 1:
    # generic actor-driven exploration and policy rollouts.
    # ------------------------------------------------------------------

    explore_actor_tree(
        agent=agent,
        env=env,
        root=root,
        max_depth=max_depth,
        min_prune_depth=min_prune_depth,
        samples_per_depth=samples_per_depth,
        positive_reward_threshold=positive_reward_threshold,
    )

    # ------------------------------------------------------------------
    # Stage 2:
    # add expert path and alternative brother rollouts.
    # ------------------------------------------------------------------

    add_expert_path(
        agent=agent,
        env=env,
        root=root,
        expert_steps=expert_steps,
        expert_reward=expert_reward,
        expert_success=expert_success,
        max_depth=max_depth,
    )

    expert_metadata = {
        "num_steps": len(expert_steps),
        "reward": expert_reward,
        "success": expert_success,
        "environment_finished": expert_finished,
    }

    return (
        root,
        expert_metadata,
    )


# =============================================================================
# Output
# =============================================================================


def append_jsonl(
    path: Path,
    record: Dict[str, Any],
) -> None:
    """
    Save one completed task immediately.

    This is intentionally JSONL instead of the pickle files used by old QLASS:
    it is portable, inspectable and survives code/class changes.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Collect QNet training trees with QLASS-style expert-anchored search.")
    )

    parser.add_argument("--benchmark", required=True, choices=sorted(TASK_REGISTRY))

    parser.add_argument("--agent-config", required=True, help="Path to actor YAML config.")

    parser.add_argument("--env-config", required=True, help="Path to benchmark YAML config.")

    parser.add_argument("--expert-data", required=True, help="Path to benchmark expert trajectories JSON.")

    parser.add_argument("--server-address", default=None, help="Optional SGLang server-address override.")

    parser.add_argument("--model-name", default=None, help="Optional actor model/path override.")

    parser.add_argument("--split", default="train")

    # ------------------------------------------------------------------
    # Worker/data slicing
    # ------------------------------------------------------------------

    parser.add_argument("--worker-idx", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=None,
        help="Maximum number of matched expert tasks processed by this worker.",
    )

    # ------------------------------------------------------------------
    # Search hyperparameters
    # ------------------------------------------------------------------

    parser.add_argument("--max-depth", type=int, default=5)

    parser.add_argument("--min-prune-depth", type=int, default=3)

    parser.add_argument("--samples-per-depth", type=int, default=2)

    parser.add_argument("--positive-reward-threshold", type=float, default=0.01)

    parser.add_argument("--history-length", type=int, default=50)

    parser.add_argument("--max-steps", type=int, default=None)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    parser.add_argument("--output", required=True, help="Worker JSONL tree output.")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not 0 <= args.worker_idx < args.num_workers:
        raise ValueError("worker_idx must satisfy 0 <= worker_idx < num_workers")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    if args.max_depth <= 0:
        raise ValueError("max_depth must be positive")
    if args.min_prune_depth < 0:
        raise ValueError("min_prune_depth must be non-negative")
    if args.samples_per_depth <= 0:
        raise ValueError("samples_per_depth must be positive")

    # ------------------------------------------------------------------
    # Resolve/load configs.
    # ------------------------------------------------------------------

    agent_config_path = repo_path(args.agent_config)
    env_config_path = repo_path(args.env_config)
    expert_data_path = repo_path(args.expert_data)
    output_path = repo_path(args.output)
    agent_config = load_yaml(agent_config_path)
    env_config = load_yaml(env_config_path)

    benchmark = args.benchmark.strip().lower()

    configured_benchmark = str(
        env_config.get("name", "")
    ).strip().lower()

    if configured_benchmark != benchmark:
        raise ValueError(
            "Benchmark/env config mismatch: "
            f"--benchmark={benchmark!r}, env config name={configured_benchmark!r}"
        )

    agent_type = str(
        agent_config.get("type", "")
    ).strip().lower()

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unsupported agent type: {agent_type!r}")

    if agent_type == "human":
        raise ValueError(
            "HumanAgent is not supported for automatic critic-data collection."
        )

    # The launcher normally overrides these two fields.
    if args.server_address:
        agent_config["server_address"] = args.server_address

    if args.model_name:
        agent_config["model_name"] = args.model_name

    # Use collection-specific prompt horizon / episode limit without changing
    # the committed benchmark YAML.
    env_config["history_length"] = args.history_length

    if args.max_steps is not None:
        env_config["max_steps"] = args.max_steps

    # ------------------------------------------------------------------
    # Load expert data.
    # ------------------------------------------------------------------

    expert_index = load_expert_index(expert_data_path, benchmark)

    logger.info(
        "Loaded %d expert trajectories from %s",
        len(expert_index), expert_data_path,
    )

    # ------------------------------------------------------------------
    # Load this worker's benchmark partition.
    # ------------------------------------------------------------------

    task_cls = TASK_REGISTRY[benchmark]

    part_idx = args.worker_idx if args.num_workers > 1 else -1

    tasks, n_tasks = task_cls.load_tasks(
        split=args.split,
        part_num=args.num_workers,
        part_idx=part_idx,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if args.overwrite:
            output_path.unlink()

        else:
            raise FileExistsError(
                f"{output_path} already exists. Pass --overwrite or use another output path."
            )

    # ------------------------------------------------------------------
    # Initialize runtime and actor.
    # ------------------------------------------------------------------

    runtime = prepare_benchmark_runtime(benchmark, env_config)

    agent_cls = AGENT_REGISTRY[agent_type]

    agent = agent_cls(agent_config)

    processed_tasks = 0
    skipped_without_expert = 0
    skipped_invalid_expert = 0

    progress_total = (
        min(n_tasks, args.max_tasks)
        if args.max_tasks is not None else n_tasks
    )

    logger.info(
        "Starting collection: benchmark=%s split=%s "
        "worker=%d/%d tasks=%d max_depth=%d "
        "samples_per_depth=%d",
        benchmark,
        args.split,
        args.worker_idx,
        args.num_workers,
        n_tasks,
        args.max_depth,
        args.samples_per_depth,
    )

    try:
        with tqdm(total=progress_total, dynamic_ncols=True, desc=f"worker {args.worker_idx}") as progress:

            for task in tasks:
                if args.max_tasks is not None and processed_tasks >= args.max_tasks:
                    break

                key = task_key(benchmark, task)

                expert_record = expert_index.get(key)

                # The task split may be larger than the available expert set.
                # Old QLASS also effectively explores their intersection.
                if expert_record is None:
                    skipped_without_expert += 1

                    logger.debug("No expert trajectory for task key=%s; skipping.", key)

                    continue

                env = build_env(
                    benchmark=benchmark,
                    task=task,
                    env_config=env_config,
                    runtime=runtime,
                )

                try:
                    root, expert_metadata = collect_task_tree(
                        agent=agent,
                        env=env,
                        expert_record=expert_record,
                        max_depth=args.max_depth,
                        min_prune_depth=args.min_prune_depth,
                        samples_per_depth=args.samples_per_depth,
                        positive_reward_threshold=args.positive_reward_threshold,
                    )

                except InvalidExpertTrajectoryError as exc:
                    skipped_invalid_expert += 1

                    logger.warning(
                        "Skipping task=%s because expert trajectory is invalid: %s",
                        key, exc,
                    )

                    continue

                record = {
                    "benchmark": benchmark,
                    "task_id": key,

                    # Important for reproducibility and for remembering which
                    # actor distribution this future critic was trained on.
                    "actor": {
                        "type": agent_type,
                        "model_name": str(agent_config.get( "model_name", "")),
                        "agent_config": str(args.agent_config),
                    },

                    "search": {
                        "max_depth": args.max_depth,
                        "min_prune_depth": args.min_prune_depth,
                        "samples_per_depth": args.samples_per_depth,
                        "positive_reward_threshold": args.positive_reward_threshold,
                        "history_length": args.history_length,
                        "max_steps": int(env_config.get("max_steps", 0)),
                    },

                    "expert": expert_metadata,
                    "root": root,
                }

                # Save after every task. A failure later in a long collection
                # run will not destroy already completed trees.
                append_jsonl(output_path, record)

                processed_tasks += 1

                progress.update(1)

                logger.info(
                    "Collected task=%s "
                    "expert_steps=%d "
                    "expert_reward=%s",
                    key,
                    expert_metadata["num_steps"],
                    expert_metadata["reward"],
                )

    finally:
        agent.close()

    if processed_tasks == 0:
        raise RuntimeError(
            "No valid task was collected. "
            f"Skipped without expert match: {skipped_without_expert}; "
            f"skipped because expert trajectory was invalid: {skipped_invalid_expert}."
        )

    logger.info(
        "Finished worker %d: collected=%d "
        "skipped_without_expert=%d "
        "skipped_invalid_expert=%d "
        "output=%s",
        args.worker_idx,
        processed_tasks,
        skipped_without_expert,
        skipped_invalid_expert,
        output_path,
    )


if __name__ == "__main__":
    main()
