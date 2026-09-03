import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_eval.agents import AGENT_REGISTRY
from agent_eval.compat import apply_sciworld_step_patch
from agent_eval.envs import AlfWorldEnv, SciWorldEnv
from agent_eval.envs.react import parse_react_action
from agent_eval.paths import SCIENCEWORLD_JAR
from agent_eval.tasks import AlfWorldTask, SciWorldTask


logger = logging.getLogger("agent_eval")


# Deliberately explicit registries. Adding a benchmark later should require
# adding its classes here, rather than introducing a complex plugin system.
TASK_REGISTRY = {
    "alfworld": AlfWorldTask,
    "sciworld": SciWorldTask,
}

ENV_REGISTRY = {
    "alfworld": AlfWorldEnv,
    "sciworld": SciWorldEnv,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load and validate a YAML config."""
    with path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")

    return data


def _repo_path(path_value: str) -> Path:
    """Resolve repository-relative config/output paths."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def _prepare_benchmark_runtime(
    benchmark: str,
    env_config: Dict[str, Any],
):
    """Create optional shared runtime objects for a benchmark.

    ALFWorld creates its own TextWorld environment for each task.

    ScienceWorld uses one Java-backed ScienceWorldEnv instance and loads a different task into it before each episode, matching current QLASS usage.
    """
    if benchmark != "sciworld":
        return None

    if not SCIENCEWORLD_JAR.exists():
        raise FileNotFoundError(
            f"ScienceWorld JAR was not found: {SCIENCEWORLD_JAR}. "
            "Copy it from the QLASS repository."
        )

    from scienceworld import ScienceWorldEnv

    # Required for the reward/completion metadata expected by SciWorldEnv.
    apply_sciworld_step_patch()

    internal_step_limit = int(env_config.get("internal_env_step_limit", 200))

    return ScienceWorldEnv(
        "",
        serverPath=str(SCIENCEWORLD_JAR),
        envStepLimit=internal_step_limit,
    )


def _build_env(
    benchmark: str,
    task,
    env_config: Dict[str, Any],
    runtime,
):
    """Instantiate the benchmark wrapper for one episode."""
    env_cls = ENV_REGISTRY[benchmark]

    # These fields configure dataset loading/runtime creation and should not be forwarded to BaseEnv.
    wrapper_config = {
        key: value for key, value in env_config.items() if key
        not in {
            "name",
            "split",
            "part_num",
            "part_idx",
            "internal_env_step_limit",
        }
    }

    if benchmark == "sciworld":
        return env_cls(task=task, env=runtime, **wrapper_config)

    return env_cls(task=task, **wrapper_config)


def _task_metadata(benchmark: str, task) -> Dict[str, Any]:
    """Store benchmark-specific identifiers without leaking them into runner logic."""
    if benchmark == "alfworld":
        return {"game_file": task.game_file}

    if benchmark == "sciworld":
        return {
            "sub_task_name": task.sub_task_name,
            "variation_idx": task.variation_idx,
        }

    return {}


def _run_episode(
    agent,
    env,
    task,
    benchmark: str,
    split: str,
    attempt_id: int,
    experiment_name: str,
) -> Dict[str, Any]:
    """Run one pure actor -> environment trajectory."""
    _, state = env.reset()

    initial_observation = env.get_current_observation()
    task_text = env.get_task_text()

    step_records = []

    while not state.finished:
        # Environment owns benchmark-specific prompt construction.
        messages = env.build_agent_messages()

        observation_before = env.get_current_observation()

        admissible_actions = env.get_admissible_commands()

        try:
            raw_output = agent.act(messages)

        except Exception as exc:
            # Preserve the failed episode instead of silently losing it.
            state.finished = True
            state.success = False
            state.terminate_reason = "agent_error"
            state.error = str(exc)

            logger.exception(
                "Agent failed on task=%s attempt=%d",
                task.task_id, attempt_id,
            )

            break

        # Save the action separately from the raw model output.
        try:
            parsed_action = parse_react_action(raw_output)
        except ValueError:
            parsed_action = "__invalid_action__"

        observation_after, state = env.step(raw_output)

        step_records.append(
            {
                "step_id": state.steps,
                "observation": observation_before,
                "agent_messages": copy.deepcopy(messages),
                "admissible_actions": list(admissible_actions),
                "raw_agent_output": raw_output,
                "parsed_action": parsed_action,
                "next_observation": observation_after,
                "reward": state.reward,
                "done": state.finished,
            }
        )

    return {
        "experiment_name": experiment_name,
        "benchmark": benchmark,
        "split": split,
        "task_id": task.task_id,
        "attempt_id": attempt_id,
        "task_text": task_text,
        "initial_observation": initial_observation,
        "success": state.success,
        # Environment-level episode reward after executing this action.
        # For ScienceWorld this follows the current QLASS behavior and is the best raw_score observed so far,
        # not the immediate reward delta.
        "reward": state.reward,
        "num_steps": state.steps,
        "terminate_reason": state.terminate_reason,
        "error": state.error,
        "task": _task_metadata(benchmark, task),
        "steps": step_records,
    }


def _append_jsonl(
    path: Path,
    record: Dict[str, Any],
) -> None:
    """Append one completed trajectory immediately.

    Writing after every episode prevents losing all results if a long run is
    interrupted halfway through.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal actor-only LLM-agent experiment.")

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML config.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output-directory override.",
    )

    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional task limit for small/debug runs.",
    )

    parser.add_argument(
        "--server-address",
        default=None,
        help="Optional SGLang server address override.",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model-name/path override for the actor config.",
    )

    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=None,
        help="Optional number of independent trajectories per task.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing trajectories.jsonl.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    # ------------------------------------------------------------------
    # Load configs.
    # ------------------------------------------------------------------

    experiment_config = _load_yaml(_repo_path(args.config))

    agent_config = _load_yaml(_repo_path(experiment_config["agent_config"]))

    env_config = _load_yaml(_repo_path(experiment_config["env_config"]))

    benchmark = str(env_config["name"]).strip().lower()

    if benchmark not in TASK_REGISTRY:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    agent_type = str(agent_config["type"]).strip().lower()

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unsupported agent type: "
            f"{agent_type}"
        )

    if args.server_address:
        agent_config["server_address"] = args.server_address
    if args.model_name:
        agent_config["model_name"] = args.model_name

    # ------------------------------------------------------------------
    # Dataset/run settings.
    # ------------------------------------------------------------------

    split = str(env_config.get("split", "test"))

    part_num = int(env_config.get("part_num", 1))

    part_idx = int(env_config.get("part_idx", -1))

    max_tasks = (
        args.max_tasks if args.max_tasks is not None else experiment_config.get("max_tasks")
    )

    if max_tasks is not None:
        max_tasks = int(max_tasks)

        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")

    num_trajectories = int(
        args.num_trajectories
        if args.num_trajectories is not None else experiment_config.get("num_trajectories", 1)
    )

    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    output_dir = _repo_path(args.output_dir or experiment_config["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories_path = output_dir / "trajectories.jsonl"
    metadata_path = output_dir / "run_metadata.json"

    if trajectories_path.exists():
        if args.overwrite:
            trajectories_path.unlink()

        else:
            raise FileExistsError(
                f"{trajectories_path} already exists. "
                "Use another output directory or pass --overwrite."
            )

    # ------------------------------------------------------------------
    # Initialize benchmark and actor.
    # ------------------------------------------------------------------

    task_cls = TASK_REGISTRY[benchmark]

    tasks, n_tasks = task_cls.load_tasks(split=split, part_num=part_num, part_idx=part_idx)

    logger.info(
        "Loaded benchmark=%s split=%s tasks=%d",
        benchmark, split, n_tasks,
    )

    runtime = _prepare_benchmark_runtime(benchmark, env_config)

    agent_cls = AGENT_REGISTRY[agent_type]
    agent = agent_cls(agent_config)

    processed_tasks = 0
    total_episodes = 0

    run_metadata = {
        "experiment_name": str(experiment_config["name"]),
        "benchmark": benchmark,
        "split": split,
        "model_name": str(agent_config["model_name"]),
        "num_trajectories": num_trajectories,
        "max_tasks": max_tasks,
        "status": "running",
    }

    try:
        # --------------------------------------------------------------
        # Benchmark-agnostic actor-only loop.
        # --------------------------------------------------------------

        for task in tasks:
            if (max_tasks is not None and processed_tasks >= max_tasks):
                break

            processed_tasks += 1

            for attempt_id in range(num_trajectories):
                env = _build_env(
                    benchmark=benchmark,
                    task=task,
                    env_config=env_config,
                    runtime=runtime,
                )

                trajectory = _run_episode(
                    agent=agent,
                    env=env,
                    task=task,
                    benchmark=benchmark,
                    split=split,
                    attempt_id=attempt_id,
                    experiment_name=str(experiment_config["name"]),
                )

                _append_jsonl(trajectories_path, trajectory)

                total_episodes += 1

                logger.info(
                    "task=%s attempt=%d success=%s reward=%s steps=%d reason=%s",
                    task.task_id, attempt_id, trajectory["success"], trajectory["reward"],
                    trajectory["num_steps"], trajectory["terminate_reason"],
                )
    except Exception as exc:
        run_metadata["status"] = "failed"
        run_metadata["failure"] = repr(exc)
        raise

    else:
        run_metadata["status"] = "completed"

    finally:
        run_metadata["processed_tasks"] = processed_tasks
        run_metadata["total_episodes"] = total_episodes

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(run_metadata, f, ensure_ascii=False, indent=2)

        agent.close()

    logger.info(
        "Finished %d episode(s) over %d task(s). Results: %s",
        total_episodes, processed_tasks, trajectories_path,
    )


if __name__ == "__main__":
    main()
