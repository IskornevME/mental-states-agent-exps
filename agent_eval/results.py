import json
import math
from collections import Counter, OrderedDict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a possibly missing value to a finite float."""
    if value is None:
        return default

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def parse_bool(value: Any) -> bool:
    """Parse bool-like values from saved JSON records."""
    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes"}:
            return True

        if normalized in {"false", "0", "no", "", "none", "null"}:
            return False

    return bool(value)


def resolve_trajectory_path(input_path: str | Path) -> Path:
    """Accept either a run directory or its trajectories.jsonl file."""
    path = Path(input_path)

    if path.is_dir():
        path = path / "trajectories.jsonl"

    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    return path


def load_trajectories(input_path: str | Path) -> list[dict[str, Any]]:
    """Load one trajectory object per JSONL line."""
    path = resolve_trajectory_path(input_path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_number}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number} must contain a JSON object")

            records.append(record)

    if not records:
        raise ValueError(f"No trajectories found in {path}")

    return records


def group_trajectories(
    records: Iterable[dict[str, Any]],
    expected_benchmark: str,
) -> OrderedDict[str, list[dict[str, Any]]]:
    """Validate attempt ids and group trajectories by task."""
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    for index, record in enumerate(records):
        benchmark = str(record.get("benchmark", "")).strip().lower()

        if benchmark != expected_benchmark:
            raise ValueError(
                f"Trajectory {index} has benchmark={benchmark!r}; expected {expected_benchmark!r}"
            )

        if record.get("task_id") is None:
            raise ValueError(f"Trajectory {index} has no task_id")

        if record.get("attempt_id") is None:
            raise ValueError(f"Trajectory {index} has no attempt_id")

        grouped.setdefault(str(record["task_id"]), []).append(record)

    for task_id, trajectories in grouped.items():
        trajectories.sort(key=lambda record: int(record["attempt_id"]))

        attempt_ids = [int(record["attempt_id"]) for record in trajectories]

        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError(f"Duplicate attempt ids for task {task_id}: {attempt_ids}")

        expected_ids = list(range(len(trajectories)))

        if attempt_ids != expected_ids:
            raise ValueError(
                f"Unexpected attempt ids for task {task_id}: {attempt_ids}; "
                f"expected {expected_ids}"
            )

    attempt_counts = Counter(len(trajectories) for trajectories in grouped.values())

    if len(attempt_counts) != 1:
        raise ValueError(
            f"Different tasks have different numbers of trajectories: {dict(sorted(attempt_counts.items()))}"
        )

    return grouped


def calculate_alfworld_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate the actor-only ALFWorld metrics used in QLASS analyses."""
    grouped = group_trajectories(records, expected_benchmark="alfworld")
    n_trajs = len(next(iter(grouped.values())))

    scores_by_task = [
        [
            float(parse_bool(trajectory.get("success"))) for trajectory in trajectories
        ]
        for trajectories in grouped.values()
    ]

    metrics: dict[str, Any] = {
        "benchmark": "alfworld",
        "num_tasks": len(grouped),
        "n_trajs": n_trajs,
        "num_trajectories": len(records),
        "first_trajectory_avg_reward": mean(scores[0] for scores in scores_by_task),
        "last_trajectory_avg_reward": mean(scores[-1] for scores in scores_by_task),
        "mean_of_n_avg_reward": mean(mean(scores) for scores in scores_by_task),
        "best_of_n_avg_reward": mean(max(scores) for scores in scores_by_task),
        "success_rate_by_attempt": [],
        "recovery_after_all_previous_failures": [],
        "recovery_after_previous_attempt_failure": [],
        "best_of_k": [],
    }

    for attempt_id in range(n_trajs):
        attempt_scores = [
            task_scores[attempt_id] for task_scores in scores_by_task
        ]

        metrics["success_rate_by_attempt"].append(
            {
                "attempt_id": attempt_id,
                "rate": mean(attempt_scores),
                "successes": int(sum(attempt_scores)),
                "num_tasks": len(attempt_scores),
            }
        )

    for attempt_id in range(1, n_trajs):
        eligible_all = [
            scores for scores in scores_by_task if max(scores[:attempt_id]) == 0
        ]

        recovered_all = sum(
            scores[attempt_id] > 0 for scores in eligible_all
        )

        metrics["recovery_after_all_previous_failures"].append(
            {
                "attempt_id": attempt_id,
                "rate": (
                    recovered_all / len(eligible_all) if eligible_all else 0.0
                ),
                "recovered": recovered_all,
                "eligible": len(eligible_all),
            }
        )

        eligible_previous = [
            scores for scores in scores_by_task if scores[attempt_id - 1] == 0
        ]

        recovered_previous = sum(scores[attempt_id] > 0 for scores in eligible_previous)

        metrics["recovery_after_previous_attempt_failure"].append(
            {
                "attempt_id": attempt_id,
                "rate": (
                    recovered_previous / len(eligible_previous)
                    if eligible_previous else 0.0
                ),
                "recovered": recovered_previous,
                "eligible": len(eligible_previous),
            }
        )

    for k in range(1, n_trajs + 1):
        metrics["best_of_k"].append(
            {
                "k": k,
                "avg_reward": mean(
                    max(scores[:k]) for scores in scores_by_task
                ),
            }
        )

    return metrics


def calculate_sciworld_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate raw-score and full-completion ScienceWorld metrics."""
    grouped = group_trajectories(records, expected_benchmark="sciworld")
    n_trajs = len(next(iter(grouped.values())))

    rewards_by_task = [
        [
            safe_float(trajectory.get("reward")) for trajectory in trajectories
        ]
        for trajectories in grouped.values()
    ]

    successes_by_task = [
        [
            parse_bool(trajectory.get("success")) for trajectory in trajectories
        ]
        for trajectories in grouped.values()
    ]

    metrics: dict[str, Any] = {
        "benchmark": "sciworld",
        "num_tasks": len(grouped),
        "n_trajs": n_trajs,
        "num_trajectories": len(records),
        "reward": {
            "first": mean(rewards[0] for rewards in rewards_by_task),
            "last": mean(rewards[-1] for rewards in rewards_by_task),
            "mean_of_n": mean(mean(rewards) for rewards in rewards_by_task),
            "best_of_n": mean(max(rewards) for rewards in rewards_by_task),
        },
        "full_success": {
            "first": mean(float(successes[0]) for successes in successes_by_task),
            "last": mean(float(successes[-1]) for successes in successes_by_task),
            "mean_of_n": mean(mean(float(success) for success in successes) for successes in successes_by_task),
            "best_of_n": mean(float(any(successes)) for successes in successes_by_task),
        },
        "by_attempt": [],
        "best_of_k": [],
    }

    for attempt_id in range(n_trajs):
        attempt_rewards = [
            rewards[attempt_id] for rewards in rewards_by_task
        ]

        attempt_successes = [
            successes[attempt_id] for successes in successes_by_task
        ]

        success_count = sum(int(success) for success in attempt_successes)

        metrics["by_attempt"].append(
            {
                "attempt_id": attempt_id,
                "avg_reward": mean(attempt_rewards),
                "full_success_rate": success_count / len(attempt_successes),
                "successes": success_count,
                "num_tasks": len(attempt_successes),
            }
        )

    for k in range(1, n_trajs + 1):
        success_count = sum(
            int(any(successes[:k])) for successes in successes_by_task
        )

        metrics["best_of_k"].append(
            {
                "k": k,
                "avg_reward": mean(
                    max(rewards[:k]) for rewards in rewards_by_task
                ),
                "full_success_rate": success_count / len(successes_by_task),
                "successes": success_count,
                "num_tasks": len(successes_by_task),
            }
        )

    return metrics


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    """Save metrics in a machine-readable form."""
    Path(path).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
