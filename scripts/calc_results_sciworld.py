#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from agent_eval.results import (
    calculate_sciworld_metrics,
    load_trajectories,
    save_metrics,
)


PAPER_SCALE = 100.0


def format_reward(value: float) -> str:
    """Show both raw ScienceWorld score and the usual 0..100 display scale."""
    return f"{value:.6f} (paper scale: {value * PAPER_SCALE:.2f})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate ScienceWorld metrics from trajectories.jsonl."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Run directory or path to trajectories.jsonl.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for machine-readable metrics JSON.",
    )

    args = parser.parse_args()

    records = load_trajectories(args.input)
    metrics = calculate_sciworld_metrics(records)

    n_trajs = metrics["n_trajs"]
    reward = metrics["reward"]
    success = metrics["full_success"]

    print("=== ScienceWorld summary ===")
    print(f"num tasks: {metrics['num_tasks']}")
    print(f"n_trajs: {n_trajs}")
    print(f"num trajectories: {metrics['num_trajectories']}")

    print("\n=== Reward metrics ===")
    print(f"first_trajectory_avg_reward: {format_reward(reward['first'])}")
    print(f"last_trajectory_avg_reward:  {format_reward(reward['last'])}")
    print(f"mean_of_{n_trajs}_avg_reward: {format_reward(reward['mean_of_n'])}")
    print(f"best_of_{n_trajs}_avg_reward: {format_reward(reward['best_of_n'])}")

    print("\n=== Full-success metrics ===")
    print(f"first_trajectory_success_rate: {success['first']:.6f}")
    print(f"last_trajectory_success_rate:  {success['last']:.6f}")
    print(f"mean_of_{n_trajs}_success_rate: {success['mean_of_n']:.6f}")
    print(f"best_of_{n_trajs}_success_rate: {success['best_of_n']:.6f}")

    print("\n=== Metrics by attempt ===")

    for item in metrics["by_attempt"]:
        print(
            f"attempt {item['attempt_id']}: "
            f"avg_reward={format_reward(item['avg_reward'])}, "
            f"full_success_rate={item['full_success_rate']:.6f} "
            f"({item['successes']}/{item['num_tasks']})"
        )

    print("\n=== Best-of-k curves ===")

    for item in metrics["best_of_k"]:
        print(
            f"best_of_{item['k']}: "
            f"avg_reward={format_reward(item['avg_reward'])}, "
            f"full_success_rate={item['full_success_rate']:.6f} "
            f"({item['successes']}/{item['num_tasks']})"
        )

    if args.output:
        save_metrics(args.output, metrics)
        print(f"\nSaved metrics to: {args.output}")


if __name__ == "__main__":
    main()
