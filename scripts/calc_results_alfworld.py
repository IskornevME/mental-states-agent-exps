#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from agent_eval.results import (
    calculate_alfworld_metrics,
    load_trajectories,
    save_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate ALFWorld metrics from trajectories.jsonl."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Run directory or path to trajectories.jsonl.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for readable metrics JSON.",
    )

    args = parser.parse_args()

    records = load_trajectories(args.input)
    metrics = calculate_alfworld_metrics(records)

    n_trajs = metrics["n_trajs"]

    print("=== ALFWorld summary ===")
    print(f"num tasks: {metrics['num_tasks']}")
    print(f"n_trajs: {n_trajs}")
    print(f"num trajectories: {metrics['num_trajectories']}")

    print()
    print(f"first_trajectory_avg_reward: {metrics['first_trajectory_avg_reward']:.6f}")
    print(f"last_trajectory_avg_reward:  {metrics['last_trajectory_avg_reward']:.6f}")
    print(f"best_of_{n_trajs}_avg_reward: {metrics['best_of_n_avg_reward']:.6f}")
    print(f"mean_of_{n_trajs}_avg_reward: {metrics['mean_of_n_avg_reward']:.6f}")

    print("\nsuccess_rate_by_attempt:")

    for item in metrics["success_rate_by_attempt"]:
        print(
            f"  attempt {item['attempt_id']}: {item['rate']:.6f} "
            f"({item['successes']}/{item['num_tasks']})"
        )

    if n_trajs > 1:
        print("\nrecovery_after_all_previous_failures:")

        for item in metrics["recovery_after_all_previous_failures"]:
            print(
                f"  attempt {item['attempt_id']}: {item['rate']:.6f} "
                f"({item['recovered']}/{item['eligible']})"
            )

        print("\nrecovery_after_previous_attempt_failure:")

        for item in metrics["recovery_after_previous_attempt_failure"]:
            print(
                f"  attempt {item['attempt_id']}: {item['rate']:.6f} "
                f"({item['recovered']}/{item['eligible']})"
            )

    print("\nbest_of_k curve:")

    for item in metrics["best_of_k"]:
        print(f"  best_of_{item['k']}: {item['avg_reward']:.6f}")

    if args.output:
        save_metrics(args.output, metrics)
        print(f"\nSaved metrics to: {args.output}")


if __name__ == "__main__":
    main()
