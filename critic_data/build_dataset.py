import argparse
import copy
import json
import random

from pathlib import Path
from typing import Any, Dict, List


def load_tree_records(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all worker tree files.

    Expected files:
        worker_0_trees.jsonl
        worker_1_trees.jsonl
        ...
    """
    tree_files = sorted(input_dir.glob("worker_*_trees.jsonl"))

    if not tree_files:
        raise FileNotFoundError(
            f"No worker_*_trees.jsonl files found in {input_dir}"
        )

    records: List[Dict[str, Any]] = []

    seen_task_ids = set()

    for path in tree_files:
        with path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_num}") from exc

                task_id = str(record["task_id"])

                if task_id in seen_task_ids:
                    raise ValueError(
                        f"Duplicate task id across worker files: {task_id!r}"
                    )

                seen_task_ids.add(task_id)

                records.append(record)

    return records


def set_vanilla_rewards(node: Dict[str, Any]) -> None:
    """
    Reproduce QLASS 'vanilla' reward preprocessing.

    All internal nodes get reward=0.
    Leaves keep their terminal environment reward.
    """
    children = node.get("children", [])

    if not children:
        return

    node["reward"] = 0.0

    for child in children:
        set_vanilla_rewards(child)


def compute_q_values(node: Dict[str, Any], gamma: float) -> float:
    """
    Propagate Q-values backwards through the tree.

    Leaf:
        Q = reward

    Internal node:
        Q = reward + gamma * max(Q(child))

    After set_vanilla_rewards(), internal reward is normally zero.
    """
    children = node.get("children", [])

    reward = float(node.get("reward", 0.0) or 0.0)

    if not children:
        q_value = reward

    else:
        child_q_values = [
            compute_q_values(child, gamma) for child in children
        ]

        q_value = reward + gamma * max(child_q_values)

    node["q_value"] = float(q_value)

    return float(q_value)


def collect_q_examples(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract non-root critic training samples.

    Every action node stores exactly:
        critic_state  -> human/user message before the action
        critic_action -> actor response/action
        q_value       -> training target
    """
    examples: List[Dict[str, Any]] = []

    def visit(node: Dict[str, Any]) -> None:
        critic_state = node.get("critic_state")

        critic_action = node.get("critic_action")

        q_value = node.get("q_value")

        if critic_state is not None and critic_action is not None and q_value is not None:
            examples.append(
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": str(critic_state),
                        },
                        {
                            "from": "gpt",
                            "value": str(critic_action),
                        },
                    ],
                    "label": float(q_value),
                }
            )

        for child in node.get("children", []):
            visit(child)

    # Do not create an example for the artificial root.
    for child in root.get("children", []):
        visit(child)

    return examples


def normalize_labels(examples: List[Dict[str, Any]]) -> None:
    """
    Min-max normalize Q labels INSIDE ONE TASK.

    This deliberately happens after optional per-task sampling because that is
    the order used by the current QLASS vanilla dataset builder.
    """
    if not examples:
        return

    values = [
        float(example["label"]) for example in examples
    ]

    min_value = min(values)

    max_value = max(values)

    for example in examples:
        if max_value > min_value:
            example["label"] = (float(example["label"]) - min_value) / (max_value - min_value)

        else:
            # Same legacy behavior as QLASS normalize_data().
            example["label"] = 0.0


def build_dataset(
    records: List[Dict[str, Any]],
    gamma: float,
    upper_num: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Convert all task trees into one QNet training dataset.

    Sampling and normalization are both performed independently per task.
    """
    rng = random.Random(seed)

    dataset: List[Dict[str, Any]] = []

    total_before_sampling = 0

    for record in records:
        task_id = str(record["task_id"])

        # Do not mutate the tree record loaded from disk.
        root = copy.deepcopy(record["root"])

        # 1. Keep rewards only at leaves.
        set_vanilla_rewards(root)

        # 2. Propagate Q-values backwards.
        compute_q_values(root, gamma=gamma)

        # 3. Convert tree nodes into state-action examples.
        task_examples = collect_q_examples(root)

        total_before_sampling += len(task_examples)

        # 4. Old QLASS limits the number of examples independently per task.
        if (
            upper_num is not None
            and len(task_examples)
            > upper_num
        ):
            task_examples = rng.sample(task_examples, upper_num)

        # 5. Important:
        # normalize only AFTER sampling and only within this task.
        normalize_labels(task_examples)

        for example in task_examples:
            dataset.append(
                {
                    "id": task_id,
                    **example,
                }
            )

    print(
        f"[Q_DATA] tasks={len(records)} "
        f"examples_before_sampling={total_before_sampling} "
        f"examples={len(dataset)} "
        f"avg_examples_per_task="
        f"{len(dataset) / max(len(records), 1):.2f}"
    )

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert critic search trees into QNet training examples."
    )

    parser.add_argument(
        "--input-dir", required=True,
        help="Collection directory containing worker_*_trees.jsonl.",
    )

    parser.add_argument("--output", required=True, help="Output JSON dataset path.")

    parser.add_argument("--gamma", type=float, default=0.9)

    parser.add_argument(
        "--upper-num", type=int, default=300,
        help="Maximum number of examples retained " "from each task tree.",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if args.upper_num <= 0:
        raise ValueError("upper_num must be positive")

    input_dir = Path(args.input_dir)

    output_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    records = load_tree_records(input_dir)

    print(f"[Q_DATA] loaded task trees: {len(records)}")

    dataset = build_dataset(
        records=records,
        gamma=args.gamma,
        upper_num=args.upper_num,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"[Q_DATA] saved: {output_path}")


if __name__ == "__main__":
    main()
