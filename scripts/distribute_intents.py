"""Distribute intents randomly and evenly between annotators.

Rules:
- 5% of intents overlap (assigned to 2+ annotators)
- 2.5% of intents are not assigned to anyone
- Remaining intents are distributed evenly (1 annotator each)

Usage:
    python scripts/distribute_intents.py [--seed 42] [--dry-run]
"""

import argparse
import random
import sys
from pathlib import Path

import yaml


def load_intents(intents_dir: Path) -> list[str]:
    """Load all intent labels from YAML files in the intents directory."""
    labels = []
    for yaml_file in sorted(intents_dir.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data:
                labels.extend(data.keys())
    return sorted(set(labels))


def load_annotators(path: Path) -> list[dict]:
    """Load annotators from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("annotators", [])


def distribute(
    intents: list[str],
    annotators: list[dict],
    overlap_pct: float = 0.05,
    unassigned_pct: float = 0.025,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Distribute intents across annotators.

    Returns:
        dict mapping annotator name -> list of assigned intent labels
    """
    rng = random.Random(seed)

    n = len(intents)
    n_unassigned = max(1, round(n * unassigned_pct))
    n_overlap = max(1, round(n * overlap_pct))
    n_single = n - n_unassigned - n_overlap

    shuffled = intents[:]
    rng.shuffle(shuffled)

    unassigned = shuffled[:n_unassigned]
    overlap_intents = shuffled[n_unassigned : n_unassigned + n_overlap]
    single_intents = shuffled[n_unassigned + n_overlap :]

    names = [a["name"] for a in annotators]
    assignment: dict[str, list[str]] = {name: [] for name in names}

    # Distribute single-assignment intents evenly (round-robin)
    for i, intent in enumerate(single_intents):
        assignee = names[i % len(names)]
        assignment[assignee].append(intent)

    # Distribute overlapping intents to 2 random annotators each
    for intent in overlap_intents:
        pair = rng.sample(names, k=2)
        for name in pair:
            assignment[name].append(intent)

    # Sort each annotator's list for readability
    for name in names:
        assignment[name].sort()

    return assignment, sorted(unassigned), sorted(overlap_intents)


def write_annotators(path: Path, annotators: list[dict], assignment: dict[str, list[str]]):
    """Write updated annotators.yaml with intent assignments."""
    for ann in annotators:
        name = ann["name"]
        ann["intents"] = assignment.get(name, [])

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            {"annotators": annotators},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Distribute intents between annotators")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Print distribution without writing")
    parser.add_argument("--overlap", type=float, default=0.05, help="Overlap percentage (default: 0.05)")
    parser.add_argument("--unassigned", type=float, default=0.025, help="Unassigned percentage (default: 0.025)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    intents_dir = project_root / "data" / "intents"
    annotators_path = project_root / "data" / "annotators.yaml"

    intents = load_intents(intents_dir)
    annotators = load_annotators(annotators_path)

    print(f"Total intents: {len(intents)}")
    print(f"Annotators: {', '.join(a['name'] for a in annotators)}")
    print(f"Overlap: {args.overlap:.1%}, Unassigned: {args.unassigned:.1%}")
    print()

    assignment, unassigned, overlap_intents = distribute(
        intents, annotators,
        overlap_pct=args.overlap,
        unassigned_pct=args.unassigned,
        seed=args.seed,
    )

    # Print summary
    for name, assigned in assignment.items():
        unique = [i for i in assigned if i not in overlap_intents]
        shared = [i for i in assigned if i in overlap_intents]
        print(f"  {name}: {len(assigned)} intents ({len(unique)} unique + {len(shared)} shared)")

    print(f"\n  Overlapping ({len(overlap_intents)}): {', '.join(overlap_intents)}")
    print(f"  Unassigned  ({len(unassigned)}): {', '.join(unassigned)}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    write_annotators(annotators_path, annotators, assignment)
    print(f"\nWritten to {annotators_path}")


if __name__ == "__main__":
    main()
