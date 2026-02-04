#!/usr/bin/env python3
"""
Script to compare intents between CSV columns and YAML files.
"""

import csv
import yaml
from pathlib import Path


def get_intents_from_yaml(yaml_dir: Path) -> dict[str, set[str]]:
    """
    Read all intents from YAML files in the given directory.
    Returns a dict mapping filename to set of intents.
    """
    intents_by_file = {}
    all_intents = set()

    for yaml_file in yaml_dir.glob("*.yaml"):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data:
                intents = set(data.keys())
                intents_by_file[yaml_file.name] = intents
                all_intents.update(intents)

    return intents_by_file, all_intents


def get_intents_from_csv(csv_path: Path) -> set[str]:
    """
    Read intent names from CSV column headers.
    Skips first 3 columns (request_text, language, lang_prob).
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        # Skip first 3 columns: request_text, language, lang_prob
        intents = set(headers[3:])
    return intents


def main():
    # Paths
    project_root = Path(__file__).parent
    yaml_dir = project_root / "data" / "intents"
    csv_path = project_root / "data" / "raw_texts_with_preds" / "nlu_records_20260127_filtered_selected.csv"

    # Get intents from both sources
    print("Reading intents from YAML files...")
    yaml_intents_by_file, yaml_intents_all = get_intents_from_yaml(yaml_dir)

    print("Reading intents from CSV file...")
    csv_intents = get_intents_from_csv(csv_path)

    # Compare
    only_in_yaml = yaml_intents_all - csv_intents
    only_in_csv = csv_intents - yaml_intents_all
    in_both = yaml_intents_all & csv_intents

    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\nTotal intents in YAML files: {len(yaml_intents_all)}")
    print(f"Total intents in CSV file: {len(csv_intents)}")
    print(f"Intents in both: {len(in_both)}")

    print("\n" + "-" * 60)
    print(f"INTENTS ONLY IN YAML ({len(only_in_yaml)}):")
    print("-" * 60)
    if only_in_yaml:
        for intent in sorted(only_in_yaml):
            # Find which file contains this intent
            source_files = [f for f, intents in yaml_intents_by_file.items() if intent in intents]
            print(f"  - {intent} (in: {', '.join(source_files)})")
    else:
        print("  (none)")

    print("\n" + "-" * 60)
    print(f"INTENTS ONLY IN CSV ({len(only_in_csv)}):")
    print("-" * 60)
    if only_in_csv:
        for intent in sorted(only_in_csv):
            print(f"  - {intent}")
    else:
        print("  (none)")

    print("\n" + "-" * 60)
    print("YAML FILES SUMMARY:")
    print("-" * 60)
    for filename, intents in sorted(yaml_intents_by_file.items()):
        missing = intents - csv_intents
        print(f"  {filename}: {len(intents)} intents ({len(missing)} missing in CSV)")


if __name__ == "__main__":
    main()
