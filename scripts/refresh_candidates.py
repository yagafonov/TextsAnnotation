"""
One-off script: replace stub-generated candidates with real CSV scores.

Preserves all texts, annotations, and skipped_texts.
Only deletes and re-inserts the candidates table rows.

Usage:
    python scripts/refresh_candidates.py
"""

import csv
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import settings

CSV_PATH = os.environ.get("CSV_PATH", "data/requests.csv")
DB_PATH = settings.db_path
PROBABILITY_THRESHOLD = settings.probability_threshold


def load_intent_labels(db_path: str) -> set:
    """Return all intent labels known to the DB (from candidates or intents table)."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT label FROM candidates").fetchall()
    return {r[0] for r in rows}


def candidates_from_row(row: dict, intent_labels: set, threshold: float) -> list:
    """Extract (label, probability) pairs from a CSV row for known intent columns."""
    scores = []
    for label in intent_labels:
        if label in row:
            try:
                prob = float(row[label])
                if prob >= threshold:
                    scores.append((label, prob))
            except (ValueError, TypeError):
                continue
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        sys.exit(1)

    print(f"DB:  {DB_PATH}")
    print(f"CSV: {CSV_PATH}")
    print(f"Probability threshold: {PROBABILITY_THRESHOLD}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Build text -> text_id lookup
        print("Loading existing texts from DB...")
        rows = conn.execute("SELECT id, text FROM texts").fetchall()
        text_to_id = {r["text"]: r["id"] for r in rows}
        print(f"  {len(text_to_id)} texts in DB")

        # Determine intent labels from CSV header
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_columns = set(reader.fieldnames or [])

        # Get labels that exist in both DB candidates and CSV columns
        db_labels = load_intent_labels(DB_PATH)
        intent_labels = db_labels & csv_columns
        print(f"  {len(intent_labels)} intent labels found in both DB and CSV")

        if not intent_labels:
            print("ERROR: No matching intent columns between DB and CSV. Aborting.")
            sys.exit(1)

        updated = 0
        skipped_no_match = 0
        skipped_no_scores = 0
        now = datetime.now(timezone.utc).isoformat()

        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("request_text", "").strip()
                if not text:
                    continue

                text_id = text_to_id.get(text)
                if text_id is None:
                    skipped_no_match += 1
                    continue

                scores = candidates_from_row(row, intent_labels, PROBABILITY_THRESHOLD)
                if not scores:
                    skipped_no_scores += 1
                    continue

                # Delete old candidates for this text
                conn.execute("DELETE FROM candidates WHERE text_id = ?", (text_id,))

                # Insert new candidates
                for rank, (label, prob) in enumerate(scores, start=1):
                    conn.execute(
                        """
                        INSERT INTO candidates (text_id, label, rank, probability, model_version, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (text_id, label, rank, prob, 0, now)
                    )

                updated += 1
                if updated % 500 == 0:
                    print(f"  ...{updated} texts refreshed")

        conn.commit()

    print(f"\nDone.")
    print(f"  Texts refreshed:          {updated}")
    print(f"  Skipped (not in DB):      {skipped_no_match}")
    print(f"  Skipped (no scores ≥ {PROBABILITY_THRESHOLD}): {skipped_no_scores}")


if __name__ == "__main__":
    main()
