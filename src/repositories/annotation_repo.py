"""Annotation repository for database operations."""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.models.text import Annotation
from src.repositories.base import BaseRepository
from src.utils.database import get_connection
from src.utils.logger import logger


class AnnotationRepository(BaseRepository):
    """Repository for annotation operations."""
    
    def save_annotations(
        self,
        text_id: int,
        annotator: str,
        decisions: Dict[str, str],
        candidate_labels: List[str],
        extra_labels: List[str],
        shown_intents_source: Dict[str, str]
    ) -> None:
        """Save annotations for a text.
        
        Args:
            text_id: Text ID
            annotator: Annotator name
            decisions: Dictionary of label -> decision
            candidate_labels: Labels that were candidates
            extra_labels: Extra labels added by annotator
            shown_intents_source: Source of each shown intent
        """
        with get_connection(self.db_path) as conn:
            now = datetime.now(timezone.utc).isoformat()
            
            # Save all decisions
            for label, decision in decisions.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO annotations (text_id, annotator, label, decision, is_candidate, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (text_id, annotator, label, decision, 1 if label in candidate_labels else 0, now)
                )
            
            # Save extra labels (all marked as "yes")
            for label in extra_labels:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO annotations (text_id, annotator, label, decision, is_candidate, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (text_id, annotator, label, "yes", 0, now)
                )
            
            # Save shown_intents tracking
            for label, source in shown_intents_source.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO shown_intents (text_id, annotator, label, source, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (text_id, annotator, label, source, now)
                )
            
            # Save extra labels as shown_intents
            for label in extra_labels:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO shown_intents (text_id, annotator, label, source, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (text_id, annotator, label, "extra", now)
                )

            # Clear skip status atomically within the same transaction
            conn.execute(
                "DELETE FROM skipped_texts WHERE text_id = ? AND annotator = ?",
                (text_id, annotator)
            )
            conn.execute("UPDATE texts SET is_skipped = 0 WHERE id = ?", (text_id,))

            conn.commit()
            logger.info(f"Saved {len(decisions) + len(extra_labels)} annotations for text#{text_id} by {annotator}")
    
    def skip_text(self, text_id: int, annotator: str) -> None:
        """Mark a text as skipped by annotator.

        Args:
            text_id: Text ID
            annotator: Annotator name
        """
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO skipped_texts (text_id, annotator, created_at)
                VALUES (?, ?, ?)
                """,
                (text_id, annotator, datetime.now(timezone.utc).isoformat())
            )
            conn.execute("UPDATE texts SET is_skipped = 1 WHERE id = ?", (text_id,))
            conn.commit()
            logger.info(f"Text#{text_id} skipped by {annotator}")

    def unskip_text(self, text_id: int, annotator: str) -> None:
        """Remove skip mark from a text.

        Args:
            text_id: Text ID
            annotator: Annotator name
        """
        with get_connection(self.db_path) as conn:
            conn.execute(
                "DELETE FROM skipped_texts WHERE text_id = ? AND annotator = ?",
                (text_id, annotator)
            )
            conn.execute("UPDATE texts SET is_skipped = 0 WHERE id = ?", (text_id,))
            conn.commit()
            logger.info(f"Text#{text_id} unskipped by {annotator}")
    
    def get_annotations_for_text(self, text_id: int) -> List[Annotation]:
        """Get all annotations for a text.
        
        Args:
            text_id: Text ID
            
        Returns:
            List of annotations
        """
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM annotations WHERE text_id = ?",
                (text_id,)
            ).fetchall()
            
            return [
                Annotation(
                    id=row["id"],
                    text_id=row["text_id"],
                    annotator=row["annotator"],
                    label=row["label"],
                    decision=row["decision"],
                    is_candidate=bool(row["is_candidate"]),
                    created_at=row["created_at"]
                )
                for row in rows
            ]
    
    def get_progress(
        self,
        annotator: str,
        clusters: Optional[List[str]] = None,
        intents: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> dict:
        """Get annotation progress for an annotator.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            intents: Filter by intents
            language: Filter by language
            
        Returns:
            Dictionary with total and done counts
        """
        with get_connection(self.db_path) as conn:
            # Total texts
            total_query = "SELECT COUNT(*) as cnt FROM texts t WHERE t.assigned_to = ?"
            total_params = [annotator]
            
            if clusters:
                placeholders = ", ".join("?" for _ in clusters)
                total_query += f" AND t.assigned_cluster IN ({placeholders})"
                total_params.extend(clusters)
            
            if intents:
                placeholders = ", ".join("?" for _ in intents)
                total_query += f" AND EXISTS (SELECT 1 FROM candidates c WHERE c.text_id = t.id AND c.label IN ({placeholders}))"
                total_params.extend(intents)
            
            if language:
                total_query += " AND (t.language = ? OR t.language IS NULL)"
                total_params.append(language)
            
            total = conn.execute(total_query, total_params).fetchone()["cnt"]
            
            # Done texts
            done_query = """
                SELECT COUNT(DISTINCT t.id) as cnt
                FROM texts t
                INNER JOIN annotations a ON a.text_id = t.id AND a.annotator = ?
                WHERE t.assigned_to = ?
            """
            done_params = [annotator, annotator]
            
            if clusters:
                placeholders = ", ".join("?" for _ in clusters)
                done_query += f" AND t.assigned_cluster IN ({placeholders})"
                done_params.extend(clusters)
            
            if intents:
                placeholders = ", ".join("?" for _ in intents)
                done_query += f" AND EXISTS (SELECT 1 FROM candidates c WHERE c.text_id = t.id AND c.label IN ({placeholders}))"
                done_params.extend(intents)
            
            if language:
                done_query += " AND (t.language = ? OR t.language IS NULL)"
                done_params.append(language)
            
            done = conn.execute(done_query, done_params).fetchone()["cnt"]
            
            return {"total": total, "done": done}
