"""Text repository for database operations."""

from datetime import datetime, timezone
from typing import List, Optional

from src.models.candidate import Candidate
from src.models.text import Text
from src.repositories.base import BaseRepository
from src.utils.database import get_connection
from src.utils.logger import logger


class TextRepository(BaseRepository):
    """Repository for text operations."""
    
    def create(
        self,
        text: str,
        language: Optional[str],
        clusters: Optional[str],
        assigned_cluster: Optional[str],
        data_version: int,
        candidates: List[Candidate],
        model_version: int,
        assigned_to: Optional[str] = None
    ) -> int:
        """Create a new text with candidates.
        
        Args:
            text: Text content
            language: Text language
            clusters: Associated clusters
            assigned_cluster: Primary cluster
            data_version: Data version
            candidates: List of ML candidates
            model_version: Model version
            assigned_to: Pre-assigned annotator
            
        Returns:
            ID of created text
        """
        with get_connection(self.db_path) as conn:
            # Check for existing text first to return its ID if ignored
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO texts (text, language, clusters, assigned_cluster, assigned_to, data_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (text, language, clusters, assigned_cluster, assigned_to, data_version, datetime.now(timezone.utc).isoformat())
            )
            
            text_id = cursor.lastrowid
            
            # If nothing inserted (row existed), fetch existing ID
            if cursor.rowcount == 0:
                row = conn.execute("SELECT id FROM texts WHERE text = ?", (text,)).fetchone()
                if row:
                    return row["id"]
                return 0 # Should not happen with UNIQUE constraint
            
            # Insert candidates only for new text
            for candidate in candidates:
                conn.execute(
                    """
                    INSERT INTO candidates (text_id, label, rank, probability, model_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        text_id,
                        candidate.label,
                        candidate.rank,
                        candidate.probability,
                        model_version,
                        datetime.now(timezone.utc).isoformat()
                    )
                )
            
            conn.commit()
            logger.debug(f"Created text#{text_id} with {len(candidates)} candidates (assigned_to: {assigned_to})")
            return text_id
    
    def get_by_id(self, text_id: int) -> Optional[Text]:
        """Get text by ID.
        
        Args:
            text_id: Text ID
            
        Returns:
            Text if found, None otherwise
        """
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT 
                    id, 
                    text, 
                    language, 
                    clusters, 
                    assigned_cluster, 
                    assigned_to,
                    data_version, 
                    created_at 
                FROM texts WHERE id = ?
                """, 
                (text_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return Text(
                id=row["id"],
                text=row["text"],
                language=row["language"],
                clusters=row["clusters"],
                assigned_cluster=row["assigned_cluster"],
                assigned_to=row["assigned_to"],
                data_version=row["data_version"],
                created_at=row["created_at"]
            )
    
    def exists(self, text: str) -> bool:
        """Check if text already exists in database.
        
        Args:
            text: Text content
            
        Returns:
            True if exists, False otherwise
        """
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT id FROM texts WHERE text = ?", (text,)).fetchone()
            return row is not None
    
    def get_candidates(self, text_id: int) -> List[Candidate]:
        """Get candidates for a text.
        
        Args:
            text_id: Text ID
            
        Returns:
            List of candidates ordered by rank
        """
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE text_id = ? ORDER BY rank ASC",
                (text_id,)
            ).fetchall()
            
            return [
                Candidate(
                    label=row["label"],
                    rank=row["rank"],
                    probability=row["probability"]
                )
                for row in rows
            ]
    
    def get_unannotated(
        self,
        annotator: str,
        clusters: Optional[List[str]] = None,
        intents: Optional[List[str]] = None,
        language: Optional[str] = None,
        show_skipped: bool = False
    ) -> List[dict]:
        """Get texts that need annotation.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            intents: Filter by intents (candidates)
            language: Filter by language
            show_skipped: Show only skipped texts
            
        Returns:
            List of text rows with metadata
        """
        with get_connection(self.db_path) as conn:
            if show_skipped:
                base_query = """
                    SELECT
                        t.id,
                        t.text,
                        t.language,
                        t.clusters,
                        t.assigned_cluster,
                        t.data_version,
                        t.created_at,
                        COUNT(DISTINCT a.annotator) as annotators
                    FROM texts t
                    LEFT JOIN annotations a ON a.text_id = t.id
                """
                params = []
                filters = [
                    "t.is_skipped = 1",
                    "NOT EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id AND a2.annotator = ?)"
                ]
                params.append(annotator)
            else:
                base_query = """
                    SELECT
                        t.id,
                        t.text,
                        t.language,
                        t.clusters,
                        t.assigned_cluster,
                        t.assigned_to,
                        t.data_version,
                        t.created_at,
                        COUNT(DISTINCT a.annotator) as annotators
                    FROM texts t
                    LEFT JOIN annotations a ON a.text_id = t.id
                """
                params = []
                filters = [
                    "NOT EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id AND a2.annotator = ?)",
                    "t.is_skipped = 0"
                ]
                params.append(annotator)
            
            # ASSIGNMENT LOGIC:
            # 1. If assigned_to is set, it MUST match annotator
            # 2. If assigned_to is NULL, use Cluster/Language/Intent logic
            
            # Combine into a complex OR clause:
            # (assigned_to = ? OR (assigned_to IS NULL AND [filters]))
            
            assignment_clause = "(t.assigned_to = ?"
            assignment_params = [annotator]
            
            fallback_filters = []
            
            # Add cluster filter to fallback
            if clusters:
                placeholders = ", ".join("?" for _ in clusters)
                fallback_filters.append(f"t.assigned_cluster IN ({placeholders})")
                assignment_params.extend(clusters)
            
            # Add intent filter to fallback
            if intents:
                placeholders = ", ".join("?" for _ in intents)
                fallback_filters.append(f"EXISTS (SELECT 1 FROM candidates c WHERE c.text_id = t.id AND c.label IN ({placeholders}))")
                assignment_params.extend(intents)
            
            # Add language filter to fallback
            if language:
                fallback_filters.append("(t.language = ? OR t.language IS NULL)")
                assignment_params.append(language)
                
            if fallback_filters:
                assignment_clause += f" OR (t.assigned_to IS NULL AND {' AND '.join(fallback_filters)})"
            else:
                 # If no fallback filters, assume all unassigned are visible? 
                 # Or strict? Usually there are filters. If none, allow all unassigned.
                assignment_clause += " OR t.assigned_to IS NULL"
                
            assignment_clause += ")"
            
            filters.append(assignment_clause)
            params.extend(assignment_params)
            
            # Build final query
            where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
            query = f"{base_query} {where_clause} GROUP BY t.id HAVING COUNT(DISTINCT a.annotator) < ? ORDER BY COUNT(DISTINCT a.annotator) DESC, t.assigned_cluster, t.created_at DESC"
            
            return conn.execute(query, params + [1]).fetchall()

    def get_all_texts_for_annotator(
        self,
        annotator: str,
        clusters: Optional[List[str]] = None,
        intents: Optional[List[str]] = None,
        language: Optional[str] = None,
        candidate_label: Optional[str] = None,
        candidate_threshold: float = 0.0,
        candidate_by_cluster: bool = False,
        uncategorized_threshold: Optional[float] = None
    ) -> List[dict]:
        """Get all texts with status for an annotator.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            intents: Filter by intents (candidates)
            language: Filter by language
            
        Returns:
            List of dicts with id, request_text, is_annotated
        """
        with get_connection(self.db_path) as conn:
            base_query = """
                SELECT
                    t.id,
                    t.text,
                    t.assigned_to,
                    t.is_skipped,
                    CASE
                        WHEN EXISTS (SELECT 1 FROM annotations a WHERE a.text_id = t.id AND a.annotator = ?) THEN 1
                        ELSE 0
                    END as is_annotated
                FROM texts t
            """
            params = [annotator]
            filters = []
            
            # ASSIGNMENT LOGIC (Same as above)
            assignment_clause = "(t.assigned_to = ?"
            assignment_params = [annotator]
            
            fallback_filters = []
            
            # Add cluster filter
            if clusters:
                placeholders = ", ".join("?" for _ in clusters)
                fallback_filters.append(f"t.assigned_cluster IN ({placeholders})")
                assignment_params.extend(clusters)
            
            # Add intent filter to fallback
            if intents:
                placeholders = ", ".join("?" for _ in intents)
                fallback_filters.append(f"EXISTS (SELECT 1 FROM candidates c WHERE c.text_id = t.id AND c.label IN ({placeholders}))")
                assignment_params.extend(intents)
            
            # Add language filter
            if language:
                fallback_filters.append("(t.language = ? OR t.language IS NULL)")
                assignment_params.append(language)
            
            if fallback_filters:
                assignment_clause += f" OR (t.assigned_to IS NULL AND {' AND '.join(fallback_filters)})"
            else:
                assignment_clause += " OR t.assigned_to IS NULL"
                
            assignment_clause += ")"
            filters.append(assignment_clause)
            params.extend(assignment_params)

            # Candidate content filter (applies to ALL texts, not just assignment)
            if candidate_label:
                if candidate_by_cluster:
                    filters.append("""
                        EXISTS (
                            SELECT 1 FROM candidates c
                            JOIN intents i ON i.label = c.label
                            WHERE c.text_id = t.id
                              AND i.cluster = ?
                              AND c.probability >= ?
                        )
                    """)
                else:
                    filters.append("""
                        EXISTS (
                            SELECT 1 FROM candidates c
                            WHERE c.text_id = t.id
                              AND c.label = ?
                              AND c.probability >= ?
                        )
                    """)
                params.extend([candidate_label, candidate_threshold])

            # Uncategorized filter: texts with no candidate above threshold
            if uncategorized_threshold is not None:
                filters.append("""
                    NOT EXISTS (
                        SELECT 1 FROM candidates c
                        WHERE c.text_id = t.id AND c.probability >= ?
                    )
                """)
                params.append(uncategorized_threshold)

            where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
            query = f"{base_query} {where_clause} ORDER BY t.id ASC"
            
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_label_stats(
        self,
        annotator: str,
        labels: List[str],
        threshold: float,
        by_cluster: bool = False
    ) -> List[dict]:
        """Get per-intent or per-cluster stats for an annotator.

        Counts texts where:
          - assigned_to = annotator
          - at least one candidate matches the label (or cluster) with probability >= threshold

        Args:
            annotator: Annotator name
            labels: List of intent labels or cluster names
            threshold: Minimum candidate probability to count
            by_cluster: If True, labels are cluster names; if False, they are intent labels

        Returns:
            List of dicts: {label, total, annotated}
        """
        results = []
        with get_connection(self.db_path) as conn:
            for label in labels:
                if by_cluster:
                    match_cond = """
                        EXISTS (
                            SELECT 1 FROM candidates c
                            JOIN intents i ON i.label = c.label
                            WHERE c.text_id = t.id
                              AND i.cluster = ?
                              AND c.probability >= ?
                        )
                    """
                else:
                    match_cond = """
                        EXISTS (
                            SELECT 1 FROM candidates c
                            WHERE c.text_id = t.id
                              AND c.label = ?
                              AND c.probability >= ?
                        )
                    """

                base = f"""
                    FROM texts t
                    WHERE t.assigned_to = ?
                      AND {match_cond}
                """
                params = [annotator, label, threshold]

                total = conn.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]
                annotated = conn.execute(
                    f"""SELECT COUNT(*) {base}
                        AND EXISTS (
                            SELECT 1 FROM annotations a
                            WHERE a.text_id = t.id AND a.annotator = ?
                        )""",
                    params + [annotator]
                ).fetchone()[0]

                results.append({"label": label, "total": total, "annotated": annotated})

        return results

    def get_assigned_labels(
        self,
        annotator: str,
        by_cluster: bool = False,
        threshold: float = 0.0
    ) -> List[str]:
        """Get all distinct cluster or intent labels for texts assigned to an annotator.

        Args:
            annotator: Annotator name
            by_cluster: If True, return cluster names; if False, intent labels
            threshold: Minimum candidate probability

        Returns:
            Sorted list of label strings
        """
        with get_connection(self.db_path) as conn:
            if by_cluster:
                rows = conn.execute("""
                    SELECT DISTINCT i.cluster AS label
                    FROM texts t
                    JOIN candidates c ON c.text_id = t.id
                    JOIN intents i ON i.label = c.label
                    WHERE t.assigned_to = ?
                      AND c.probability >= ?
                    ORDER BY label
                """, (annotator, threshold)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT DISTINCT c.label
                    FROM texts t
                    JOIN candidates c ON c.text_id = t.id
                    WHERE t.assigned_to = ?
                      AND c.probability >= ?
                    ORDER BY label
                """, (annotator, threshold)).fetchall()
            return [r["label"] for r in rows]

    def get_uncategorized_count(
        self,
        annotator: str,
        threshold: float = 0.0
    ) -> dict:
        """Count texts assigned to an annotator with no candidate above threshold.

        Returns:
            Dict with 'total' and 'annotated' counts
        """
        with get_connection(self.db_path) as conn:
            base = """
                FROM texts t
                WHERE t.assigned_to = ?
                  AND NOT EXISTS (
                    SELECT 1 FROM candidates c
                    WHERE c.text_id = t.id AND c.probability >= ?
                  )
            """
            params = [annotator, threshold]
            total = conn.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]
            annotated = conn.execute(
                f"""SELECT COUNT(*) {base}
                    AND EXISTS (
                        SELECT 1 FROM annotations a
                        WHERE a.text_id = t.id AND a.annotator = ?
                    )""",
                params + [annotator]
            ).fetchone()[0]
            return {"total": total, "annotated": annotated}
