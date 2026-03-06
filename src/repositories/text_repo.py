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
        shown_cluster: Optional[str] = None,
        shown_uncategorized: bool = False,
        shown_top_k: int = 5,
        shown_threshold: float = 0.0,
        shown_annotator_intents: Optional[List[str]] = None,
        shown_annotator_clusters: Optional[List[str]] = None,
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

            # Shown-candidate cluster filter
            if shown_cluster or shown_uncategorized:
                shown_cond, shown_params = self._shown_candidate_condition(
                    shown_top_k, shown_threshold,
                    shown_annotator_intents, shown_annotator_clusters
                )
                if shown_cluster:
                    filters.append(f"""
                        EXISTS (
                            SELECT 1 FROM candidates c
                            JOIN intents i ON i.label = c.label
                            WHERE c.text_id = t.id
                              AND i.cluster = ?
                              AND {shown_cond}
                        )
                    """)
                    params.append(shown_cluster)
                    params.extend(shown_params)
                elif shown_uncategorized:
                    filters.append(f"""
                        NOT EXISTS (
                            SELECT 1 FROM candidates c
                            JOIN intents i ON i.label = c.label
                            WHERE c.text_id = t.id
                              AND {shown_cond}
                        )
                    """)
                    params.extend(shown_params)

            where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
            query = f"{base_query} {where_clause} ORDER BY t.id ASC"
            
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def _shown_candidate_condition(
        self,
        top_k: int,
        threshold: float,
        annotator_intents: Optional[List[str]] = None,
        annotator_clusters: Optional[List[str]] = None,
    ) -> tuple:
        """Build SQL condition + params for 'shown candidates'.

        A candidate is shown if:
          rank <= top_k OR label IN annotator_intents OR cluster IN annotator_clusters
        AND probability > threshold.

        Returns (sql_fragment, params) — the fragment uses aliases c (candidates) and i (intents).
        """
        union_parts = ["c.rank <= ?"]
        union_params: list = [top_k]

        if annotator_intents:
            ph = ", ".join("?" for _ in annotator_intents)
            union_parts.append(f"c.label IN ({ph})")
            union_params.extend(annotator_intents)

        if annotator_clusters:
            ph = ", ".join("?" for _ in annotator_clusters)
            union_parts.append(f"i.cluster IN ({ph})")
            union_params.extend(annotator_clusters)

        union_cond = " OR ".join(union_parts)
        sql = f"c.probability > ? AND ({union_cond})"
        params = [threshold] + union_params
        return sql, params

    def get_shown_label_stats(
        self,
        annotator: str,
        top_k: int,
        threshold: float,
        annotator_intents: Optional[List[str]] = None,
        annotator_clusters: Optional[List[str]] = None,
        by_cluster: bool = True
    ) -> List[dict]:
        """Get per-cluster (or per-intent) stats based on shown candidates.

        A candidate is 'shown' if rank <= top_k OR in annotator's intents/clusters.
        Only shown candidates with probability > threshold are counted.
        A text may appear in multiple groups.

        Returns:
            List of dicts: {label, total, annotated}
        """
        shown_cond, shown_params = self._shown_candidate_condition(
            top_k, threshold, annotator_intents, annotator_clusters
        )
        group_col = "i.cluster" if by_cluster else "c.label"

        with get_connection(self.db_path) as conn:
            rows = conn.execute(f"""
                SELECT
                    {group_col} AS label,
                    COUNT(DISTINCT t.id) AS total,
                    COUNT(DISTINCT CASE WHEN EXISTS (
                        SELECT 1 FROM annotations a
                        WHERE a.text_id = t.id AND a.annotator = ?
                    ) THEN t.id END) AS annotated
                FROM texts t
                JOIN candidates c ON c.text_id = t.id
                JOIN intents i ON i.label = c.label
                WHERE t.assigned_to = ?
                  AND {shown_cond}
                GROUP BY {group_col}
                ORDER BY label
            """, [annotator, annotator] + shown_params).fetchall()
            return [dict(r) for r in rows]

    def get_shown_uncategorized_count(
        self,
        annotator: str,
        top_k: int,
        threshold: float,
        annotator_intents: Optional[List[str]] = None,
        annotator_clusters: Optional[List[str]] = None,
    ) -> dict:
        """Count texts with no shown candidate above threshold.

        Returns:
            Dict with 'total' and 'annotated' counts
        """
        shown_cond, shown_params = self._shown_candidate_condition(
            top_k, threshold, annotator_intents, annotator_clusters
        )

        with get_connection(self.db_path) as conn:
            row = conn.execute(f"""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN EXISTS (
                        SELECT 1 FROM annotations a
                        WHERE a.text_id = t.id AND a.annotator = ?
                    ) THEN 1 ELSE 0 END) AS annotated
                FROM texts t
                WHERE t.assigned_to = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM candidates c
                      JOIN intents i ON i.label = c.label
                      WHERE c.text_id = t.id
                        AND {shown_cond}
                  )
            """, [annotator, annotator] + shown_params).fetchone()
            return {"total": row["total"], "annotated": row["annotated"] or 0}
