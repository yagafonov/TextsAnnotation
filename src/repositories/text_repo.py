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
            cursor = conn.execute(
                """
                INSERT INTO texts (text, language, clusters, assigned_cluster, assigned_to, data_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (text, language, clusters, assigned_cluster, assigned_to, data_version, datetime.now(timezone.utc).isoformat())
            )
            text_id = cursor.lastrowid
            
            # Insert candidates
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
            logger.info(f"Created text#{text_id} with {len(candidates)} candidates (assigned_to: {assigned_to})")
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
                    text as request_text, 
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
                text=row["request_text"],
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
        language: Optional[str] = None,
        min_annotators: int = 2,
        show_skipped: bool = False
    ) -> List[dict]:
        """Get texts that need annotation.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            language: Filter by language
            min_annotators: Minimum number of required annotators
            show_skipped: Show only skipped texts
            
        Returns:
            List of text rows with metadata
        """
        with get_connection(self.db_path) as conn:
            if show_skipped:
                base_query = """
                    SELECT
                        t.id,
                        t.text as request_text,
                        t.language,
                        t.clusters,
                        t.assigned_cluster,
                        t.data_version,
                        t.created_at,
                        COUNT(DISTINCT a.annotator) as annotators
                    FROM texts t
                    LEFT JOIN annotations a ON a.text_id = t.id
                    INNER JOIN skipped_texts s ON s.text_id = t.id AND s.annotator = ?
                """
                params = [annotator]
                filters = [
                    "NOT EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id AND a2.annotator = ?)"
                ]
                params.append(annotator)
            else:
                base_query = """
                    SELECT
                        t.id,
                        t.text as request_text,
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
                # Exclude if already annotated or skipped by this annotator
                filters = [
                    "NOT EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id AND a2.annotator = ?)",
                    "NOT EXISTS (SELECT 1 FROM skipped_texts s WHERE s.text_id = t.id AND s.annotator = ?)"
                ]
                params.extend([annotator, annotator])
            
            # ASSIGNMENT LOGIC:
            # 1. If assigned_to is set, it MUST match annotator
            # 2. If assigned_to is NULL, use Cluster/Language logic
            
            # Combine into a complex OR clause:
            # (assigned_to = ? OR (assigned_to IS NULL AND [cluster/lang filters]))
            
            assignment_clause = "(t.assigned_to = ?"
            assignment_params = [annotator]
            
            fallback_filters = []
            
            # Add cluster filter to fallback
            if clusters:
                placeholders = ", ".join("?" for _ in clusters)
                fallback_filters.append(f"t.assigned_cluster IN ({placeholders})")
                assignment_params.extend(clusters)
            
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
            
            return conn.execute(query, params + [min_annotators]).fetchall()

    def get_all_texts_for_annotator(
        self,
        annotator: str,
        clusters: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> List[dict]:
        """Get all texts with status for an annotator.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            language: Filter by language
            
        Returns:
            List of dicts with id, request_text, is_annotated
        """
        with get_connection(self.db_path) as conn:
            base_query = """
                SELECT
                    t.id,
                    t.text as request_text,
                    t.assigned_to,
                    CASE 
                        WHEN EXISTS (SELECT 1 FROM annotations a WHERE a.text_id = t.id AND a.annotator = ?) THEN 1
                        ELSE 0
                    END as is_annotated,
                    CASE 
                        WHEN EXISTS (SELECT 1 FROM skipped_texts s WHERE s.text_id = t.id AND s.annotator = ?) THEN 1
                        ELSE 0
                    END as is_skipped
                FROM texts t
            """
            params = [annotator, annotator]
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
            
            where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
            query = f"{base_query} {where_clause} ORDER BY t.id ASC"
            
            return conn.execute(query, params).fetchall()
