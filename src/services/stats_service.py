"""Stats service for admin dashboard metrics."""

import pandas as pd
from typing import Dict, List, Optional

from src.repositories.base import BaseRepository
from src.utils.database import get_connection
from src.utils.logger import logger


class StatsService(BaseRepository):
    """Service for generating statistics and metrics."""
    
    def get_overall_stats(self) -> pd.DataFrame:
        """Get overall annotation statistics.
        
        Returns:
            DataFrame with overall stats
        """
        query = """
            SELECT 
                COUNT(DISTINCT t.id) as total_texts,
                COUNT(DISTINCT a.annotator) as total_annotators,
                COUNT(DISTINCT a.id) as total_annotations,
                COUNT(DISTINCT CASE WHEN a.decision = 'yes' THEN a.id END) as positive_annotations,
                (
                    SELECT COUNT(*) FROM (
                        SELECT text_id 
                        FROM annotations 
                        GROUP BY text_id 
                        HAVING COUNT(DISTINCT annotator) >= ?
                    )
                ) as fully_annotated_texts,
                (
                    SELECT COUNT(DISTINCT text_id) 
                    FROM annotations 
                    WHERE is_candidate = 0
                ) as texts_with_extra_intents
            FROM texts t
            LEFT JOIN annotations a ON t.id = a.text_id
        """
        
        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(1,))
    
    def get_daily_activity(self) -> pd.DataFrame:
        """Get annotation activity by day.
        
        Returns:
            DataFrame with daily activity [date, annotator, count]
        """
        query = """
            SELECT
                DATE(created_at) as date,
                annotator,
                COUNT(*) as count
            FROM annotations
            GROUP BY DATE(created_at), annotator
            ORDER BY date DESC
        """
        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_hourly_activity(self) -> pd.DataFrame:
        """Get annotation activity by hour.
        
        Returns:
            DataFrame with hourly activity [date, hour, annotator, count]
        """
        query = """
            SELECT
                DATE(created_at) as date,
                CAST(strftime('%H', created_at) AS INTEGER) as hour,
                annotator,
                COUNT(*) as count
            FROM annotations
            GROUP BY DATE(created_at), strftime('%H', created_at), annotator
            ORDER BY date DESC, hour
        """
        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def get_annotator_stats(self) -> pd.DataFrame:
        """Get per-annotator statistics.

        Returns:
            DataFrame with annotator stats
        """
        query = """
            WITH assigned AS (
                SELECT assigned_to AS annotator,
                       COUNT(*) AS texts_assigned
                FROM texts
                WHERE assigned_to IS NOT NULL AND assigned_to != ''
                GROUP BY assigned_to
            ),
            annotated AS (
                SELECT a.annotator,
                       COUNT(DISTINCT a.text_id) AS texts_annotated,
                       COUNT(a.id) AS total_decisions,
                       SUM(CASE WHEN a.decision = 'yes' THEN 1 ELSE 0 END) AS yes_count,
                       SUM(CASE WHEN a.decision = 'no' THEN 1 ELSE 0 END) AS no_count,
                       AVG(CASE WHEN a.decision = 'yes' THEN 1.0 ELSE 0.0 END) AS yes_rate
                FROM annotations a
                GROUP BY a.annotator
            )
            SELECT
                COALESCE(asgn.annotator, ann.annotator) AS annotator,
                COALESCE(asgn.texts_assigned, 0) AS texts_assigned,
                COALESCE(ann.texts_annotated, 0) AS texts_annotated,
                CASE WHEN COALESCE(asgn.texts_assigned, 0) > 0
                     THEN CAST(COALESCE(ann.texts_annotated, 0) AS REAL) / asgn.texts_assigned
                     ELSE 0.0 END AS annotated_pct,
                COALESCE(ann.total_decisions, 0) AS total_decisions,
                COALESCE(ann.yes_count, 0) AS yes_count,
                COALESCE(ann.no_count, 0) AS no_count,
                COALESCE(ann.yes_rate, 0.0) AS yes_rate
            FROM assigned asgn
            LEFT JOIN annotated ann ON asgn.annotator = ann.annotator
            UNION ALL
            SELECT
                ann.annotator,
                0 AS texts_assigned,
                ann.texts_annotated,
                0.0 AS annotated_pct,
                ann.total_decisions,
                ann.yes_count,
                ann.no_count,
                ann.yes_rate
            FROM annotated ann
            WHERE ann.annotator NOT IN (SELECT annotator FROM assigned)
            ORDER BY texts_assigned DESC, texts_annotated DESC
        """

        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def get_intent_quality(self) -> pd.DataFrame:
        """Get intent quality metrics based on model predictions.
        
        Returns:
            DataFrame with intent quality metrics
        """
        query = """
            WITH top1_metrics AS (
                SELECT 
                    c.label,
                    COUNT(DISTINCT c.text_id) as top1_count,
                    COALESCE(SUM(CASE WHEN a.decision = 'yes' THEN 1 ELSE 0 END), 0) as top1_yes
                FROM candidates c
                LEFT JOIN annotations a ON a.text_id = c.text_id AND a.label = c.label
                WHERE c.rank = 1
                GROUP BY c.label
            ),
            candidate_metrics AS (
                SELECT
                    c.label,
                    COUNT(DISTINCT c.text_id) as candidate_count,
                    COUNT(DISTINCT CASE WHEN a.id IS NOT NULL THEN c.text_id END) as candidate_annotated_count
                FROM candidates c
                LEFT JOIN annotations a ON a.text_id = c.text_id
                GROUP BY c.label
            ),
            missed_metrics AS (
                SELECT 
                    a.label,
                    COUNT(DISTINCT a.text_id) as missed_count
                FROM annotations a
                WHERE a.decision = 'yes' 
                  AND a.is_candidate = 0
                GROUP BY a.label
            ),
            confusions AS (
                SELECT
                    a.label as actual_label,
                    c.label as predicted_label,
                    COUNT(*) as count,
                    ROW_NUMBER() OVER (PARTITION BY a.label ORDER BY COUNT(*) DESC) as rn
                FROM annotations a
                JOIN candidates c ON a.text_id = c.text_id
                WHERE a.decision = 'yes'
                  AND c.probability >= 0.999
                  AND a.label != c.label
                GROUP BY a.label, c.label
            )
            SELECT 
                i.label,
                i.cluster,
                i.complexity,
                COALESCE(t.top1_count, 0) as top1_shown,
                COALESCE(t.top1_yes, 0) as top1_accepted,
                COALESCE(m.missed_count, 0) as missed,
                cf.predicted_label as confusion_label,
                CASE
                    WHEN (COALESCE(t.top1_count, 0) + COALESCE(m.missed_count, 0)) > 0 THEN
                        CAST(COALESCE(cf.count, 0) AS FLOAT) / (COALESCE(t.top1_count, 0) + COALESCE(m.missed_count, 0))
                    ELSE 0.0
                END as confusion_percentage,
                CASE 
                    WHEN COALESCE(cm.candidate_count, 0) > 0 THEN 
                        CAST(COALESCE(cm.candidate_annotated_count, 0) AS FLOAT) / cm.candidate_count
                    ELSE 0.0
                END as annotation_rate,
                CASE 
                    WHEN COALESCE(t.top1_count, 0) > 0 THEN 
                        CAST(COALESCE(t.top1_yes, 0) AS FLOAT) / t.top1_count
                    ELSE 0.0
                END as top1_precision,
                CASE 
                    WHEN (COALESCE(t.top1_count, 0) + COALESCE(m.missed_count, 0)) > 0 THEN 
                        CAST(COALESCE(m.missed_count, 0) AS FLOAT) / (COALESCE(t.top1_count, 0) + COALESCE(m.missed_count, 0))
                    ELSE 0.0
                END as miss_rate
            FROM intents i
            LEFT JOIN top1_metrics t ON t.label = i.label
            LEFT JOIN candidate_metrics cm ON cm.label = i.label
            LEFT JOIN missed_metrics m ON m.label = i.label
            LEFT JOIN confusions cf ON cf.actual_label = i.label AND cf.rn = 1
            ORDER BY i.cluster, i.label
        """
        
        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_model_quality_legacy(self) -> pd.DataFrame:
        """Get per-intent model quality metrics (legacy logic).
        
        Returns:
            DataFrame with intent quality metrics:
            - top1_precision
            - missed_rate
            - top1_count, top1_yes, top1_no
            - potential_count, missed_opportunity_count
        """
        with get_connection(self.db_path) as conn:
            # Get top-1 precision per intent
            top1_stats = conn.execute(
                """
                SELECT
                    c.label,
                    COUNT(DISTINCT c.text_id || '-' || a.annotator) as top1_count,
                    SUM(CASE WHEN a.decision = 'yes' THEN 1 ELSE 0 END) as top1_yes,
                    SUM(CASE WHEN a.decision = 'no' THEN 1 ELSE 0 END) as top1_no
                FROM candidates c
                JOIN annotations a ON a.text_id = c.text_id AND a.label = c.label
                WHERE c.rank = 1
                GROUP BY c.label
                """
            ).fetchall()

            # Get missed opportunities: intent is rank 2-N, top1 is no, this intent is yes
            missed_stats = conn.execute(
                """
                SELECT
                    c_other.label,
                    COUNT(*) as missed_opportunity_count
                FROM candidates c_top1
                JOIN candidates c_other ON c_other.text_id = c_top1.text_id
                    AND c_other.rank > 1
                JOIN annotations a_top1 ON a_top1.text_id = c_top1.text_id
                    AND a_top1.label = c_top1.label
                JOIN annotations a_other ON a_other.text_id = c_other.text_id
                    AND a_other.label = c_other.label
                    AND a_other.annotator = a_top1.annotator
                WHERE c_top1.rank = 1
                    AND a_top1.decision = 'no'
                    AND a_other.decision = 'yes'
                GROUP BY c_other.label
                """
            ).fetchall()

            # Get total times each intent appeared in rank 2-N when top1 was rejected
            potential_missed = conn.execute(
                """
                SELECT
                    c_other.label,
                    COUNT(DISTINCT c_other.text_id || '-' || a_top1.annotator) as potential_count
                FROM candidates c_top1
                JOIN candidates c_other ON c_other.text_id = c_top1.text_id
                    AND c_other.rank > 1
                JOIN annotations a_top1 ON a_top1.text_id = c_top1.text_id
                    AND a_top1.label = c_top1.label
                WHERE c_top1.rank = 1
                    AND a_top1.decision = 'no'
                GROUP BY c_other.label
                """
            ).fetchall()

            # Get cluster info
            intent_clusters = conn.execute(
                "SELECT label, cluster FROM intents"
            ).fetchall()

        # Build dataframes
        top1_df = pd.DataFrame(top1_stats, columns=[
            "label", "top1_count", "top1_yes", "top1_no"
        ]) if top1_stats else pd.DataFrame(columns=["label", "top1_count", "top1_yes", "top1_no"])

        missed_df = pd.DataFrame(missed_stats, columns=[
            "label", "missed_opportunity_count"
        ]) if missed_stats else pd.DataFrame(columns=["label", "missed_opportunity_count"])

        potential_df = pd.DataFrame(potential_missed, columns=[
            "label", "potential_count"
        ]) if potential_missed else pd.DataFrame(columns=["label", "potential_count"])

        cluster_df = pd.DataFrame(intent_clusters, columns=[
            "label", "cluster"
        ]) if intent_clusters else pd.DataFrame(columns=["label", "cluster"])

        # Merge all
        result = top1_df.merge(missed_df, on="label", how="outer")
        result = result.merge(potential_df, on="label", how="outer")
        result = result.merge(cluster_df, on="label", how="left")

        # Fill NaN with 0
        for col in ["top1_count", "top1_yes", "top1_no", "missed_opportunity_count", "potential_count"]:
            result[col] = result[col].fillna(0).astype(int)

        # Calculate metrics
        result["top1_precision"] = result.apply(
            lambda r: r["top1_yes"] / r["top1_count"] if r["top1_count"] > 0 else None,
            axis=1
        )
        result["missed_rate"] = result.apply(
            lambda r: r["missed_opportunity_count"] / r["potential_count"] if r["potential_count"] > 0 else None,
            axis=1
        )

        return result.sort_values("top1_count", ascending=False)
    
    def get_cluster_progress(self) -> pd.DataFrame:
        """Get annotation progress by cluster.
        
        Returns:
            DataFrame with cluster progress
        """
        query = """
            SELECT 
                t.assigned_cluster as cluster,
                COUNT(DISTINCT t.id) as total_texts,
                COUNT(DISTINCT a.text_id) as annotated_texts,
                CAST(COUNT(DISTINCT a.text_id) AS FLOAT) / COUNT(DISTINCT t.id) as completion_rate
            FROM texts t
            LEFT JOIN annotations a ON a.text_id = t.id
            WHERE t.assigned_cluster IS NOT NULL
            GROUP BY t.assigned_cluster
            ORDER BY completion_rate ASC
        """
        
        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def get_disagreements(self) -> pd.DataFrame:
        """Get texts with annotation disagreements.
        
        Returns:
            DataFrame with disagreement cases
        """
        query = """
            SELECT 
                t.id as text_id,
                t.text,
                a.label,
                COUNT(DISTINCT a.annotator) as annotator_count,
                SUM(CASE WHEN a.decision = 'yes' THEN 1 ELSE 0 END) as yes_count,
                SUM(CASE WHEN a.decision = 'no' THEN 1 ELSE 0 END) as no_count
            FROM texts t
            JOIN annotations a  ON a.text_id = t.id
            GROUP BY t.id, t.text, a.label
            HAVING COUNT(DISTINCT a.annotator) >= ?
                AND SUM(CASE WHEN a.decision = 'yes' THEN 1 ELSE 0 END) > 0
                AND SUM(CASE WHEN a.decision = 'no' THEN 1 ELSE 0 END) > 0
            ORDER BY annotator_count DESC, t.id
        """
        
        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(2,))
    
    def export_annotations(self, output_path: str) -> int:
        """Export all annotations to CSV.
        
        Args:
            output_path: Path to output CSV file
            
        Returns:
            Number of rows exported
        """
        query = """
            SELECT 
                t.id as text_id,
                t.text,
                t.language,
                t.assigned_cluster,
                a.annotator,
                a.label,
                a.decision,
                a.is_candidate,
                a.created_at
            FROM texts t
            JOIN annotations a ON a.text_id = t.id
            ORDER BY t.id, a.annotator, a.label
        """
        
        with get_connection(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
            df.to_csv(output_path, index=False, encoding="utf-8")
            logger.info(f"Exported {len(df)} annotations to {output_path}")
            return len(df)
    def get_text_detailed_overview(
        self,
        search_query: str = "",
        top5_intents: List[str] = None,
        human_intents: List[str] = None,
        assigned_annotators: List[str] = None,
        languages: List[str] = None,
        is_annotated: Optional[bool] = None,
        limit: Optional[int] = 100,
        offset: int = 0
    ) -> pd.DataFrame:
        """Get detailed text overview with candidates and annotations."""
        where_clauses = ["1=1"]
        params = []
        
        if search_query:
            where_clauses.append("t.text LIKE ?")
            params.append(f"%{search_query}%")
            
        if is_annotated is not None:
            if is_annotated:
                where_clauses.append("EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id)")
            else:
                where_clauses.append("NOT EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id)")
                
        if top5_intents:
            placeholders = ", ".join(["?"] * len(top5_intents))
            where_clauses.append(f"EXISTS (SELECT 1 FROM candidates c2 WHERE c2.text_id = t.id AND c2.rank <= 5 AND c2.label IN ({placeholders}))")
            params.extend(top5_intents)
            
        if human_intents:
            placeholders = ", ".join(["?"] * len(human_intents))
            where_clauses.append(f"EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id AND a2.decision = 'yes' AND a2.label IN ({placeholders}))")
            params.extend(human_intents)

        if assigned_annotators:
            named = [a for a in assigned_annotators if a != "[Unassigned]"]
            include_null = "[Unassigned]" in assigned_annotators
            parts = []
            if named:
                placeholders = ", ".join(["?"] * len(named))
                parts.append(f"t.assigned_to IN ({placeholders})")
                params.extend(named)
            if include_null:
                parts.append("t.assigned_to IS NULL")
            if parts:
                where_clauses.append(f"({' OR '.join(parts)})")

        if languages:
            placeholders = ", ".join(["?"] * len(languages))
            where_clauses.append(f"t.language IN ({placeholders})")
            params.extend(languages)

        where_sql = " AND ".join(where_clauses)

        base_query = f"""
            SELECT
                t.id,
                t.text,
                t.language,
                t.assigned_to,
                t.assigned_cluster as cluster,
                (SELECT GROUP_CONCAT(DISTINCT annotator) FROM annotations WHERE text_id = t.id) as actual_annotators
            FROM texts t
            WHERE {where_sql}
            ORDER BY t.created_at DESC
        """
        
        if limit is not None:
            base_query += f" LIMIT {limit} OFFSET {offset}"
            
        with get_connection(self.db_path) as conn:
            df = pd.read_sql_query(base_query, conn, params=params)
            
            if df.empty:
                return df
                
            text_ids = df['id'].tolist()
            text_ids_placeholders = ", ".join(["?"] * len(text_ids))
            
            # Fetch candidates (Top 5)
            candidates_query = f"""
                SELECT text_id, label, probability, rank
                FROM candidates
                WHERE text_id IN ({text_ids_placeholders}) AND rank <= 5
                ORDER BY text_id, rank
            """
            candidates_df = pd.read_sql_query(candidates_query, conn, params=text_ids)
            
            # Fetch annotations (Yes only)
            annotations_query = f"""
                SELECT text_id, label
                FROM annotations
                WHERE text_id IN ({text_ids_placeholders}) AND decision = 'yes'
                GROUP BY text_id, label
            """
            annotations_df = pd.read_sql_query(annotations_query, conn, params=text_ids)
            
        # Build lookups for yes-chosen labels and top-K candidate labels
        human_labels: dict = (
            annotations_df.groupby('text_id')['label'].apply(list).to_dict()
            if not annotations_df.empty else {}
        )
        human_yes_sets: dict = {tid: set(labels) for tid, labels in human_labels.items()}
        top_k_sets: dict = (
            candidates_df.groupby('text_id')['label'].apply(set).to_dict()
            if not candidates_df.empty else {}
        )

        # Process candidates: Pivot to separate columns.
        # Mark with ✅ if the annotator voted yes for that label.
        for i in range(1, 6):
            rank_df = candidates_df[candidates_df['rank'] == i].set_index('text_id')

            def _fmt_model_top(x, _rank_df=rank_df, _yes=human_yes_sets):
                if x not in _rank_df.index:
                    return ""
                label = _rank_df.at[x, 'label']
                prob = _rank_df.at[x, 'probability']
                marker = "✅ " if label in _yes.get(x, set()) else ""
                return f"{marker}{label} ({prob * 100:.1f}%)"

            df[f'Model Top {i}'] = df['id'].map(_fmt_model_top)

        # Build Additional Intent columns:
        # yes-chosen labels that are NOT among the top-K candidates for that text.
        additional_labels: dict = {
            tid: [lbl for lbl in labels if lbl not in top_k_sets.get(tid, set())]
            for tid, labels in human_labels.items()
        }
        max_additional = max((len(v) for v in additional_labels.values()), default=0)

        for i in range(max_additional):
            def _fmt_additional(x, _add=additional_labels, _i=i):
                extras = _add.get(x, [])
                return extras[_i] if _i < len(extras) else ""

            df[f'Additional Intent {i + 1}'] = df['id'].map(_fmt_additional)

        return df

    def get_text_count(
        self,
        search_query: str = "",
        model_intents: List[str] = None,
        top5_intents: List[str] = None,
        human_intents: List[str] = None,
        assigned_annotators: List[str] = None,
        languages: List[str] = None,
        is_annotated: Optional[bool] = None,
    ) -> int:
        """Get count of texts matching filters."""
        where_clauses = ["1=1"]
        params = []
        
        if search_query:
            where_clauses.append("t.text LIKE ?")
            params.append(f"%{search_query}%")
            
        if is_annotated is not None:
            if is_annotated:
                where_clauses.append("EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id)")
            else:
                where_clauses.append("NOT EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id)")
                
        if model_intents:
            placeholders = ", ".join(["?"] * len(model_intents))
            where_clauses.append(f"EXISTS (SELECT 1 FROM candidates c2 WHERE c2.text_id = t.id AND c2.probability >= 0.999 AND c2.label IN ({placeholders}))")
            params.extend(model_intents)

        if top5_intents:
            placeholders = ", ".join(["?"] * len(top5_intents))
            where_clauses.append(f"EXISTS (SELECT 1 FROM candidates c2 WHERE c2.text_id = t.id AND c2.rank <= 5 AND c2.label IN ({placeholders}))")
            params.extend(top5_intents)
            
        if human_intents:
            placeholders = ", ".join(["?"] * len(human_intents))
            where_clauses.append(f"EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id AND a2.decision = 'yes' AND a2.label IN ({placeholders}))")
            params.extend(human_intents)

        if assigned_annotators:
            named = [a for a in assigned_annotators if a != "[Unassigned]"]
            include_null = "[Unassigned]" in assigned_annotators
            parts = []
            if named:
                placeholders = ", ".join(["?"] * len(named))
                parts.append(f"t.assigned_to IN ({placeholders})")
                params.extend(named)
            if include_null:
                parts.append("t.assigned_to IS NULL")
            if parts:
                where_clauses.append(f"({' OR '.join(parts)})")

        if languages:
            placeholders = ", ".join(["?"] * len(languages))
            where_clauses.append(f"t.language IN ({placeholders})")
            params.extend(languages)
            
        where_sql = " AND ".join(where_clauses)
        query = f"SELECT COUNT(*) as count FROM texts t WHERE {where_sql}"
        
        with get_connection(self.db_path) as conn:
            result = conn.execute(query, params).fetchone()
            return result['count'] if result else 0

    def get_all_intents(self) -> List[str]:
        """Get list of all intents in the system.
        
        Returns:
            List of intent labels
        """
        with get_connection(self.db_path) as conn:
            rows = conn.execute("SELECT label FROM intents ORDER BY label").fetchall()
            return [row['label'] for row in rows]

    def get_unique_assigned_annotators(self) -> List[str]:
        """Get list of all unique assigned annotators."""
        with get_connection(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT assigned_to FROM texts WHERE assigned_to IS NOT NULL AND assigned_to != '' ORDER BY assigned_to").fetchall()
            return [row['assigned_to'] for row in rows]

    def get_unique_languages(self) -> List[str]:
        """Get list of all unique languages in the texts table."""
        with get_connection(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT language FROM texts WHERE language IS NOT NULL AND language != '' ORDER BY language").fetchall()
            return [row['language'] for row in rows]
