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
                COUNT(DISTINCT CASE WHEN a.decision = 'yes' THEN a.id END) as positive_annotations
            FROM texts t
            LEFT JOIN annotations a ON t.id = a.text_id
        """
        
        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def get_annotator_stats(self) -> pd.DataFrame:
        """Get per-annotator statistics.
        
        Returns:
            DataFrame with annotator stats
        """
        query = """
            SELECT 
                a.annotator,
                COUNT(DISTINCT a.text_id) as texts_annotated,
                COUNT(a.id) as total_decisions,
                SUM(CASE WHEN a.decision = 'yes' THEN 1 ELSE 0 END) as yes_count,
                SUM(CASE WHEN a.decision = 'no' THEN 1 ELSE 0 END) as no_count,
                AVG(CASE WHEN a.decision = 'yes' THEN 1.0 ELSE 0.0 END) as yes_rate
            FROM annotations a
            GROUP BY a.annotator
            ORDER BY texts_annotated DESC
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
    
    def get_disagreements(self, min_annotators: int = 2) -> pd.DataFrame:
        """Get texts with annotation disagreements.
        
        Args:
            min_annotators: Minimum annotators required
            
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
            return pd.read_sql_query(query, conn, params=(min_annotators,))
    
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
        is_annotated: Optional[bool] = None,
        only_disagreements: bool = False,
        limit: Optional[int] = 100,
        offset: int = 0
    ) -> pd.DataFrame:
        """Get detailed text overview with candidates and annotations.
        
        Args:
            search_query: Text to search in requests
            top5_intents: Filter by intents present in Top-5
            human_intents: Filter by intents chosen by humans
            assigned_annotators: Filter by assigned annotators
            is_annotated: Filter by annotation status
            only_disagreements: Filter to show only disagreements
            limit: Pagination limit
            offset: Pagination offset
            
        Returns:
            DataFrame with detailed text info
        """
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
            placeholders = ", ".join(["?"] * len(assigned_annotators))
            where_clauses.append(f"t.assigned_to IN ({placeholders})")
            params.extend(assigned_annotators)
            
        if only_disagreements:
            where_clauses.append("""
                EXISTS (
                    SELECT 1 FROM annotations a3 
                    WHERE a3.text_id = t.id 
                    GROUP BY a3.text_id, a3.label 
                    HAVING COUNT(DISTINCT a3.annotator) >= 2 
                       AND SUM(CASE WHEN a3.decision = 'yes' THEN 1 ELSE 0 END) > 0 
                       AND SUM(CASE WHEN a3.decision = 'no' THEN 1 ELSE 0 END) > 0
                )
            """)
            
        where_sql = " AND ".join(where_clauses)
        
        # Base query to get text IDs and core info
        base_query = f"""
            SELECT 
                t.id, 
                t.text, 
                t.assigned_to,
                t.assigned_cluster as cluster,
                (SELECT GROUP_CONCAT(DISTINCT annotator) FROM annotations WHERE text_id = t.id) as actual_annotators,
                EXISTS (
                    SELECT 1 FROM annotations a3 
                    WHERE a3.text_id = t.id 
                    GROUP BY a3.text_id, a3.label 
                    HAVING COUNT(DISTINCT a3.annotator) >= 2 
                       AND SUM(CASE WHEN a3.decision = 'yes' THEN 1 ELSE 0 END) > 0 
                       AND SUM(CASE WHEN a3.decision = 'no' THEN 1 ELSE 0 END) > 0
                ) as has_disagreement
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
            
        # Process candidates: Pivot to separate columns
        for i in range(1, 6):
            rank_candidates = candidates_df[candidates_df['rank'] == i].set_index('text_id')
            df[f'Model Top {i}'] = df['id'].map(
                lambda x: f"{rank_candidates.at[x, 'label']} ({rank_candidates.at[x, 'probability']*100:.1f}%)" 
                if x in rank_candidates.index else ""
            )
            
        # Process human annotations: Dynamic columns for "Yes" labels
        # Group by text_id to get list of labels
        human_labels = annotations_df.groupby('text_id')['label'].apply(list).to_dict()
        
        max_labels = 0
        if human_labels:
            max_labels = max(len(labels) for labels in human_labels.values())
            
        for i in range(max_labels):
            df[f'Chosen Intent {i+1}'] = df['id'].map(
                lambda x: human_labels[x][i] if x in human_labels and len(human_labels[x]) > i else ""
            )
            
        return df

    def get_text_count(
        self,
        search_query: str = "",
        top5_intents: List[str] = None,
        human_intents: List[str] = None,
        assigned_annotators: List[str] = None,
        is_annotated: Optional[bool] = None,
        only_disagreements: bool = False
    ) -> int:
        """Get count of texts matching filters.
        
        Returns:
            Count of texts
        """
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
            placeholders = ", ".join(["?"] * len(assigned_annotators))
            where_clauses.append(f"t.assigned_to IN ({placeholders})")
            params.extend(assigned_annotators)
            
        if only_disagreements:
            where_clauses.append("""
                EXISTS (
                    SELECT 1 FROM annotations a3 
                    WHERE a3.text_id = t.id 
                    GROUP BY a3.text_id, a3.label 
                    HAVING COUNT(DISTINCT a3.annotator) >= 2 
                       AND SUM(CASE WHEN a3.decision = 'yes' THEN 1 ELSE 0 END) > 0 
                       AND SUM(CASE WHEN a3.decision = 'no' THEN 1 ELSE 0 END) > 0
                )
            """)
            
        where_sql = " AND ".join(where_clauses)
        
        # Base query to get text IDs and core info
        base_query = f"""
            SELECT 
                t.id, 
                t.text, 
                t.assigned_to,
                t.assigned_cluster as cluster,
                (SELECT GROUP_CONCAT(DISTINCT annotator) FROM annotations WHERE text_id = t.id) as actual_annotators,
                EXISTS (
                    SELECT 1 FROM annotations a3 
                    WHERE a3.text_id = t.id 
                    GROUP BY a3.text_id, a3.label 
                    HAVING COUNT(DISTINCT a3.annotator) >= 2 
                       AND SUM(CASE WHEN a3.decision = 'yes' THEN 1 ELSE 0 END) > 0 
                       AND SUM(CASE WHEN a3.decision = 'no' THEN 1 ELSE 0 END) > 0
                ) as has_disagreement
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
            
        # Process candidates: Pivot to separate columns
        for i in range(1, 6):
            rank_candidates = candidates_df[candidates_df['rank'] == i].set_index('text_id')
            df[f'Model Top {i}'] = df['id'].map(
                lambda x: f"{rank_candidates.at[x, 'label']} ({rank_candidates.at[x, 'probability']*100:.1f}%)" 
                if x in rank_candidates.index else ""
            )
            
        # Process human annotations: Dynamic columns for "Yes" labels
        # Group by text_id to get list of labels
        human_labels = annotations_df.groupby('text_id')['label'].apply(list).to_dict()
        
        max_labels = 0
        if human_labels:
            max_labels = max(len(labels) for labels in human_labels.values())
            
        for i in range(max_labels):
            df[f'Chosen Intent {i+1}'] = df['id'].map(
                lambda x: human_labels[x][i] if x in human_labels and len(human_labels[x]) > i else ""
            )
            
        return df

    def get_text_count(
        self,
        search_query: str = "",
        model_intents: List[str] = None,
        top5_intents: List[str] = None,
        human_intents: List[str] = None,
        assigned_annotators: List[str] = None,
        is_annotated: Optional[bool] = None,
        only_disagreements: bool = False
    ) -> int:
        """Get count of texts matching filters.
        
        Returns:
            Count of texts
        """
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
            placeholders = ", ".join(["?"] * len(assigned_annotators))
            where_clauses.append(f"t.assigned_to IN ({placeholders})")
            params.extend(assigned_annotators)
            
        if only_disagreements:
            where_clauses.append("""
                EXISTS (
                    SELECT 1 FROM annotations a3 
                    WHERE a3.text_id = t.id 
                    GROUP BY a3.text_id, a3.label 
                    HAVING COUNT(DISTINCT a3.annotator) >= 2 
                       AND SUM(CASE WHEN a3.decision = 'yes' THEN 1 ELSE 0 END) > 0 
                       AND SUM(CASE WHEN a3.decision = 'no' THEN 1 ELSE 0 END) > 0
                )
            """)
            
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
        """Get list of all unique assigned annotators.
        
        Returns:
            List of annotator names
        """
        with get_connection(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT assigned_to FROM texts WHERE assigned_to IS NOT NULL AND assigned_to != '' ORDER BY assigned_to").fetchall()
            return [row['assigned_to'] for row in rows]
