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
                    SUM(CASE WHEN a.decision = 'yes' THEN 1 ELSE 0 END) as top1_yes
                FROM candidates c
                JOIN annotations a ON a.text_id = c.text_id AND a.label = c.label
                WHERE c.rank = 1
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
            )
            SELECT 
                i.label,
                i.cluster,
                i.complexity,
                COALESCE(t.top1_count, 0) as top1_shown,
                COALESCE(t.top1_yes, 0) as top1_accepted,
                COALESCE(m.missed_count, 0) as missed,
                CAST(COALESCE(t.top1_yes, 0) AS FLOAT) / NULLIF(t.top1_count, 0) as top1_precision,
                CAST(COALESCE(m.missed_count, 0) AS FLOAT) / NULLIF(t.top1_count + COALESCE(m.missed_count, 0), 0) as miss_rate
            FROM intents i
            LEFT JOIN top1_metrics t ON t.label = i.label
            LEFT JOIN missed_metrics m ON m.label = i.label
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
