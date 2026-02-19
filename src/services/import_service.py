"""Import service for CSV data import."""

import csv
import os
from typing import Dict, List, Optional

from modeling import TopKModelStub
from src.models.candidate import Candidate
from src.models.intent import Intent
from src.repositories.text_repo import TextRepository
from src.utils.logger import logger


class ImportService:
    """Service for importing data from CSV files."""
    
    def __init__(self, db_path: str):
        """Initialize import service.
        
        Args:
            db_path: Path to database
        """
        self.text_repo = TextRepository(db_path)
    
    def import_from_csv(
        self,
        csv_path: str,
        intents: Dict[str, Intent],
        top_k: int,
        model_version: int,
        data_version: int,
        probability_threshold: float = 0.0,
        annotators: list = None,
        annotation_service = None
    ) -> int:
        """Import texts from CSV file with model predictions.
        
        Args:
            csv_path: Path to CSV file
            intents: Dictionary of available intents
            top_k: Number of top candidates to generate
            model_version: Current model version
            data_version: Current data version
            annotators: List of annotators for assignment
            annotation_service: Service to calculate assignment
            
        Returns:
            Number of texts imported
        """
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            return 0
        
        imported_count = 0
        model = TopKModelStub(intents, top_k=top_k)
        
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    text = row.get("request_text", "").strip()
                    print(f"DEBUG IMPORT: text='{text}', keys={row.keys()}")
                    if not text:
                        print("DEBUG IMPORT: Missing text")
                        continue
                    if self.text_repo.exists(text):
                        print("DEBUG IMPORT: Text exists")
                        continue
                    
                    language = self._normalize_language(row.get("language"))
                    clusters = row.get("clusters")
                    
                    # Generate candidates: read from CSV score columns if present,
                    # otherwise fall back to model stub
                    candidates = self._candidates_from_row(row, intents, probability_threshold)
                    if candidates is None:
                        candidates = model.predict(text)
                    
                    # Determine assigned cluster from scores or clusters field
                    assigned_cluster = self._determine_cluster(row, intents)
                    
                    # Fallback: if unknown, use top candidate's cluster
                    if assigned_cluster == "unknown" and candidates:
                        top_candidate_label = candidates[0].label
                        if top_candidate_label in intents:
                            assigned_cluster = intents[top_candidate_label].cluster
                    
                    # Calculate assignment if services provided
                    assigned_to = None
                    if annotators and annotation_service:
                        assigned_to = annotation_service.calculate_assignment(candidates, annotators, language)
                    
                    # Create text with candidates
                    self.text_repo.create(
                        text=text,
                        language=language,
                        clusters=clusters,
                        assigned_cluster=assigned_cluster,
                        data_version=data_version,
                        candidates=candidates,
                        model_version=model_version,
                        assigned_to=assigned_to
                    )
                    
                    imported_count += 1
            
            logger.info(f"Imported {imported_count} texts from {csv_path}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import from CSV: {e}")
            raise
    
    def _candidates_from_row(
        self,
        row: dict,
        intents: Dict[str, Intent],
        probability_threshold: float
    ) -> Optional[List[Candidate]]:
        """Extract candidates from CSV row using intent-label column names.

        Returns None if no intent-label columns are found in the row,
        which signals the caller to fall back to the model stub.
        """
        scores = []
        for label in intents:
            if label in row:
                try:
                    prob = float(row[label])
                    if prob >= probability_threshold:
                        scores.append((label, prob))
                except (ValueError, TypeError):
                    continue

        if not scores:
            return None

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            Candidate(label=label, rank=idx + 1, probability=prob)
            for idx, (label, prob) in enumerate(scores)
        ]

    def _determine_cluster(self, row: Dict[str, str], intents: Dict[str, Intent]) -> str:
        """Determine cluster from CSV row.
        
        Args:
            row: CSV row
            intents: Available intents
            
        Returns:
            Cluster name
        """
        # Try to get from explicit clusters field
        clusters = row.get("clusters", "").strip()
        if clusters:
            return clusters.split(",")[0].strip()
        
        # Try to determine from scores
        scores = {}
        for key, value in row.items():
            if key.startswith("score_"):
                intent_label = key.replace("score_", "")
                try:
                    scores[intent_label] = float(value)
                except (ValueError, TypeError):
                    continue
        
        if scores:
            # Sum scores by cluster
            cluster_scores: Dict[str, float] = {}
            for label, score in scores.items():
                intent = intents.get(label)
                if intent and intent.cluster:
                    cluster_scores[intent.cluster] = cluster_scores.get(intent.cluster, 0) + score
            
            if cluster_scores:
                return max(cluster_scores.items(), key=lambda x: x[1])[0]
        
        return "unknown"

    def _normalize_language(self, lang: str) -> str:
        """Normalize language code.
        
        Args:
            lang: Language string
            
        Returns:
            Normalized language code (ru/kk) or original
        """
        if not lang:
            return None
            
        lang = lang.strip()
        mapping = {
            "Русский": "ru",
            "russian": "ru",
            "ru": "ru",
            "Казахский": "kk",
            "kazakh": "kk",
            "kz": "kk",
            "kk": "kk"
        }
        return mapping.get(lang, lang)
