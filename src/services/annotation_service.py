"""Annotation service for managing annotation workflow."""

from typing import Dict, List, Optional, Tuple

import streamlit as st
from src.models.annotator import Annotator
from src.models.candidate import Candidate
from src.models.text import Text
from src.repositories.annotation_repo import AnnotationRepository
from src.repositories.text_repo import TextRepository
from src.utils.database import get_connection
from src.utils.logger import logger


class AnnotationService:
    """Service for managing annotation workflow."""
    
    def __init__(self, db_path: str):
        """Initialize annotation service.
        
        Args:
            db_path: Path to database
        """
        self.text_repo = TextRepository(db_path)
        self.annotation_repo = AnnotationRepository(db_path)
    
    def get_next_text(
        self,
        annotator: str,
        clusters: Optional[List[str]] = None,
        intents: Optional[List[str]] = None,
        language: Optional[str] = None,
        show_skipped: bool = False
    ) -> Optional[dict]:
        """Get next text for annotation.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            intents: Filter by intents
            language: Filter by language
            show_skipped: Show only skipped texts
            
        Returns:
            Text row if found, None otherwise
        """
        texts = self.text_repo.get_unannotated(
            annotator=annotator,
            clusters=clusters,
            intents=intents,
            language=language,
            show_skipped=show_skipped
        )
        
        if not texts:
            return None
        
        return dict(texts[0])

    
    def get_text_with_candidates(self, text_id: int) -> Tuple[Optional[Text], List[Candidate]]:
        """Get text and its candidates.
        
        Args:
            text_id: Text ID
            
        Returns:
            Tuple of (Text, candidates list)
        """
        text = self.text_repo.get_by_id(text_id)
        if not text:
            return None, []
        
        candidates = self.text_repo.get_candidates(text_id)
        return text, candidates
    
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
            decisions: Dict of label -> decision (yes/no)
            candidate_labels: Labels from candidates
            extra_labels: Extra labels added
            shown_intents_source: Source tracking for intents
        """
        self.annotation_repo.save_annotations(
            text_id=text_id,
            annotator=annotator,
            decisions=decisions,
            candidate_labels=candidate_labels,
            extra_labels=extra_labels,
            shown_intents_source=shown_intents_source
        )
        logger.info(f"Saved {len(decisions) + len(extra_labels)} annotations for text#{text_id}")
    
    def skip_text(self, text_id: int, annotator: str) -> None:
        """Skip a text.
        
        Args:
            text_id: Text ID
            annotator: Annotator name
        """
        self.annotation_repo.skip_text(text_id, annotator)
    
    def unskip_text(self, text_id: int, annotator: str) -> None:
        """Unskip a previously skipped text.
        
        Args:
            text_id: Text ID
            annotator: Annotator name
        """
        self.annotation_repo.unskip_text(text_id, annotator)
    
    def get_progress(
        self,
        annotator: str,
        clusters: Optional[List[str]] = None,
        intents: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> dict:
        """Get annotation progress.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            intents: Filter by intents
            language: Filter by language
            
        Returns:
            Dict with 'total' and 'done' counts
        """
        return self.annotation_repo.get_progress(
            annotator=annotator,
            clusters=clusters,
            intents=intents,
            language=language
        )

    @st.cache_data
    def get_all_texts(
        _self,
        annotator: str,
        clusters: Optional[List[str]] = None,
        intents: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> List[dict]:
        """Get all texts for navigation (cached)."""
        return _self.text_repo.get_all_texts_for_annotator(
            annotator=annotator,
            clusters=clusters,
            intents=intents,
            language=language
        )

    def calculate_assignment(self, candidates: List[Candidate], annotators: List[Annotator], language: str) -> Optional[str]:
        """Calculate best annotator assignment based on intent weights.
        
        Args:
            candidates: List of prediction candidates
            annotators: List of available annotators
            language: Text language
            
        Returns:
            Name of assigned annotator or None
        """
        # Filter annotators by language
        eligible_annotators = [a for a in annotators if a.language == language and a.intents]
        
        if not eligible_annotators:
            return None
            
        best_annotator = None
        max_score = 0.0
        
        for annotator in eligible_annotators:
            # Score = sum of probabilities for candidates matching annotator's intents
            score = sum(c.probability for c in candidates if c.label in annotator.intents)
            
            if score > max_score:
                max_score = score
                best_annotator = annotator.name
                
        return best_annotator

    def assign_unannotated_texts(self, annotators: List[Annotator]) -> int:
        """Re-assign all unannotated texts based on current annotator configuration.
        
        Args:
            annotators: List of all annotators
            
        Returns:
            Count of updated texts
        """
        # Fetch ALL unannotated texts (we need a new repo method or iterate)
        # Iterating might be slow if many texts. 
        # But we need candidates for each.
        # Let's add a method to repo to get ALL unannotated texts regardless of user.
        # Or simple SQL.
        
        # For now, let's just get IDs of unannotated texts.
        # Then for each, get candidates, calculate, update.
        
        logger.info("Starting re-assignment of unannotated texts...")
        updated_count = 0
        
        with get_connection(self.text_repo.db_path) as conn:
            # Get unannotated texts that need assignment check
            # We treat all unannotated texts as candidates for re-assignment
            # regardless of current assigned_to (in case config changed)
            rows = conn.execute("""
                SELECT t.id, t.language, t.assigned_to 
                FROM texts t
                LEFT JOIN annotations a ON a.text_id = t.id
                WHERE a.id IS NULL
            """).fetchall()
            
            for row in rows:
                text_id = row["id"]
                language = row["language"]
                current_assigned = row["assigned_to"]
                
                # Get candidates
                # Using internal repo method or direct SQL for speed?
                # Repo method is fine.
                candidates = self.text_repo.get_candidates(text_id)
                
                # Calculate new assignment
                new_assigned = self.calculate_assignment(candidates, annotators, language)
                
                # Update if changed
                if new_assigned != current_assigned:
                    conn.execute(
                        "UPDATE texts SET assigned_to = ? WHERE id = ?",
                        (new_assigned, text_id)
                    )
                    updated_count += 1
            
            conn.commit()
            
        logger.info(f"Re-assigned {updated_count} texts")
        return updated_count
