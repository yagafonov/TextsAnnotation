"""Annotation service for managing annotation workflow."""

from typing import Dict, List, Optional, Tuple

from src.models.candidate import Candidate
from src.models.text import Text
from src.repositories.annotation_repo import AnnotationRepository
from src.repositories.text_repo import TextRepository
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
        language: Optional[str] = None,
        min_annotators: int = 2,
        show_skipped: bool = False
    ) -> Optional[dict]:
        """Get next text for annotation.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            language: Filter by language
            min_annotators: Minimum annotators required
            show_skipped: Show only skipped texts
            
        Returns:
            Text row if found, None otherwise
        """
        texts = self.text_repo.get_unannotated(
            annotator=annotator,
            clusters=clusters,
            language=language,
            min_annotators=min_annotators,
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
        language: Optional[str] = None
    ) -> dict:
        """Get annotation progress.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            language: Filter by language
            
        Returns:
            Dict with 'total' and 'done' counts
        """
        return self.annotation_repo.get_progress(
            annotator=annotator,
            clusters=clusters,
            language=language
        )
