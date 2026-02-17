"""Tests for AnnotationService.

This file provides example tests for the AnnotationService class.
These tests demonstrate proper testing patterns and can be used as a template.
"""

import pytest

from src.services.annotation_service import AnnotationService
from src.models.candidate import Candidate


class TestAnnotationServiceGetNextText:
    """Tests for get_next_text functionality."""
    
    def test_get_next_text_basic(self, temp_db, text_repo):
        """Test getting next unannotated text."""
        # Arrange: Create test text
        text_repo.create(
            text="Test annotation text",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Act: Get next text for annotation
        service = AnnotationService(temp_db)
        next_text = service.get_next_text(
            annotator="user1",
            clusters=["cluster1"],
            language="ru"
        )
        
        # Assert: Text is returned
        assert next_text is not None
        assert next_text["request_text"] == "Test annotation text"
        assert next_text["language"] == "ru"
    
    def test_get_next_text_filters_by_cluster(self, temp_db, text_repo):
        """Test cluster filtering works correctly."""
        # Arrange: Create texts in different clusters
        text_repo.create(
            text="Cluster 1 text",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        text_repo.create(
            text="Cluster 2 text",
            language="ru",
            clusters="cluster2",
            assigned_cluster="cluster2",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Act: Get text for cluster1 only
        service = AnnotationService(temp_db)
        next_text = service.get_next_text(
            annotator="user1",
            clusters=["cluster1"],
            language="ru"
        )
        
        # Assert: Only cluster1 text is returned
        assert next_text is not None
        assert "Cluster 1" in next_text["text"]
    
    def test_get_next_text_filters_by_language(self, temp_db, text_repo):
        """Test language filtering works correctly."""
        # Arrange: Create texts in different languages
        text_repo.create(
            text="Russian text",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        text_repo.create(
            text="English text",
            language="en",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Act: Get text for Russian language only
        service = AnnotationService(temp_db)
        next_text = service.get_next_text(
            annotator="user1",
            clusters=["cluster1"],
            language="ru"
        )
        
        # Assert: Only Russian text is returned
        assert next_text is not None
        assert next_text["language"] == "ru"
    
    def test_get_next_text_no_available(self, temp_db):
        """Test returns None when no texts are available."""
        # Arrange: Empty database
        service = AnnotationService(temp_db)
        
        # Act: Try to get next text
        next_text = service.get_next_text(
            annotator="user1",
            clusters=["cluster1"],
            language="ru"
        )
        
        # Assert: None is returned
        assert next_text is None
    
    def test_get_next_text_skips_already_annotated(self, temp_db, text_repo, annotation_repo):
        """Test skips texts already annotated by the user."""
        # Arrange: Create text and annotate it
        text_id = text_repo.create(
            text="Already annotated",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        annotation_repo.save_annotations(
            text_id=text_id,
            annotator="user1",
            decisions={"intent_a": "yes"},
            candidate_labels=["intent_a"],
            extra_labels=[],
            shown_intents_source={"intent_a": "candidate"}
        )
        
        # Act: Try to get next text for same user
        service = AnnotationService(temp_db)
        next_text = service.get_next_text(
            annotator="user1",
            clusters=["cluster1"],
            language="ru"
        )
        
        # Assert: None is returned (user already annotated the only text)
        assert next_text is None


class TestAnnotationServiceCandidates:
    """Tests for getting text with candidates."""
    
    def test_get_text_with_candidates(self, temp_db, text_repo):
        """Test retrieving text with its candidates."""
        # Arrange: Create text with candidates
        candidates = [
            Candidate(label="intent_a", rank=1, probability=0.9),
            Candidate(label="intent_b", rank=2, probability=0.7)
        ]
        
        text_id = text_repo.create(
            text="Test text with candidates",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=candidates,
            model_version=1
        )
        
        # Act: Get text with candidates
        service = AnnotationService(temp_db)
        text, retrieved_candidates = service.get_text_with_candidates(text_id)
        
        # Assert: Text and candidates are correct
        assert text is not None
        assert text.text == "Test text with candidates"
        assert len(retrieved_candidates) == 2
        assert retrieved_candidates[0].label == "intent_a"
        assert retrieved_candidates[0].rank == 1
        assert retrieved_candidates[1].label == "intent_b"
        assert retrieved_candidates[1].rank == 2
    
    def test_get_text_with_candidates_nonexistent(self, temp_db):
        """Test getting candidates for nonexistent text."""
        # Arrange: No text in database
        service = AnnotationService(temp_db)
        
        # Act: Try to get nonexistent text
        text, candidates = service.get_text_with_candidates(99999)
        
        # Assert: Returns None and empty list
        assert text is None
        assert candidates == []


class TestAnnotationServiceSaveAnnotations:
    """Tests for saving annotations."""
    
    def test_save_annotations_basic(self, temp_db, text_repo):
        """Test saving basic annotations."""
        # Arrange: Create text
        text_id = text_repo.create(
            text="Test text",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Act: Save annotations
        service = AnnotationService(temp_db)
        service.save_annotations(
            text_id=text_id,
            annotator="user1",
            decisions={"intent_a": "yes", "intent_b": "no"},
            candidate_labels=["intent_a", "intent_b"],
            extra_labels=[],
            shown_intents_source={"intent_a": "candidate", "intent_b": "candidate"}
        )
        
        # Assert: Annotations were saved (verify through repo)
        from src.repositories.annotation_repo import AnnotationRepository
        repo = AnnotationRepository(temp_db)
        annotations = repo.get_annotations_for_text(text_id)
        
        assert len(annotations) == 2
        decisions_map = {a.label: a.decision for a in annotations}
        assert decisions_map["intent_a"] == "yes"
        assert decisions_map["intent_b"] == "no"
    
    def test_save_annotations_with_extra_labels(self, temp_db, text_repo):
        """Test saving annotations with extra labels added by user."""
        # Arrange: Create text
        text_id = text_repo.create(
            text="Test text",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Act: Save annotations with extra labels
        service = AnnotationService(temp_db)
        service.save_annotations(
            text_id=text_id,
            annotator="user1",
            decisions={"intent_a": "yes"},
            candidate_labels=["intent_a"],
            extra_labels=["intent_extra"],
            shown_intents_source={"intent_a": "candidate"}
        )
        
        # Assert: Extra labels are saved
        from src.repositories.annotation_repo import AnnotationRepository
        repo = AnnotationRepository(temp_db)
        annotations = repo.get_annotations_for_text(text_id)
        
        assert len(annotations) == 2  # 1 from decisions + 1 extra
        labels = [a.label for a in annotations]
        assert "intent_a" in labels
        assert "intent_extra" in labels


class TestAnnotationServiceSkipUnskip:
    """Tests for skip/unskip functionality."""
    
    def test_skip_text(self, temp_db, text_repo):
        """Test skipping a text."""
        # Arrange: Create text
        text_id = text_repo.create(
            text="Difficult text",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Act: Skip the text
        service = AnnotationService(temp_db)
        service.skip_text(text_id, "user1")
        
        # Assert: Text is skipped (verify it's not returned by default)
        next_text = service.get_next_text(
            annotator="user1",
            clusters=["cluster1"],
            language="ru",
            show_skipped=False
        )
        assert next_text is None
    
    def test_unskip_text(self, temp_db, text_repo, annotation_repo):
        """Test unskipping a previously skipped text."""
        # Arrange: Create and skip text
        text_id = text_repo.create(
            text="Skipped text",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        annotation_repo.skip_text(text_id, "user1")
        
        # Act: Unskip the text
        service = AnnotationService(temp_db)
        service.unskip_text(text_id, "user1")
        
        # Assert: Text is available again
        next_text = service.get_next_text(
            annotator="user1",
            clusters=["cluster1"],
            language="ru"
        )
        assert next_text is not None
        assert next_text["id"] == text_id


class TestAnnotationServiceProgress:
    """Tests for progress tracking."""
    
    def test_get_progress_empty(self, temp_db):
        """Test progress with no texts."""
        # Arrange: Empty database
        service = AnnotationService(temp_db)
        
        # Act: Get progress
        progress = service.get_progress(
            annotator="user1",
            clusters=["cluster1"],
            language="ru"
        )
        
        # Assert: Both totals are 0
        assert progress["total"] == 0
        assert progress["done"] == 0
    
    def test_get_progress_with_annotations(self, temp_db, text_repo, annotation_repo):
        """Test progress calculation with some annotations."""
        # Arrange: Create multiple texts, annotate some
        text_ids = []
        for i in range(5):
            text_id = text_repo.create(
                text=f"Test text {i}",
                language="ru",
                clusters="cluster1",
                assigned_cluster="cluster1",
                data_version=1,
                candidates=[],
                model_version=1
            )
            text_ids.append(text_id)
        
        # Annotate first 3 texts
        for text_id in text_ids[:3]:
            annotation_repo.save_annotations(
                text_id=text_id,
                annotator="user1",
                decisions={"intent_a": "yes"},
                candidate_labels=["intent_a"],
                extra_labels=[],
                shown_intents_source={"intent_a": "candidate"}
            )
        
        # Act: Get progress
        service = AnnotationService(temp_db)
        progress = service.get_progress(
            annotator="user1",
            clusters=["cluster1"],
            language="ru"
        )
        
        # Assert: Progress is correct
        assert progress["total"] == 5
        assert progress["done"] == 3
