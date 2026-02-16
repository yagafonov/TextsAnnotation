"""Tests for data models."""

import pytest
from pydantic import ValidationError

from src.models.annotator import Annotator, AnnotatorConfig
from src.models.candidate import Candidate
from src.models.intent import Intent
from src.models.text import Annotation, Text


class TestAnnotator:
    """Tests for Annotator model."""
    
    def test_create_annotator(self):
        """Test creating a valid annotator."""
        annotator = Annotator(
            name="test_user",
            password="test_pass",
            language="ru",
            clusters=["cluster1", "cluster2"]
        )
        
        assert annotator.name == "test_user"
        assert annotator.password == "test_pass"
        assert annotator.language == "ru"
        assert annotator.clusters == ["cluster1", "cluster2"]
    
    def test_language_normalization(self):
        """Test language is normalized to lowercase."""
        annotator = Annotator(
            name="test",
            password="pass",
            language="  RU  ",
            clusters=[]
        )
        
        assert annotator.language == "ru"
    
    def test_clusters_from_string(self):
        """Test clusters can be parsed from comma-separated string."""
        annotator = Annotator(
            name="test",
            password="pass",
            language="ru",
            clusters="cluster1, cluster2, cluster3"  # type: ignore
        )
        
        assert annotator.clusters == ["cluster1", "cluster2", "cluster3"]
    
    def test_invalid_annotator(self):
        """Test validation fails for invalid  data."""
        with pytest.raises(ValidationError):
            Annotator(
                name="",  # Empty name
                password="pass",
                language="ru",
                clusters=[]
            )


class TestAnnotatorConfig:
    """Tests for AnnotatorConfig."""
    
    def test_get_by_name(self, sample_annotators_config):
        """Test getting annotator by name."""
        annotator = sample_annotators_config.get_by_name("annotator1")
        
        assert annotator is not None
        assert annotator.name == "annotator1"
        assert annotator.password == "password1"
    
    def test_get_by_name_not_found(self, sample_annotators_config):
        """Test getting non-existent annotator returns None."""
        annotator = sample_annotators_config.get_by_name("nonexistent")
        
        assert annotator is None


class TestIntent:
    """Tests for Intent model."""
    
    def test_create_intent(self):
        """Test creating a valid intent."""
        intent = Intent(
            label="test_intent",
            description="Test description",
            train=["example 1", "example 2"],
            complexity="low",
            cluster="test_cluster",
            source_file="test.yaml"
        )
        
        assert intent.label == "test_intent"
        assert intent.description == "Test description"
        assert len(intent.train) == 2
        assert intent.cluster == "test_cluster"
    
    def test_intent_defaults(self):
        """Test intent with default values."""
        intent = Intent(label="test")
        
        assert intent.description == ""
        assert intent.train == []
        assert intent.complexity == ""
        assert intent.cluster == "unknown"


class TestCandidate:
    """Tests for Candidate model."""
    
    def test_create_candidate(self):
        """Test creating a valid candidate."""
        candidate = Candidate(
            label="test_intent",
            rank=1,
            probability=0.95
        )
        
        assert candidate.label == "test_intent"
        assert candidate.rank == 1
        assert candidate.probability == 0.95
    
    def test_invalid_rank(self):
        """Test validation fails for invalid rank."""
        with pytest.raises(ValueError, match="Rank must be >= 1"):
            Candidate(label="test", rank=0, probability=0.5)
    
    def test_invalid_probability(self):
        """Test validation fails for invalid probability."""
        with pytest.raises(ValueError, match="Probability must be in"):
            Candidate(label="test", rank=1, probability=1.5)


class TestText:
    """Tests for Text model."""
    
    def test_create_text(self):
        """Test creating a valid text."""
        text = Text(
            text="Test text content",
            language="ru",
            assigned_cluster="cluster1",
            data_version=1
        )
        
        assert text.text == "Test text content"
        assert text.language == "ru"
        assert text.assigned_cluster == "cluster1"
        assert text.data_version == 1
    
    def test_text_validation(self):
        """Test text validation."""
        with pytest.raises(ValidationError):
            Text(text="", language="ru", data_version=1)  # Empty text


class TestAnnotation:
    """Tests for Annotation model."""
    
    def test_create_annotation(self):
        """Test creating a valid annotation."""
        annotation = Annotation(
            text_id=1,
            annotator="test_user",
            label="test_intent",
            decision="yes",
            is_candidate=True
        )
        
        assert annotation.text_id == 1
        assert annotation.annotator == "test_user"
        assert annotation.label == "test_intent"
        assert annotation.decision == "yes"
        assert annotation.is_candidate is True
    
    def test_invalid_decision(self):
        """Test validation fails for invalid decision."""
        with pytest.raises(ValidationError):
            Annotation(
                text_id=1,
                annotator="test",
                label="intent",
                decision="maybe"  # Invalid decision
            )
