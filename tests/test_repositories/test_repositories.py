"""Tests for repositories."""

import tempfile
from datetime import datetime

import pytest

from src.models.candidate import Candidate
from src.models.intent import Intent
from src.models.text import Text
from src.utils.database import init_database


class TestTextRepository:
    """Tests for TextRepository."""
    
    def test_create_text(self, text_repo):
        """Test creating a text with candidates."""
        candidates = [
            Candidate(label="intent_a", rank=1, probability=0.9),
            Candidate(label="intent_b", rank=2, probability=0.7)
        ]
        
        text_id = text_repo.create(
            text="Test text",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=candidates,
            model_version=1
        )
        
        assert text_id > 0
        
        # Verify text was created
        text = text_repo.get_by_id(text_id)
        assert text is not None
        assert text.text == "Test text"
        assert text.language == "ru"
        
        # Verify candidates were created
        stored_candidates = text_repo.get_candidates(text_id)
        assert len(stored_candidates) == 2
        assert stored_candidates[0].rank == 1
        assert stored_candidates[1].rank == 2
    
    def test_text_exists(self, text_repo):
        """Test checking if text exists."""
        assert not text_repo.exists("Nonexistent text")
        
        # Create a text
        text_repo.create(
            text="Unique text",
            language="ru",
            clusters=None,
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        assert text_repo.exists("Unique text")
    
    def test_get_unannotated(self, text_repo):
        """Test getting unannotated texts."""
        # Create some texts
        for i in range(3):
            text_repo.create(
                text=f"Text {i}",
                language="ru",
                clusters="cluster1",
                assigned_cluster="cluster1",
                data_version=1,
                candidates=[],
                model_version=1
            )
        
        # Get unannotated texts
        texts = text_repo.get_unannotated(
            annotator="test_annotator",
            clusters=["cluster1"],
            language="ru"
        )
        
        assert len(texts) == 3


class TestAnnotationRepository:
    """Tests for AnnotationRepository."""
    
    def test_save_annotations(self, annotation_repo, text_repo):
        """Test saving annotations."""
        # Create a text first
        text_id = text_repo.create(
            text="Test text",
            language="ru",
            clusters=None,
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Save annotations
        annotation_repo.save_annotations(
            text_id=text_id,
            annotator="test_user",
            decisions={"intent_a": "yes", "intent_b": "no"},
            candidate_labels=["intent_a", "intent_b"],
            extra_labels=["intent_c"],
            shown_intents_source={"intent_a": "candidate", "intent_b": "candidate"}
        )
        
        # Verify annotations were created
        annotations = annotation_repo.get_annotations_for_text(text_id)
        assert len(annotations) == 3  # 2 decisions + 1 extra
        
        # Check decisions
        decisions_map = {a.label: a.decision for a in annotations}
        assert decisions_map.get("intent_a") == "yes"
        assert decisions_map.get("intent_b") == "no"
        assert decisions_map.get("intent_c") == "yes"  # Extra label defaults to yes
    
    def test_skip_text(self, annotation_repo, text_repo):
        """Test skipping a text."""
        text_id = text_repo.create(
            text="Test text",
            language="ru",
            clusters=None,
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Skip text
        annotation_repo.skip_text(text_id, "test_user")
        
        # Try to skip again (should not error due to UNIQUE constraint)
        annotation_repo.skip_text(text_id, "test_user")
    
    def test_unskip_text(self, annotation_repo, text_repo):
        """Test unskipping a text."""
        text_id = text_repo.create(
            text="Test text",
            language="ru",
            clusters=None,
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Skip then unskip
        annotation_repo.skip_text(text_id, "test_user")
        annotation_repo.unskip_text(text_id, "test_user")
        
        # Unskipping again should not error
        annotation_repo.unskip_text(text_id, "test_user")
    
    def test_get_progress(self, annotation_repo, text_repo):
        """Test getting annotation progress."""
        # Create texts
        for i in range(5):
            text_id = text_repo.create(
                text=f"Text {i}",
                language="ru",
                clusters="cluster1",
                assigned_cluster="cluster1",
                data_version=1,
                candidates=[],
                model_version=1
            )
            
            # Annotate first 3
            if i < 3:
                annotation_repo.save_annotations(
                    text_id=text_id,
                    annotator="test_user",
                    decisions={"intent_a": "yes"},
                    candidate_labels=["intent_a"],
                    extra_labels=[],
                    shown_intents_source={"intent_a": "candidate"}
                )
        
        progress = annotation_repo.get_progress(
            annotator="test_user",
            clusters=["cluster1"],
            language="ru"
        )
        
        assert progress["total"] == 5
        assert progress["done"] == 3


class TestIntentRepository:
    """Tests for IntentRepository."""
    
    def test_upsert_intent(self, intent_repo):
        """Test inserting and updating an intent."""
        intent = Intent(
            label="test_intent",
            description="Test description",
            train=["example 1", "example 2"],
            complexity="low",
            cluster="cluster1",
            source_file="test.yaml"
        )
        
        # Insert
        intent_repo.upsert(intent)
        
        # Retrieve
        retrieved = intent_repo.get_by_label("test_intent")
        assert retrieved is not None
        assert retrieved.label == "test_intent"
        assert retrieved.description == "Test description"
        assert len(retrieved.train) == 2
        
        # Update
        intent.description = "Updated description"
        intent_repo.upsert(intent)
        
        # Retrieve again
        retrieved = intent_repo.get_by_label("test_intent")
        assert retrieved.description == "Updated description"
    
    def test_get_by_cluster(self, intent_repo):
        """Test getting intents by cluster."""
        intents = [
            Intent(label="intent_a", cluster="cluster1", source_file="test.yaml"),
            Intent(label="intent_b", cluster="cluster1", source_file="test.yaml"),
            Intent(label="intent_c", cluster="cluster2", source_file="test.yaml"),
        ]
        
        for intent in intents:
            intent_repo.upsert(intent)
        
        cluster1_intents = intent_repo.get_by_cluster("cluster1")
        assert len(cluster1_intents) == 2
        
        cluster2_intents = intent_repo.get_by_cluster("cluster2")
        assert len(cluster2_intents) == 1
    
    def test_get_all(self, intent_repo):
        """Test getting all intents."""
        intents = [
            Intent(label=f"intent_{i}", cluster="cluster1", source_file="test.yaml")
            for i in range(5)
        ]
        
        for intent in intents:
            intent_repo.upsert(intent)
        
        all_intents = intent_repo.get_all()
        assert len(all_intents) == 5
    
    def test_load_from_yaml(self, intent_repo, temp_yaml_file):
        """Test loading intents from YAML file."""
        import yaml
        
        # Create YAML file
        yaml_content = {
            "intent_a": {
                "description": "Intent A",
                "train": ["example 1", "example 2"],
                "complexity": "low"
            },
            "intent_b": {
                "description": "Intent B",
                "train": ["example 3"],
                "complexity": "high"
            }
        }
        
        with open(temp_yaml_file, "w") as f:
            yaml.dump(yaml_content, f)
        
        # Load from YAML
        intents = intent_repo.load_from_yaml(temp_yaml_file)
        
        assert len(intents) == 2
        assert "intent_a" in intents
        assert "intent_b" in intents
        assert intents["intent_a"].description == "Intent A"
        assert len(intents["intent_a"].train) == 2
