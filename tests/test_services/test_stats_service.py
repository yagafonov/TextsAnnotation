"""Tests for StatsService.

This file tests the statistics service used by the admin dashboard.
"""

import pytest
import pandas as pd

from src.services.stats_service import StatsService
from src.models.intent import Intent
from src.models.candidate import Candidate


class TestStatsServiceOverall:
    """Tests for overall statistics."""
    
    def test_get_overall_stats_empty_db(self, temp_db):
        """Test overall stats with empty database."""
        # Arrange
        service = StatsService(temp_db)
        
        # Act
        df = service.get_overall_stats()
        
        # Assert
        assert len(df) == 1
        assert df['total_texts'].iloc[0] == 0
        assert df['total_annotators'].iloc[0] == 0
        assert df['total_annotations'].iloc[0] == 0
    
    def test_get_overall_stats_with_data(self, temp_db, text_repo, annotation_repo):
        """Test overall stats with actual data."""
        # Arrange: Create texts and annotations
        text_ids = []
        for i in range(3):
            text_id = text_repo.create(
                text=f"Text {i}",
                language="ru",
                clusters="cluster1",
                assigned_cluster="cluster1",
                data_version=1,
                candidates=[],
                model_version=1
            )
            text_ids.append(text_id)
        
        # Annotate texts
        for text_id in text_ids[:2]:
            annotation_repo.save_annotations(
                text_id=text_id,
                annotator="user1",
                decisions={"intent_a": "yes", "intent_b": "no"},
                candidate_labels=["intent_a", "intent_b"],
                extra_labels=[],
                shown_intents_source={"intent_a": "candidate", "intent_b": "candidate"}
            )
        
        service = StatsService(temp_db)
        
        # Act
        df = service.get_overall_stats()
        
        # Assert
        assert df['total_texts'].iloc[0] == 3
        assert df['total_annotators'].iloc[0] == 1
        assert df['total_annotations'].iloc[0] == 4  # 2 texts * 2 decisions
        assert df['positive_annotations'].iloc[0] == 2  # 2 'yes' decisions
        assert df['fully_annotated_texts'].iloc[0] == 2
        assert df['texts_with_extra_intents'].iloc[0] == 0

class TestStatsServiceActivity:
    """Tests for activity metrics."""
    
    def test_get_daily_activity(self, temp_db, text_repo, annotation_repo):
        """Test daily activity tracking."""
        # Arrange
        text_id = text_repo.create(
            text="Test", 
            language="ru",
            clusters="c1",
            assigned_cluster="c1",
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
        
        service = StatsService(temp_db)
        
        # Act
        df = service.get_daily_activity()
        
        # Assert
        assert len(df) == 1
        assert 'date' in df.columns
        assert df['annotator'].iloc[0] == 'user1'
        assert df['count'].iloc[0] == 1

    def test_get_hourly_activity(self, temp_db, text_repo, annotation_repo):
        """Test hourly activity tracking."""
        # Arrange
        text_id = text_repo.create(
            text="Test", 
            language="ru",
            clusters="c1",
            assigned_cluster="c1",
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
        
        service = StatsService(temp_db)
        
        # Act
        df = service.get_hourly_activity()
        
        # Assert
        assert len(df) == 1
        assert 'hour' in df.columns
        assert df['count'].iloc[0] == 1


class TestStatsServiceAnnotators:
    """Tests for per-annotator statistics."""
    
    def test_get_annotator_stats_empty(self, temp_db):
        """Test annotator stats with no annotations."""
        # Arrange
        service = StatsService(temp_db)
        
        # Act
        df = service.get_annotator_stats()
        
        # Assert: Empty dataframe
        assert len(df) == 0
    
    def test_get_annotator_stats_single_user(self, temp_db, text_repo, annotation_repo):
        """Test annotator stats for single user."""
        # Arrange: Create and annotate text
        text_id = text_repo.create(
            text="Test",
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
            decisions={"intent_a": "yes", "intent_b": "no", "intent_c": "yes"},
            candidate_labels=["intent_a", "intent_b", "intent_c"],
            extra_labels=[],
            shown_intents_source={
                "intent_a": "candidate",
                "intent_b": "candidate",
                "intent_c": "candidate"
            }
        )
        
        service = StatsService(temp_db)
        
        # Act
        df = service.get_annotator_stats()
        
        # Assert
        assert len(df) == 1
        assert df['annotator'].iloc[0] == 'user1'
        assert df['texts_annotated'].iloc[0] == 1
        assert df['total_decisions'].iloc[0] == 3
        assert df['yes_count'].iloc[0] == 2
        assert df['no_count'].iloc[0] == 1
        assert abs(df['yes_rate'].iloc[0] - 2/3) < 0.01
    
    def test_get_annotator_stats_multiple_users(self, temp_db, text_repo, annotation_repo):
        """Test annotator stats for multiple users."""
        # Arrange: Create text annotated by two users
        text_id = text_repo.create(
            text="Test",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # User 1 annotations
        annotation_repo.save_annotations(
            text_id=text_id,
            annotator="user1",
            decisions={"intent_a": "yes"},
            candidate_labels=["intent_a"],
            extra_labels=[],
            shown_intents_source={"intent_a": "candidate"}
        )
        
        # User 2 annotations
        annotation_repo.save_annotations(
            text_id=text_id,
            annotator="user2",
            decisions={"intent_a": "no", "intent_b": "yes"},
            candidate_labels=["intent_a", "intent_b"],
            extra_labels=[],
            shown_intents_source={"intent_a": "candidate", "intent_b": "candidate"}
        )
        
        service = StatsService(temp_db)
        
        # Act
        df = service.get_annotator_stats()
        
        # Assert
        assert len(df) == 2
        annotators = df['annotator'].tolist()
        assert 'user1' in annotators
        assert 'user2' in annotators


class TestStatsServiceIntentQuality:
    """Tests for intent quality metrics."""
    
    def test_get_intent_quality_empty(self, temp_db):
        """Test intent quality with no data."""
        # Arrange
        service = StatsService(temp_db)
        
        # Act
        df = service.get_intent_quality()
        
        # Assert: Empty dataframe
        assert len(df) == 0
    
    def test_get_intent_quality_with_intents(self, temp_db, intent_repo):
        """Test intent quality returns all intents even without annotations."""
        # Arrange: Create intents
        intents = [
            Intent(label="intent_a", cluster="cluster1", source_file="test.yaml"),
            Intent(label="intent_b", cluster="cluster1", source_file="test.yaml")
        ]
        
        for intent in intents:
            intent_repo.upsert(intent)
        
        service = StatsService(temp_db)
        
        # Act
        df = service.get_intent_quality()
        
        # Assert
        assert len(df) == 2
        labels = df['label'].tolist()
        assert 'intent_a' in labels
        assert 'intent_b' in labels

    def test_get_model_quality_legacy(self, temp_db, text_repo, annotation_repo, intent_repo):
        """Test legacy model quality metrics."""
        # Arrange
        intent_repo.upsert(Intent(label="intent_a", cluster="cluster1", source_file="test.yaml"))
        text_id = text_repo.create(
            text="Test", 
            language="ru", 
            clusters="c1",
            assigned_cluster="c1",
            data_version=1,
            candidates=[Candidate(label="intent_a", rank=1, probability=0.9)],
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
        
        service = StatsService(temp_db)
        
        # Act
        df = service.get_model_quality_legacy()
        
        # Assert
        assert len(df) >= 1
        assert 'top1_precision' in df.columns
        assert df[df['label'] == 'intent_a']['top1_yes'].iloc[0] == 1


class TestStatsServiceClusterProgress:
    """Tests for cluster progress calculation."""
    
    def test_get_cluster_progress_empty(self, temp_db):
        """Test cluster progress with no data."""
        # Arrange
        service = StatsService(temp_db)
        
        # Act
        df = service.get_cluster_progress()
        
        # Assert: Empty dataframe
        assert len(df) == 0
    
    def test_get_cluster_progress_with_data(self, temp_db, text_repo, annotation_repo):
        """Test cluster progress calculation."""
        # Arrange: Create texts in different clusters
        cluster1_ids = []
        for i in range(3):
            text_id = text_repo.create(
                text=f"Cluster1 text {i}",
                language="ru",
                clusters="cluster1",
                assigned_cluster="cluster1",
                data_version=1,
                candidates=[],
                model_version=1
            )
            cluster1_ids.append(text_id)
        
        cluster2_ids = []
        for i in range(2):
            text_id = text_repo.create(
                text=f"Cluster2 text {i}",
                language="ru",
                clusters="cluster2",
                assigned_cluster="cluster2",
                data_version=1,
                candidates=[],
                model_version=1
            )
            cluster2_ids.append(text_id)
        
        # Annotate 2 out of 3 cluster1 texts
        for text_id in cluster1_ids[:2]:
            annotation_repo.save_annotations(
                text_id=text_id,
                annotator="user1",
                decisions={"intent_a": "yes"},
                candidate_labels=["intent_a"],
                extra_labels=[],
                shown_intents_source={"intent_a": "candidate"}
            )
        
        # Annotate all cluster2 texts
        for text_id in cluster2_ids:
            annotation_repo.save_annotations(
                text_id=text_id,
                annotator="user1",
                decisions={"intent_b": "yes"},
                candidate_labels=["intent_b"],
                extra_labels=[],
                shown_intents_source={"intent_b": "candidate"}
            )
        
        service = StatsService(temp_db)
        
        # Act
        df = service.get_cluster_progress()
        
        # Assert
        assert len(df) == 2
        
        # Find cluster1 row
        cluster1 = df[df['cluster'] == 'cluster1'].iloc[0]
        assert cluster1['total_texts'] == 3
        assert cluster1['annotated_texts'] == 2
        assert abs(cluster1['completion_rate'] - 2/3) < 0.01
        
        # Find cluster2 row
        cluster2 = df[df['cluster'] == 'cluster2'].iloc[0]
        assert cluster2['total_texts'] == 2
        assert cluster2['annotated_texts'] == 2
        assert cluster2['completion_rate'] == 1.0


class TestStatsServiceDisagreements:
    """Tests for annotation disagreements detection."""
    
    def test_get_disagreements_empty(self, temp_db):
        """Test disagreements with no data."""
        # Arrange
        service = StatsService(temp_db)
        
        # Act
        df = service.get_disagreements()
        
        # Assert: Empty dataframe
        assert len(df) == 0
    
    def test_get_disagreements_detects_conflicts(self, temp_db, text_repo, annotation_repo):
        """Test disagreement detection when annotators disagree."""
        # Arrange: Create text with conflicting annotations
        text_id = text_repo.create(
            text="Disagreement test",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # User1: yes
        annotation_repo.save_annotations(
            text_id=text_id,
            annotator="user1",
            decisions={"intent_a": "yes"},
            candidate_labels=["intent_a"],
            extra_labels=[],
            shown_intents_source={"intent_a": "candidate"}
        )
        
        # User2: no
        annotation_repo.save_annotations(
            text_id=text_id,
            annotator="user2",
            decisions={"intent_a": "no"},
            candidate_labels=["intent_a"],
            extra_labels=[],
            shown_intents_source={"intent_a": "candidate"}
        )
        
        service = StatsService(temp_db)
        
        # Act
        df = service.get_disagreements()
        
        # Assert: Disagreement detected
        assert len(df) > 0
        assert df['text_id'].iloc[0] == text_id
        assert df['label'].iloc[0] == 'intent_a'
        assert df['yes_count'].iloc[0] == 1
        assert df['no_count'].iloc[0] == 1
    
    def test_get_disagreements_ignores_agreement(self, temp_db, text_repo, annotation_repo):
        """Test that perfect agreement is not flagged as disagreement."""
        # Arrange: Create text with agreeing annotations
        text_id = text_repo.create(
            text="Agreement test",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Both users: yes
        for user in ["user1", "user2"]:
            annotation_repo.save_annotations(
                text_id=text_id,
                annotator=user,
                decisions={"intent_a": "yes"},
                candidate_labels=["intent_a"],
                extra_labels=[],
                shown_intents_source={"intent_a": "candidate"}
            )
        
        service = StatsService(temp_db)
        
        # Act
        df = service.get_disagreements()
        
        # Assert: No disagreements
        assert len(df) == 0


class TestStatsServiceExport:
    """Tests for CSV export functionality."""
    
    def test_export_annotations_empty(self, temp_db, tmp_path):
        """Test exporting from empty database."""
        # Arrange
        service = StatsService(temp_db)
        output_path = str(tmp_path / "export.csv")
        
        # Act
        count = service.export_annotations(output_path)
        
        # Assert
        assert count == 0
        assert os.path.exists(output_path)
    
    def test_export_annotations_with_data(self, temp_db, text_repo, annotation_repo, tmp_path):
        """Test exporting annotations to CSV."""
        # Arrange: Create and annotate text
        text_id = text_repo.create(
            text="Export test",
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
            decisions={"intent_a": "yes", "intent_b": "no"},
            candidate_labels=["intent_a", "intent_b"],
            extra_labels=[],
            shown_intents_source={"intent_a": "candidate", "intent_b": "candidate"}
        )
        
        service = StatsService(temp_db)
        output_path = str(tmp_path / "export.csv")
        
        # Act
        count = service.export_annotations(output_path)
        
        # Assert
        assert count == 2  # 2 annotations
        assert os.path.exists(output_path)
        
        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert 'text_id' in df.columns
        assert 'text' in df.columns
        assert 'annotator' in df.columns
        assert 'label' in df.columns
        assert 'decision' in df.columns


# Import os for path checks
import os

class TestStatsServiceOverview:
    """Tests for detailed text overview."""
    
    def test_get_text_detailed_overview(self, temp_db, text_repo, annotation_repo):
        """Test text overview with various filters."""
        # Arrange
        text_id = text_repo.create(
            text="Detailed search text",
            language="ru",
            clusters="c1",
            assigned_cluster="c1",
            data_version=1,
            assigned_to="user1",
            candidates=[Candidate(label="intent_a", rank=1, probability=0.9)],
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
        
        service = StatsService(temp_db)
        
        # Act: Get text detailed overview with multiple filters
        df_annotated = service.get_text_detailed_overview(is_annotated=True)
        df_unannotated = service.get_text_detailed_overview(is_annotated=False)
        df_intent = service.get_text_detailed_overview(top5_intents=["intent_a"])
        df_annotator = service.get_text_detailed_overview(assigned_annotators=["user1"])
        df_search = service.get_text_detailed_overview(search_query="search")
        
        # Assert
        assert len(df_annotated) == 1
        assert len(df_unannotated) == 0
        assert len(df_intent) == 1
        assert len(df_annotator) == 1
        assert len(df_search) == 1

    def test_get_text_count(self, temp_db, text_repo):
        """Test text counting with filters."""
        # Arrange
        text_repo.create(
            text="Count me", 
            language="ru",
            clusters="c1",
            assigned_cluster="c1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        service = StatsService(temp_db)
        
        # Act
        count = service.get_text_count(search_query="Count")
        
        # Assert
        assert count == 1

    def test_get_all_intents(self, temp_db, intent_repo):
        """Test getting all intent labels."""
        # Arrange
        intent_repo.upsert(Intent(label="intent_z", cluster="c", source_file="s"))
        
        service = StatsService(temp_db)
        
        # Act
        intents = service.get_all_intents()
        
        # Assert
        assert "intent_z" in intents

    def test_get_unique_assigned_annotators(self, temp_db, text_repo):
        """Test getting unique assigned annotators."""
        # Arrange
        text_repo.create(
            text="T1", 
            language="ru", 
            assigned_to="annotator_x",
            clusters="c1",
            assigned_cluster="c1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        service = StatsService(temp_db)
        
        # Act
        annotators = service.get_unique_assigned_annotators()
        
        # Assert
        assert "annotator_x" in annotators

    def test_get_text_detailed_overview_extra_filters(self, temp_db, text_repo, annotation_repo):
        """Test detailed overview with human intents and disagreements."""
        # Arrange
        text_id = text_repo.create(
            text="Human intent test",
            language="ru",
            clusters="c1",
            assigned_cluster="c1",
            data_version=1,
            candidates=[Candidate(label="intent_a", rank=1, probability=0.9)],
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
        annotation_repo.save_annotations(
            text_id=text_id,
            annotator="user2",
            decisions={"intent_a": "no"},  # Disagreement here
            candidate_labels=["intent_a"],
            extra_labels=[],
            shown_intents_source={"intent_a": "candidate"}
        )
        
        service = StatsService(temp_db)
        
        # Act
        df_human = service.get_text_detailed_overview(human_intents=["intent_a"])
        df_disagree = service.get_text_detailed_overview(only_disagreements=True)
        
        # Assert
        assert len(df_human) == 1
        assert len(df_disagree) == 1

    def test_get_cluster_progress_filtered(self, temp_db, text_repo):
        """Test cluster progress with filters."""
        text_repo.create(text="T1", language="ru", clusters="c1", assigned_cluster="c1", data_version=1, candidates=[], model_version=1)
        
        service = StatsService(temp_db)
        df = service.get_cluster_progress()
        assert len(df) >= 1
