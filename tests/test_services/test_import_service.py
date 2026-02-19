"""Tests for ImportService.

This file tests the CSV import functionality which is critical for the data pipeline.
"""

import csv
import os
import tempfile
from typing import Dict

import pytest

from src.models.intent import Intent
from src.services.import_service import ImportService


class TestImportServiceCSV:
    """Tests for CSV import functionality."""
    
    @pytest.fixture
    def sample_csv_path(self):
        """Create a temporary CSV file with valid data."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
            newline=''
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['text', 'language', 'clusters', 'score_intent_a', 'score_intent_b']
            )
            writer.writeheader()
            writer.writerow({
                'text': 'Test text 1',
                'language': 'ru',
                'clusters': 'cluster1',
                'score_intent_a': '0.9',
                'score_intent_b': '0.7'
            })
            writer.writerow({
                'text': 'Test text 2',
                'language': 'en',
                'clusters': 'cluster2',
                'score_intent_a': '0.6',
                'score_intent_b': '0.8'
            })
            csv_path = f.name
        
        yield csv_path
        
        # Cleanup
        if os.path.exists(csv_path):
            os.unlink(csv_path)
    
    @pytest.fixture
    def sample_intents(self) -> Dict[str, Intent]:
        """Create sample intents for testing."""
        return {
            'intent_a': Intent(
                label='intent_a',
                description='Intent A',
                cluster='cluster1',
                source_file='test.yaml'
            ),
            'intent_b': Intent(
                label='intent_b',
                description='Intent B',
                cluster='cluster2',
                source_file='test.yaml'
            ),
            'intent_c': Intent(
                label='intent_c',
                description='Intent C',
                cluster='cluster1',
                source_file='test.yaml'
            )
        }
    
    def test_import_csv_success(self, temp_db, sample_csv_path, sample_intents):
        """Test successful CSV import."""
        # Arrange
        service = ImportService(temp_db)
        
        # Act
        count = service.import_from_csv(
            csv_path=sample_csv_path,
            intents=sample_intents,
            top_k=5,
            model_version=1,
            data_version=1
        )
        
        # Assert
        assert count == 2
    
    def test_import_csv_skips_duplicates(self, temp_db, sample_csv_path, sample_intents, text_repo):
        """Test that duplicate texts are skipped during import."""
        # Arrange: Pre-create one text
        text_repo.create(
            text="Test text 1",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        service = ImportService(temp_db)
        
        # Act: Import CSV with duplicate
        count = service.import_from_csv(
            csv_path=sample_csv_path,
            intents=sample_intents,
            top_k=5,
            model_version=1,
            data_version=1
        )
        
        # Assert: Only 1 new text imported (duplicate skipped)
        assert count == 1
    
    def test_import_csv_file_not_found(self, temp_db, sample_intents):
        """Test handling of missing CSV file."""
        # Arrange
        service = ImportService(temp_db)
        
        # Act
        count = service.import_from_csv(
            csv_path="/nonexistent/file.csv",
            intents=sample_intents,
            top_k=5,
            model_version=1,
            data_version=1
        )
        
        # Assert: No imports, no crash
        assert count == 0
    
    def test_import_csv_empty_file(self, temp_db, sample_intents):
        """Test handling of empty CSV file."""
        # Arrange: Create empty CSV
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write("text,language,clusters\n")  # Header only
            empty_csv = f.name
        
        try:
            service = ImportService(temp_db)
            
            # Act
            count = service.import_from_csv(
                csv_path=empty_csv,
                intents=sample_intents,
                top_k=5,
                model_version=1,
                data_version=1
            )
            
            # Assert
            assert count == 0
        finally:
            if os.path.exists(empty_csv):
                os.unlink(empty_csv)
    
    def test_import_csv_with_empty_text(self, temp_db, sample_intents):
        """Test that rows with empty text are skipped."""
        # Arrange: Create CSV with empty text
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
            newline=''
        ) as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'language'])
            writer.writeheader()
            writer.writerow({'text': '', 'language': 'ru'})  # Empty text
            writer.writerow({'text': '   ', 'language': 'ru'})  # Whitespace only
            writer.writerow({'text': 'Valid text', 'language': 'ru'})
            csv_path = f.name
        
        try:
            service = ImportService(temp_db)
            
            # Act
            count = service.import_from_csv(
                csv_path=csv_path,
                intents=sample_intents,
                top_k=5,
                model_version=1,
                data_version=1
            )
            
            # Assert: Only valid text imported
            assert count == 1
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_import_csv_generates_candidates(self, temp_db, sample_csv_path, sample_intents, text_repo):
        """Test that candidates are generated during import."""
        # Arrange
        service = ImportService(temp_db)
        
        # Act
        count = service.import_from_csv(
            csv_path=sample_csv_path,
            intents=sample_intents,
            top_k=3,
            model_version=1,
            data_version=1
        )
        
        # Assert: Verify candidates were created
        assert count == 2
        
        # Get the first imported text
        from src.repositories.text_repo import TextRepository
        repo = TextRepository(temp_db)
        texts = repo.get_unannotated(
            annotator="test",
            clusters=None,
            language=None
        )
        
        if texts:
            candidates = repo.get_candidates(texts[0]['id'])
            assert len(candidates) > 0
            assert len(candidates) <= 3  # top_k=3


class TestImportServiceClusterDetermination:
    """Tests for cluster determination logic."""
    
    @pytest.fixture
    def sample_intents(self) -> Dict[str, Intent]:
        """Create sample intents with different clusters."""
        return {
            'intent_a': Intent(label='intent_a', cluster='cluster1', source_file='test.yaml'),
            'intent_b': Intent(label='intent_b', cluster='cluster2', source_file='test.yaml'),
            'intent_c': Intent(label='intent_c', cluster='cluster1', source_file='test.yaml'),
        }
    
    def test_determine_cluster_from_explicit_field(self, temp_db, sample_intents):
        """Test cluster determination from explicit clusters field."""
        # Arrange: CSV with explicit clusters
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
            newline=''
        ) as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'clusters'])
            writer.writeheader()
            writer.writerow({'text': 'Test', 'clusters': 'cluster1,cluster2'})
            csv_path = f.name
        
        try:
            service = ImportService(temp_db)
            
            # Act
            count = service.import_from_csv(
                csv_path=csv_path,
                intents=sample_intents,
                top_k=5,
                model_version=1,
                data_version=1
            )
            
            # Assert
            assert count == 1
            
            # Verify cluster assignment
            from src.repositories.text_repo import TextRepository
            repo = TextRepository(temp_db)
            texts = repo.get_unannotated(
                annotator="test",
                clusters=None,
                language=None
            )
            assert texts[0]['assigned_cluster'] == 'cluster1'  # First cluster
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_determine_cluster_from_scores(self, temp_db, sample_intents):
        """Test cluster determination from intent scores."""
        # Arrange: CSV with score columns
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
            newline=''
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['text', 'score_intent_a', 'score_intent_b', 'score_intent_c']
            )
            writer.writeheader()
            # intent_b (cluster2) has highest score, but intent_a + intent_c (cluster1) sum higher
            writer.writerow({
                'text': 'Test',
                'score_intent_a': '0.4',
                'score_intent_b': '0.5',
                'score_intent_c': '0.3'
            })
            csv_path = f.name
        
        try:
            service = ImportService(temp_db)
            
            # Act
            count = service.import_from_csv(
                csv_path=csv_path,
                intents=sample_intents,
                top_k=5,
                model_version=1,
                data_version=1
            )
            
            # Assert
            assert count == 1
            
            # Verify cluster is cluster1 (0.4 + 0.3 = 0.7 > 0.5)
            from src.repositories.text_repo import TextRepository
            repo = TextRepository(temp_db)
            texts = repo.get_unannotated(
                annotator="test",
                clusters=None,
                language=None,
                min_annotators=1
            )
            assert texts[0]['assigned_cluster'] == 'cluster1'
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_determine_cluster_defaults_to_unknown(self, temp_db, sample_intents):
        """Test cluster defaults to 'unknown' when not determinable."""
        # Arrange: CSV without clusters or scores
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
            newline=''
        ) as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'language'])
            writer.writeheader()
            writer.writerow({'text': 'Test', 'language': 'ru'})
            csv_path = f.name
        
        try:
            service = ImportService(temp_db)
            
            # Act
            count = service.import_from_csv(
                csv_path=csv_path,
                intents=sample_intents,
                top_k=5,
                model_version=1,
                data_version=1
            )
            
            # Assert
            assert count == 1
            
            # Verify cluster is 'unknown'
            from src.repositories.text_repo import TextRepository
            repo = TextRepository(temp_db)
            texts = repo.get_unannotated(
                annotator="test",
                clusters=None,
                language=None
            )
            assert texts[0]['assigned_cluster'] == 'unknown'
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


class TestImportServiceEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.fixture
    def sample_intents(self) -> Dict[str, Intent]:
        """Create sample intents."""
        return {
            'intent_a': Intent(label='intent_a', cluster='cluster1', source_file='test.yaml')
        }
    
    def test_import_csv_with_unicode(self, temp_db, sample_intents):
        """Test importing CSV with Unicode characters."""
        # Arrange: CSV with Unicode
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
            newline=''
        ) as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'language'])
            writer.writeheader()
            writer.writerow({'text': 'Привет мир! 🌍', 'language': 'ru'})
            writer.writerow({'text': '你好世界', 'language': 'zh'})
            csv_path = f.name
        
        try:
            service = ImportService(temp_db)
            
            # Act
            count = service.import_from_csv(
                csv_path=csv_path,
                intents=sample_intents,
                top_k=5,
                model_version=1,
                data_version=1
            )
            
            # Assert: Both texts imported successfully
            assert count == 2
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_import_csv_with_quoted_fields(self, temp_db, sample_intents):
        """Test importing CSV with quoted fields containing commas."""
        # Arrange: CSV with quoted fields
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
            newline=''
        ) as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'clusters'])
            writer.writeheader()
            writer.writerow({'text': 'Text with, comma', 'clusters': 'cluster1,cluster2'})
            csv_path = f.name
        
        try:
            service = ImportService(temp_db)
            
            # Act
            count = service.import_from_csv(
                csv_path=csv_path,
                intents=sample_intents,
                top_k=5,
                model_version=1,
                data_version=1
            )
            
            # Assert
            assert count == 1
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_import_csv_preserves_data_version(self, temp_db, sample_intents, text_repo):
        """Test that data version is correctly stored."""
        # Arrange
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
            newline=''
        ) as f:
            writer = csv.DictWriter(f, fieldnames=['text'])
            writer.writeheader()
            writer.writerow({'text': 'Version test'})
            csv_path = f.name
        
        try:
            service = ImportService(temp_db)
            
            # Act: Import with data_version=5
            count = service.import_from_csv(
                csv_path=csv_path,
                intents=sample_intents,
                top_k=5,
                model_version=1,
                data_version=5
            )
            
            # Assert
            assert count == 1
            
            # Verify data version
            texts = text_repo.get_unannotated(
                annotator="test",
                clusters=None,
                language=None
            )
            assert texts[0]['data_version'] == 5
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
