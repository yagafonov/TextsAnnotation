"""Integration tests for complete workflows.

These tests verify end-to-end functionality across multiple components.
"""

import tempfile

import pytest
import yaml

from src.models.intent import Intent
from src.repositories.intent_repo import IntentRepository
from src.services.annotation_service import AnnotationService
from src.services.auth_service import AuthService
from src.services.import_service import ImportService


class TestCompleteAnnotationWorkflow:
    """Test end-to-end annotation workflow."""
    
    @pytest.fixture
    def annotators_yaml(self):
        """Create temporary annotators configuration."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yaml',
            delete=False,
            encoding='utf-8'
        ) as f:
            yaml.dump({
                'annotators': [
                    {
                        'name': 'annotator1',
                        'password': 'pass1',
                        'language': 'ru',
                        'clusters': ['cluster1']
                    },
                    {
                        'name': 'annotator2',
                        'password': 'pass2',
                        'language': 'ru',
                        'clusters': ['cluster1']
                    }
                ]
            }, f)
            path = f.name
        
        yield path
        
        import os
        if os.path.exists(path):
            os.unlink(path)
    
    def test_full_annotation_cycle(self, temp_db, annotators_yaml):
        """Test complete cycle: authenticate -> get text -> annotate -> check progress."""
        # Step 1: Setup services
        auth_service = AuthService(annotators_yaml)
        annotation_service = AnnotationService(temp_db)
        intent_repo = IntentRepository(temp_db)
        
        # Step 2: Add intents to database
        intents = {
            'intent_a': Intent(label='intent_a', cluster='cluster1', source_file='test.yaml'),
            'intent_b': Intent(label='intent_b', cluster='cluster1', source_file='test.yaml')
        }
        for intent in intents.values():
            intent_repo.upsert(intent)
        
        # Step 3: Create test text
        from src.repositories.text_repo import TextRepository
        text_repo = TextRepository(temp_db)
        text_id = text_repo.create(
            text="Integration test text",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Step 4: Authenticate first user
        annotator1 = auth_service.authenticate('annotator1', 'pass1')
        assert annotator1 is not None
        assert annotator1.name == 'annotator1'
        
        # Step 5: Get next text for annotation
        next_text = annotation_service.get_next_text(
            annotator='annotator1',
            clusters=['cluster1'],
            language='ru'
        )
        assert next_text is not None
        assert next_text['id'] == text_id
        
        # Step 6: Save annotations from first user
        annotation_service.save_annotations(
            text_id=text_id,
            annotator='annotator1',
            decisions={'intent_a': 'yes', 'intent_b': 'no'},
            candidate_labels=['intent_a', 'intent_b'],
            extra_labels=[],
            shown_intents_source={'intent_a': 'candidate', 'intent_b': 'candidate'}
        )
        
        # Step 7: Check progress for first user
        progress = annotation_service.get_progress(
            annotator='annotator1',
            clusters=['cluster1'],
            language='ru'
        )
        assert progress['done'] == 1
        assert progress['total'] == 1
        
        # Step 8: Authenticate second user
        annotator2 = auth_service.authenticate('annotator2', 'pass2')
        assert annotator2 is not None
        
        # Step 9: Second user should see the same text (needs 2 annotators)
        next_text_2 = annotation_service.get_next_text(
            annotator='annotator2',
            clusters=['cluster1'],
            language='ru',
            min_annotators=2
        )
        assert next_text_2 is not None
        assert next_text_2['id'] == text_id
        
        # Step 10: Second user annotates
        annotation_service.save_annotations(
            text_id=text_id,
            annotator='annotator2',
            decisions={'intent_a': 'yes', 'intent_b': 'yes'},
            candidate_labels=['intent_a', 'intent_b'],
            extra_labels=[],
            shown_intents_source={'intent_a': 'candidate', 'intent_b': 'candidate'}
        )
        
        # Step 11: Verify both users have completed
        progress1 = annotation_service.get_progress(
            annotator='annotator1',
            clusters=['cluster1'],
            language='ru'
        )
        progress2 = annotation_service.get_progress(
            annotator='annotator2',
            clusters=['cluster1'],
            language='ru'
        )
        assert progress1['done'] == 1
        assert progress2['done'] == 1


class TestImportToAnnotationWorkflow:
    """Test workflow from CSV import to annotation."""
    
    def test_import_then_annotate(self, temp_db):
        """Test importing CSV data and then annotating it."""
        # Step 1: Setup services
        import_service = ImportService(temp_db)
        annotation_service = AnnotationService(temp_db)
        intent_repo = IntentRepository(temp_db)
        
        # Step 2: Create intents
        intents = {
            'intent_a': Intent(label='intent_a', cluster='cluster1', source_file='test.yaml'),
            'intent_b': Intent(label='intent_b', cluster='cluster1', source_file='test.yaml')
        }
        for intent in intents.values():
            intent_repo.upsert(intent)
        
        # Step 3: Create CSV file
        import csv
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
            newline=''
        ) as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'language', 'clusters'])
            writer.writeheader()
            writer.writerow({
                'text': 'Imported text for annotation',
                'language': 'ru',
                'clusters': 'cluster1'
            })
            csv_path = f.name
        
        try:
            # Step 4: Import CSV
            imported_count = import_service.import_from_csv(
                csv_path=csv_path,
                intents=intents,
                top_k=5,
                model_version=1,
                data_version=1
            )
            assert imported_count == 1
            
            # Step 5: Get the imported text for annotation
            next_text = annotation_service.get_next_text(
                annotator='test_user',
                clusters=['cluster1'],
                language='ru'
            )
            assert next_text is not None
            assert next_text['text'] == 'Imported text for annotation'
            
            # Step 6: Get candidates (should be generated during import)
            text, candidates = annotation_service.get_text_with_candidates(next_text['id'])
            assert text is not None
            assert len(candidates) > 0
            
            # Step 7: Annotate the imported text
            annotation_service.save_annotations(
                text_id=next_text['id'],
                annotator='test_user',
                decisions={'intent_a': 'yes'},
                candidate_labels=[c.label for c in candidates],
                extra_labels=[],
                shown_intents_source={c.label: 'candidate' for c in candidates}
            )
            
            # Step 8: Verify annotation was saved
            from src.repositories.annotation_repo import AnnotationRepository
            annotation_repo = AnnotationRepository(temp_db)
            annotations = annotation_repo.get_annotations_for_text(next_text['id'])
            assert len(annotations) > 0
            
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


class TestSkipWorkflow:
    """Test skip/unskip workflow."""
    
    def test_skip_and_unskip_cycle(self, temp_db, text_repo):
        """Test skipping a text and then returning to it."""
        # Step 1: Setup
        annotation_service = AnnotationService(temp_db)
        
        # Step 2: Create text
        text_id = text_repo.create(
            text="Difficult text to skip",
            language="ru",
            clusters="cluster1",
            assigned_cluster="cluster1",
            data_version=1,
            candidates=[],
            model_version=1
        )
        
        # Step 3: Get text
        text = annotation_service.get_next_text(
            annotator='user1',
            clusters=['cluster1'],
            language='ru'
        )
        assert text is not None
        assert text['id'] == text_id
        
        # Step 4: Skip the text
        annotation_service.skip_text(text_id, 'user1')
        
        # Step 5: Verify it's not in normal queue
        next_text = annotation_service.get_next_text(
            annotator='user1',
            clusters=['cluster1'],
            language='ru',
            show_skipped=False
        )
        assert next_text is None
        
        # Step 6: Verify it appears in skipped queue
        skipped_text = annotation_service.get_next_text(
            annotator='user1',
            clusters=['cluster1'],
            language='ru',
            show_skipped=True
        )
        assert skipped_text is not None
        assert skipped_text['id'] == text_id
        
        # Step 7: Unskip the text
        annotation_service.unskip_text(text_id, 'user1')
        
        # Step 8: Verify it's back in normal queue
        text_again = annotation_service.get_next_text(
            annotator='user1',
            clusters=['cluster1'],
            language='ru',
            show_skipped=False
        )
        assert text_again is not None
        assert text_again['id'] == text_id
