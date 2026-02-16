"""Tests for ML model stub.

This file tests the TopKModelStub used for generating candidate predictions.
"""

import pytest

from src.ml.model_stub import TopKModelStub
from src.models.intent import Intent


class TestTopKModelStub:
    """Tests for TopK model stub."""
    
    @pytest.fixture
    def sample_intents(self):
        """Create sample intents for testing."""
        return {
            'intent_a': Intent(label='intent_a', cluster='cluster1', source_file='test.yaml'),
            'intent_b': Intent(label='intent_b', cluster='cluster1', source_file='test.yaml'),
            'intent_c': Intent(label='intent_c', cluster='cluster2', source_file='test.yaml'),
            'intent_d': Intent(label='intent_d', cluster='cluster2', source_file='test.yaml'),
            'intent_e': Intent(label='intent_e', cluster='cluster3', source_file='test.yaml'),
        }
    
    def test_predict_returns_correct_count(self, sample_intents):
        """Test that predict returns exactly top_k candidates."""
        # Arrange
        model = TopKModelStub(sample_intents, top_k=3, seed=42)
        
        # Act
        candidates = model.predict("Test text")
        
        # Assert
        assert len(candidates) == 3
    
    def test_predict_candidates_ranked(self, sample_intents):
        """Test that candidates are properly ranked 1 to K."""
        # Arrange
        model = TopKModelStub(sample_intents, top_k=3, seed=42)
        
        # Act
        candidates = model.predict("Test text")
        
        # Assert: Ranks are 1, 2, 3
        ranks = [c.rank for c in candidates]
        assert ranks == [1, 2, 3]
    
    def test_predict_probabilities_descending(self, sample_intents):
        """Test that probabilities are in descending order."""
        # Arrange
        model = TopKModelStub(sample_intents, top_k=4, seed=42)
        
        # Act
        candidates = model.predict("Test text")
        
        # Assert: Probabilities decrease
        probabilities = [c.probability for c in candidates]
        for i in range(len(probabilities) - 1):
            assert probabilities[i] >= probabilities[i + 1]
    
    def test_predict_probabilities_in_range(self, sample_intents):
        """Test that all probabilities are between 0 and 1."""
        # Arrange
        model = TopKModelStub(sample_intents, top_k=5, seed=42)
        
        # Act
        candidates = model.predict("Test text")
        
        # Assert: All probabilities between 0 and 1
        for candidate in candidates:
            assert 0.0 <= candidate.probability <= 1.0
    
    def test_predict_reproducible_with_seed(self, sample_intents):
        """Test that predictions are reproducible with same seed."""
        # Arrange: Two models with same seed
        model1 = TopKModelStub(sample_intents, top_k=3, seed=12345)
        model2 = TopKModelStub(sample_intents, top_k=3, seed=12345)
        
        # Act: Get predictions
        candidates1 = model1.predict("Test text")
        candidates2 = model2.predict("Test text")
        
        # Assert: Same predictions
        assert len(candidates1) == len(candidates2)
        for c1, c2 in zip(candidates1, candidates2):
            assert c1.label == c2.label
            assert c1.rank == c2.rank
            assert abs(c1.probability - c2.probability) < 0.0001
    
    def test_predict_different_with_different_seed(self, sample_intents):
        """Test that different seeds produce different results."""
        # Arrange: Two models with different seeds
        model1 = TopKModelStub(sample_intents, top_k=3, seed=1)
        model2 = TopKModelStub(sample_intents, top_k=3, seed=2)
        
        # Act
        candidates1 = model1.predict("Test text")
        candidates2 = model2.predict("Test text")
        
        # Assert: At least some differences (labels or probabilities)
        labels1 = [c.label for c in candidates1]
        labels2 = [c.label for c in candidates2]
        # It's possible (but unlikely) that labels are the same, so check probabilities too
        different = labels1 != labels2
        if not different:
            probs1 = [c.probability for c in candidates1]
            probs2 = [c.probability for c in candidates2]
            different = any(abs(p1 - p2) > 0.001 for p1, p2 in zip(probs1, probs2))
        
        assert different
    
    def test_top_k_larger_than_intents(self, sample_intents):
        """Test that top_k is capped at number of available intents."""
        # Arrange: Request more than available
        model = TopKModelStub(sample_intents, top_k=100, seed=42)
        
        # Act
        candidates = model.predict("Test text")
        
        # Assert: Returns only available intents
        assert len(candidates) == len(sample_intents)
    
    def test_top_k_zero(self):
        """Test edge case of top_k=0."""
        # Arrange
        intents = {'intent_a': Intent(label='intent_a', cluster='c1', source_file='test.yaml')}
        model = TopKModelStub(intents, top_k=0, seed=42)
        
        # Act
        candidates = model.predict("Test text")
        
        # Assert: Returns empty list
        assert len(candidates) == 0
    
    def test_predict_all_labels_from_intents(self, sample_intents):
        """Test that all candidate labels come from provided intents."""
        # Arrange
        model = TopKModelStub(sample_intents, top_k=5, seed=42)
        
        # Act
        candidates = model.predict("Test text")
        
        # Assert: All labels are in sample_intents
        candidate_labels = [c.label for c in candidates]
        for label in candidate_labels:
            assert label in sample_intents
    
    def test_predict_no_duplicate_labels(self, sample_intents):
        """Test that each intent appears at most once in predictions."""
        # Arrange
        model = TopKModelStub(sample_intents, top_k=5, seed=42)
        
        # Act
        candidates = model.predict("Test text")
        
        # Assert: No duplicates
        labels = [c.label for c in candidates]
        assert len(labels) == len(set(labels))
