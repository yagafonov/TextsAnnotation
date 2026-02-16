"""ML model stub for TopK predictions."""

import os
import random
from typing import Dict, List

from src.models.candidate import Candidate


class TopKModelStub:
    """Stub model that generates random TopK predictions."""
    
    def __init__(self, intents: Dict[str, dict], top_k: int = 5, seed: int = 42):
        """Initialize model stub.
        
        Args:
            intents: Dictionary of available intents
            top_k: Number of top predictions to return
            seed: Random seed for reproducibility
        """
        self.intents = intents
        self.top_k = min(top_k, len(intents))
        self.random = random.Random(seed)
    
    def predict(self, text: str) -> List[Candidate]:
        """Generate random TopK predictions.
        
        Args:
            text: Input text
            
        Returns:
            List of Candidate predictions
        """
        labels = list(self.intents.keys())
        scores = {label: self.random.random() for label in labels}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.top_k]
        
        return [
            Candidate(label=label, rank=idx + 1, probability=score)
            for idx, (label, score) in enumerate(ranked)
        ]
