"""ML model stub for TopK predictions."""

import os
import random
from typing import Dict, List, Optional

from src.models.candidate import Candidate


class TopKModelStub:
    """Stub model that generates random TopK predictions.
    
    This is intended for testing/seeding purposes only. Do NOT use
    in production import pipelines — real confidence scores must come
    from an actual NLU model.
    """
    
    def __init__(self, intents: Dict[str, dict], top_k: Optional[int] = None, seed: int = 42):
        """Initialize model stub.
        
        Args:
            intents: Dictionary of available intents
            top_k: Number of top predictions to return. None or 0 means all.
            seed: Random seed for reproducibility
        """
        self.intents = intents
        if top_k is None or top_k <= 0:
            self.top_k = len(intents)
        else:
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


def compute_metrics(
    targets: List[str],
    candidates: List[Candidate],
    margin_threshold: float,
) -> Dict[str, bool]:
    """Compute evaluation metrics for model predictions.
    
    Args:
        targets: Ground-truth intent labels
        candidates: Model-predicted candidates (sorted by confidence)
        margin_threshold: Threshold for margin error rate
        
    Returns:
        Dictionary of metric name -> result
    """
    targets_set = set(targets)
    top1 = candidates[0].label if candidates else None
    top1_hit_rate = top1 in targets_set if top1 else False
    shown_labels = {candidate.label for candidate in candidates}
    topk_coverage = len(targets_set.intersection(shown_labels)) > 0
    margin_error_rate = False
    if len(candidates) > 1:
        margin = candidates[0].probability - candidates[1].probability
        margin_error_rate = margin <= margin_threshold and top1 not in targets_set
    outside_topk = len(targets_set.difference(shown_labels)) > 0
    return {
        "top1_hit_rate": top1_hit_rate,
        "topK_coverage": topk_coverage,
        "margin_error_rate": margin_error_rate,
        "outside_topK_rate": outside_topk,
    }
