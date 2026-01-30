import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

DEFAULT_TOP_K = int(os.environ.get("TEXTS_TOP_K", "5"))


@dataclass
class Candidate:
    label: str
    rank: int
    probability: float


class TopKModelStub:
    def __init__(self, intents: Dict[str, dict], top_k: int = DEFAULT_TOP_K, seed: int = 42):
        self.intents = intents
        self.top_k = min(top_k, len(intents))
        self.random = random.Random(seed)

    def predict(self, text: str) -> List[Candidate]:
        labels = list(self.intents.keys())
        scores = {label: self.random.random() for label in labels}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.top_k]
        return [Candidate(label=label, rank=idx + 1, probability=score) for idx, (label, score) in enumerate(ranked)]


def compute_metrics(
    targets: List[str],
    candidates: List[Candidate],
    margin_threshold: float,
) -> Dict[str, bool]:
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
