"""Annotation service for managing annotation workflow."""

from typing import Dict, List, Optional, Tuple

import streamlit as st
from src.models.annotator import Annotator
from src.models.candidate import Candidate
from src.models.text import Text
from src.repositories.annotation_repo import AnnotationRepository
from src.repositories.text_repo import TextRepository
from src.utils.database import get_connection
from src.utils.logger import logger


class AnnotationService:
    """Service for managing annotation workflow."""
    
    def __init__(self, db_path: str):
        """Initialize annotation service.
        
        Args:
            db_path: Path to database
        """
        from src.repositories.intent_repo import IntentRepository
        self.text_repo = TextRepository(db_path)
        self.annotation_repo = AnnotationRepository(db_path)
        self.intent_repo = IntentRepository(db_path)
        self._intent_to_cluster = None # Lazy load mapping
    
    def get_next_text(
        self,
        annotator: str,
        clusters: Optional[List[str]] = None,
        intents: Optional[List[str]] = None,
        language: Optional[str] = None,
        show_skipped: bool = False
    ) -> Optional[dict]:
        """Get next text for annotation.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            intents: Filter by intents
            language: Filter by language
            show_skipped: Show only skipped texts
            
        Returns:
            Text row if found, None otherwise
        """
        texts = self.text_repo.get_unannotated(
            annotator=annotator,
            clusters=clusters,
            intents=intents,
            language=language,
            show_skipped=show_skipped
        )
        
        if not texts:
            return None
        
        return dict(texts[0])

    
    def get_text_with_candidates(self, text_id: int) -> Tuple[Optional[Text], List[Candidate]]:
        """Get text and its candidates.
        
        Args:
            text_id: Text ID
            
        Returns:
            Tuple of (Text, candidates list)
        """
        text = self.text_repo.get_by_id(text_id)
        if not text:
            return None, []
        
        candidates = self.text_repo.get_candidates(text_id)
        return text, candidates
    
    def save_annotations(
        self,
        text_id: int,
        annotator: str,
        decisions: Dict[str, str],
        candidate_labels: List[str],
        extra_labels: List[str],
        shown_intents_source: Dict[str, str]
    ) -> None:
        """Save annotations for a text.
        
        Args:
            text_id: Text ID
            annotator: Annotator name
            decisions: Dict of label -> decision (yes/no)
            candidate_labels: Labels from candidates
            extra_labels: Extra labels added
            shown_intents_source: Source tracking for intents
        """
        self.annotation_repo.save_annotations(
            text_id=text_id,
            annotator=annotator,
            decisions=decisions,
            candidate_labels=candidate_labels,
            extra_labels=extra_labels,
            shown_intents_source=shown_intents_source
        )
        logger.info(f"Saved {len(decisions) + len(extra_labels)} annotations for text#{text_id}")
    
    def skip_text(self, text_id: int, annotator: str) -> None:
        """Skip a text.
        
        Args:
            text_id: Text ID
            annotator: Annotator name
        """
        self.annotation_repo.skip_text(text_id, annotator)
    
    def unskip_text(self, text_id: int, annotator: str) -> None:
        """Unskip a previously skipped text.
        
        Args:
            text_id: Text ID
            annotator: Annotator name
        """
        self.annotation_repo.unskip_text(text_id, annotator)
    
    def get_progress(
        self,
        annotator: str,
        clusters: Optional[List[str]] = None,
        intents: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> dict:
        """Get annotation progress.
        
        Args:
            annotator: Annotator name
            clusters: Filter by clusters
            intents: Filter by intents
            language: Filter by language
            
        Returns:
            Dict with 'total' and 'done' counts
        """
        return self.annotation_repo.get_progress(
            annotator=annotator,
            clusters=clusters,
            intents=intents,
            language=language
        )

    def get_all_texts(
        self,
        annotator: str,
        clusters: Optional[List[str]] = None,
        intents: Optional[List[str]] = None,
        language: Optional[str] = None,
        candidate_label: Optional[str] = None,
        candidate_threshold: float = 0.0,
        candidate_by_cluster: bool = False
    ) -> List[dict]:
        """Get all texts for navigation."""
        return self.text_repo.get_all_texts_for_annotator(
            annotator=annotator,
            clusters=clusters,
            intents=intents,
            language=language,
            candidate_label=candidate_label,
            candidate_threshold=candidate_threshold,
            candidate_by_cluster=candidate_by_cluster
        )

    @st.cache_data
    def get_label_stats(
        _self,
        annotator: str,
        labels: tuple,  # tuple for hashability with st.cache_data
        threshold: float,
        by_cluster: bool = False
    ) -> List[dict]:
        """Get per-intent or per-cluster stats for an annotator (cached)."""
        return _self.text_repo.get_label_stats(
            annotator=annotator,
            labels=list(labels),
            threshold=threshold,
            by_cluster=by_cluster
        )

    def _score_annotators(
        self,
        candidates: List[Candidate],
        annotators: List[Annotator],
        language: str,
    ) -> Dict[str, float]:
        """Score each annotator for a text's candidates.

        Rules:
        - If an annotator has intents defined → score ONLY on intent matches
          (clusters are ignored for that annotator).
        - If an annotator has NO intents defined → score on cluster matches only.
        - Annotators whose language ≠ text language are skipped.
        - Only annotators with score > 0 are included in the result.
        """
        from src.utils.config import settings

        text_lang = (language or "").strip().lower()

        # Load intent-to-cluster mapping once per service instance
        if self._intent_to_cluster is None:
            all_intents = self.intent_repo.get_all()
            self._intent_to_cluster = {
                label: intent.cluster for label, intent in all_intents.items()
            }

        scores: Dict[str, float] = {}

        for annotator in annotators:
            if text_lang != annotator.language:
                continue

            score = 0.0
            use_intents = bool(annotator.intents)

            for candidate in candidates:
                intent_label = candidate.label

                if use_intents:
                    if intent_label in annotator.intents:
                        score += candidate.probability
                else:
                    intent_cluster = self._intent_to_cluster.get(intent_label)
                    if intent_cluster and intent_cluster in annotator.clusters:
                        score += candidate.probability

            if score > 0:
                scores[annotator.name] = score

        return scores

    def calculate_assignment(
        self,
        candidates: List[Candidate],
        annotators: List[Annotator],
        language: str,
        running_counts: Optional[Dict[str, int]] = None,
        rng=None,
    ) -> Optional[str]:
        """Calculate best annotator assignment with intent-first scoring.

        When running_counts + rng are provided (import / bulk-assign contexts),
        ties are broken via load-balancing on running_counts with seeded random.
        When they are absent, the first tied annotator in list order wins.

        Args:
            candidates:     NLU candidate list for the text.
            annotators:     Ordered list of annotators to consider.
            language:       ISO language code of the text.
            running_counts: Mutable dict {annotator_name: texts_assigned_so_far}.
                            Updated in-place when a winner is chosen.
            rng:            Seeded random.Random instance for reproducible ties.
        """
        import random as _random

        scores = self._score_annotators(candidates, annotators, language)
        if not scores:
            return None

        max_score = max(scores.values())
        winners = [
            ann.name
            for ann in annotators          # preserve list order
            if scores.get(ann.name, 0.0) == max_score
        ]

        if len(winners) == 1:
            winner = winners[0]
        elif running_counts is not None:
            # Load-balance among tied annotators
            min_count = min(running_counts.get(name, 0) for name in winners)
            candidates_with_min = [
                name for name in winners
                if running_counts.get(name, 0) == min_count
            ]
            if len(candidates_with_min) == 1:
                winner = candidates_with_min[0]
            else:
                _rng = rng if rng is not None else _random.Random(42)
                winner = _rng.choice(sorted(candidates_with_min))
        else:
            # No counts available — pick first in list order
            winner = winners[0]

        # Update running count in-place so the next call sees fresh numbers
        if running_counts is not None:
            running_counts[winner] = running_counts.get(winner, 0) + 1

        return winner

    def assign_unannotated_texts(
        self, annotators: List[Annotator], seed: int = 42
    ) -> int:
        """Re-assign all unannotated texts with intent-first scoring and tie-breaking.

        Algorithm:
        1. Score every unannotated text against all annotators.
        2. Clear winners  → immediately recorded.
           Ties           → queued as (text_id, bitmask, tied_names).
        3. Tie queue sorted by bitmask so identical competitor sets are batched.
        4. Each batch resolved via load-balancing on running assigned-text counts
           (includes assignments made during this run).  Equal counts → seeded random.
        5. All updates applied in a single transaction.

        Bitmask encoding: n = len(annotators), annotator at index i (0-based in the
        supplied list) sets bit (n-1-i), i.e. the leftmost bit = first annotator.
        Example: [adina, madina, roma, iskander], adina+madina tied → 0b1100 = 12.
        """
        import random

        rng = random.Random(seed)
        n = len(annotators)
        annotator_index = {ann.name: i for i, ann in enumerate(annotators)}

        logger.info("Starting re-assignment of unannotated texts...")

        # --- Phase 0: fetch initial assignment counts from DB ---
        with get_connection(self.text_repo.db_path) as conn:
            count_rows = conn.execute(
                "SELECT assigned_to, COUNT(*) as cnt FROM texts "
                "WHERE assigned_to IS NOT NULL GROUP BY assigned_to"
            ).fetchall()
            running_counts: Dict[str, int] = {
                row["assigned_to"]: row["cnt"] for row in count_rows
            }
            for ann in annotators:
                running_counts.setdefault(ann.name, 0)

            # Fetch all unannotated texts (no annotations at all)
            rows = conn.execute("""
                SELECT t.id, t.language, t.assigned_to
                FROM texts t
                LEFT JOIN annotations a ON a.text_id = t.id
                WHERE a.id IS NULL
            """).fetchall()

        # --- Phase 1: score each text ---
        clear_assignments: List[Tuple[int, str]] = []   # (text_id, winner)
        tie_queue: List[Tuple[int, int, List[str]]] = []  # (text_id, mask, names)

        for row in rows:
            text_id = row["id"]
            language = row["language"]

            candidates = self.text_repo.get_candidates(text_id)
            scores = self._score_annotators(candidates, annotators, language)

            if not scores:
                # Fallback: find all annotators matching the text's language
                text_lang = (language or "").strip().lower()
                lang_matched = [
                    ann.name for ann in annotators
                    if (ann.language or "").strip().lower() == text_lang
                ]
                
                if not lang_matched:
                    # No language match either → leave unassigned
                    continue
                
                # Resolve via load-balancing on language-matched annotators
                min_count = min(running_counts[name] for name in lang_matched)
                candidates_with_min = [
                    name for name in lang_matched if running_counts[name] == min_count
                ]
                
                if len(candidates_with_min) == 1:
                    winner = candidates_with_min[0]
                else:
                    winner = rng.choice(sorted(candidates_with_min))
                
                clear_assignments.append((text_id, winner))
                running_counts[winner] += 1
                continue

            max_score = max(scores.values())
            winners = [
                ann.name
                for ann in annotators              # preserve list order
                if scores.get(ann.name, 0.0) == max_score
            ]

            if len(winners) == 1:
                clear_assignments.append((text_id, winners[0]))
                running_counts[winners[0]] += 1
            else:
                mask = 0
                for name in winners:
                    i = annotator_index[name]
                    mask |= 1 << (n - 1 - i)
                tie_queue.append((text_id, mask, winners))

        # Apply clear assignments
        updates: Dict[int, str] = {}
        for text_id, winner in clear_assignments:
            updates[text_id] = winner

        # --- Phase 2: sort tie queue by bitmask (groups same competitor sets) ---
        tie_queue.sort(key=lambda t: t[1])

        # --- Phase 3: resolve ties with load-balancing ---
        for text_id, _mask, tied_names in tie_queue:
            min_count = min(running_counts[name] for name in tied_names)
            candidates_with_min = [
                name for name in tied_names if running_counts[name] == min_count
            ]

            if len(candidates_with_min) == 1:
                winner = candidates_with_min[0]
            else:
                # Seeded random: sort for determinism, then pick
                winner = rng.choice(sorted(candidates_with_min))

            updates[text_id] = winner
            running_counts[winner] += 1

        # --- Phase 4: apply all updates in a single transaction ---
        updated_count = 0
        if updates:
            with get_connection(self.text_repo.db_path) as conn:
                for text_id, new_assigned in updates.items():
                    row = conn.execute(
                        "SELECT assigned_to FROM texts WHERE id = ?", (text_id,)
                    ).fetchone()
                    if row and row["assigned_to"] != new_assigned:
                        conn.execute(
                            "UPDATE texts SET assigned_to = ? WHERE id = ?",
                            (new_assigned, text_id),
                        )
                        updated_count += 1
                conn.commit()

        logger.info(f"Re-assigned {updated_count} texts")
        return updated_count
