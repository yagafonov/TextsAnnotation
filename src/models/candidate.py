"""ML candidate model."""

from dataclasses import dataclass


@dataclass
class Candidate:
    """ML model prediction candidate."""
    
    label: str
    rank: int
    probability: float
    
    def __post_init__(self):
        """Validate candidate data."""
        if self.rank < 1:
            raise ValueError(f"Rank must be >= 1, got {self.rank}")
        if not 0 <= self.probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {self.probability}")
