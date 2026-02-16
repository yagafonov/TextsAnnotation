"""Intent data models."""

from typing import List, Optional

from pydantic import BaseModel, Field


class Intent(BaseModel):
    """Intent definition model."""
    
    label: str = Field(..., min_length=1, description="Intent label/identifier")
    description: str = Field(default="", description="Intent description")
    train: List[str] = Field(default_factory=list, description="Training examples")
    complexity: str = Field(default="", description="Intent complexity level")
    cluster: str = Field(default="unknown", description="Intent cluster/category")
    source_file: Optional[str] = Field(default=None, description="Source YAML file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "transfer_money",
                "description": "User wants to transfer money",
                "train": ["перевести деньги", "отправить платеж"],
                "complexity": "medium",
                "cluster": "transactions"
            }
        }


class IntentDatabase(BaseModel):
    """Intent as stored in database."""
    
    label: str
    description: str
    examples: str  # Comma-separated examples
    complexity: str
    cluster: str
    source_file: str
    updated_at: str
    
    def to_intent(self) -> Intent:
        """Convert database model to Intent model."""
        return Intent(
            label=self.label,
            description=self.description,
            train=[ex.strip() for ex in self.examples.split(",") if ex.strip()],
            complexity=self.complexity,
            cluster=self.cluster,
            source_file=self.source_file
        )
