"""Text and annotation models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Text(BaseModel):
    """Text model."""
    
    id: Optional[int] = Field(default=None, description="Text ID (auto-assigned)")
    text: str = Field(..., min_length=1, max_length=10000, description="Text content")
    language: Optional[str] = Field(default=None, max_length=10, description="Text language")
    clusters: Optional[str] = Field(default=None, description="Associated clusters")
    assigned_cluster: Optional[str] = Field(default=None, description="Primary assigned cluster")
    data_version: int = Field(default=0, description="Data version")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Как перевести деньги?",
                "language": "ru",
                "assigned_cluster": "transactions",
                "data_version": 1
            }
        }


class Annotation(BaseModel):
    """Annotation model."""
    
    id: Optional[int] = Field(default=None)
    text_id: int = Field(..., description="Referenced text ID")
    annotator: str = Field(..., min_length=1, description="Annotator name")
    label: str = Field(..., min_length=1, description="Intent label")
    decision: str = Field(..., pattern="^(yes|no)$", description="Annotation decision") 
    is_candidate: bool = Field(default=False, description="Was this from candidates")
    created_at: Optional[str] = Field(default=None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text_id": 1,
                "annotator": "annotator1",
                "label": "transfer_money",
                "decision": "yes",
                "is_candidate": True
            }
        }


class SkippedText(BaseModel):
    """Skipped text model."""
    
    id: Optional[int] = Field(default=None)
    text_id: int
    annotator: str
    created_at: Optional[str] = Field(default=None)
