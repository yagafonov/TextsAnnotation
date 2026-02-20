"""Annotator configuration models."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Annotator(BaseModel):
    """Annotator configuration model."""
    
    name: str = Field(..., min_length=1, description="Annotator username")
    password: str = Field(..., min_length=1, description="Annotator password")
    language: str = Field(..., min_length=2, max_length=10, description="Annotator's working language")
    clusters: List[str] = Field(default_factory=list, description="Allowed clusters for this annotator")
    intents: List[str] = Field(default_factory=list, description="Preferred intents for assignment")
    
    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, v):
        """Normalize language to ISO code.
        
        Accepts both ISO codes ('ru', 'kk') and display names
        ('Русский', 'Казахский') for backwards compatibility.
        """
        _DISPLAY_TO_CODE = {
            "русский": "ru",
            "казахский": "kk",
            "узбекский": "uz",
            "english": "en",
            "английский": "en",
        }
        if isinstance(v, str):
            lowered = v.strip().lower()
            return _DISPLAY_TO_CODE.get(lowered, lowered)
        return v
    
    @field_validator("clusters", mode="before")
    @classmethod
    def normalize_clusters(cls, v):
        """Normalize clusters from various input formats."""
        if isinstance(v, str):
            # Handle comma-separated string
            return [item.strip() for item in v.split(",") if item.strip()]
        elif isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "annotator1",
                "password": "secure_password",
                "language": "ru",
                "clusters": ["general_rus", "card_products_rus"]
            }
        }
    )


class AnnotatorConfig(BaseModel):
    """Root configuration for annotators file."""
    
    annotators: List[Annotator] = Field(default_factory=list)
    
    def get_by_name(self, name: str) -> Optional[Annotator]:
        """Get annotator by name.
        
        Args:
            name: Annotator username
            
        Returns:
            Annotator if found, None otherwise
        """
        for annotator in self.annotators:
            if annotator.name == name:
                return annotator
        return None
