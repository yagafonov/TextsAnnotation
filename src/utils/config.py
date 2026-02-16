"""Centralized configuration management using Pydantic."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Database settings
    db_path: str = Field(default="data/db/app.db", env="TEXTS_DB_PATH")
    db_dump_path: str = Field(default="data/dumps/backup.sql", env="TEXTS_DB_DUMP_PATH")
    db_dump_interval: int = Field(default=60, env="TEXTS_DB_DUMP_INTERVAL_SEC", ge=10)
    
    # Application settings
    top_k: int = Field(default=5, env="TEXTS_TOP_K", gt=0, le=20)
    margin_threshold: float = Field(default=0.1, env="TEXTS_MARGIN_THRESHOLD", ge=0, le=1)
    probability_threshold: float = Field(default=0.1, env="TEXTS_PROBABILITY_THRESHOLD", ge=0, le=1)
    min_annotators: int = Field(default=1, env="TEXTS_MIN_ANNOTATORS", gt=0)
    
    # Paths
    intents_path: str = Field(default="data/intents", env="TEXTS_INTENTS_PATH")
    annotators_path: str = Field(default="data/annotators.yaml", env="TEXTS_ANNOTATORS_PATH")
    import_csv_path: str = Field(default="data/requests.csv", env="TEXTS_IMPORT_CSV_PATH")
    
    # Admin
    admin_password: str = Field(default="admin123", env="TEXTS_ADMIN_PASSWORD")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
