"""Centralized configuration management using Pydantic."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""

    # Database settings
    db_path: str = Field(default_factory=lambda: os.getenv("TEXTS_DB_PATH", "data/db/app.db"))
    db_dump_path: str = Field(default_factory=lambda: os.getenv("TEXTS_DB_DUMP_PATH", "data/dumps/backup.sql"))
    db_dump_interval: int = Field(default_factory=lambda: int(os.getenv("TEXTS_DB_DUMP_INTERVAL_SEC", "60")), ge=10)

    # Application settings
    top_k: int = Field(default_factory=lambda: int(os.getenv("TEXTS_TOP_K", "5")), gt=0, le=20)
    margin_threshold: float = Field(default_factory=lambda: float(os.getenv("TEXTS_MARGIN_THRESHOLD", "0.1")), ge=0, le=1)
    probability_threshold: float = Field(default_factory=lambda: float(os.getenv("TEXTS_PROBABILITY_THRESHOLD", "0.1")), ge=0, le=1)
    annotators_intents_confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("ANNOTATORS_INTENTS_CONFIDENCE_THRESHOLD", "0.4")), ge=0, le=1)

    # Paths
    intents_path: str = Field(default_factory=lambda: os.getenv("TEXTS_INTENTS_PATH", "data/intents"))
    annotators_path: str = Field(default_factory=lambda: os.getenv("TEXTS_ANNOTATORS_PATH", "data/annotators.yaml"))
    import_csv_path: str = Field(default_factory=lambda: os.getenv("TEXTS_IMPORT_CSV_PATH", "data/requests.csv"))

    # Admin
    admin_password: str = Field(default_factory=lambda: os.getenv("TEXTS_ADMIN_PASSWORD", "admin123"))

    # Logging
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = Field(default_factory=lambda: os.getenv("LOG_FILE", "logs/app.log"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
