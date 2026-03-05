"""Centralized configuration management using Pydantic."""

import logging
from pathlib import Path
from typing import Optional

from pydantic import Field, ConfigDict, AliasChoices, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import logger early to ensure it's configured before Settings instantiation
from src.utils.logger import logger as app_logger

# Set up a logger for configuration
logger = logging.getLogger("texts_annotation.config")

# Human-readable language names keyed by ISO code.
# Used across the UI to display language names instead of raw codes.
LANGUAGE_NAMES: dict = {
    "ru": "Русский",
    "kk": "Казахский",
    "uz": "Узбекский",
    "en": "English",
}


class Settings(BaseSettings):
    """Application settings with validation."""

    # Database settings
    db_path: str = Field(default="data/db/app.db", validation_alias=AliasChoices("TEXTS_DB_PATH", "db_path"))
    db_dump_path: str = Field(default="data/dumps/backup.sql", validation_alias=AliasChoices("TEXTS_DB_DUMP_PATH", "db_dump_path"))
    db_dump_interval: int = Field(default=60, validation_alias=AliasChoices("TEXTS_DB_DUMP_INTERVAL_SEC", "db_dump_interval"), ge=10)

    # Application settings
    top_k: int = Field(default=5, validation_alias=AliasChoices("TEXTS_TOP_K", "top_k"), gt=0, le=20)
    margin_threshold: float = Field(default=0.1, validation_alias=AliasChoices("TEXTS_MARGIN_THRESHOLD", "margin_threshold"), ge=0, le=1)
    probability_threshold: float = Field(default=0.1, validation_alias=AliasChoices("TEXTS_PROBABILITY_THRESHOLD", "probability_threshold"), ge=0, le=1)
    annotators_intents_confidence_threshold: float = Field(default=0.4, validation_alias=AliasChoices("ANNOTATORS_INTENTS_CONFIDENCE_THRESHOLD", "annotators_intents_confidence_threshold"), ge=0, le=1)

    @field_validator("db_dump_interval", mode="before")
    @classmethod
    def validate_db_dump_interval(cls, v):
        try:
            val = int(v)
            if val < 10:
                logger.warning(f"TEXTS_DB_DUMP_INTERVAL_SEC ({v}) is too low. Using default: 60")
                return 60
            return val
        except (ValueError, TypeError):
            logger.warning(f"Invalid value for TEXTS_DB_DUMP_INTERVAL_SEC in .env: {v}. Using default: 60")
            return 60

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, v):
        try:
            val = int(v)
            if not (0 < val <= 20):
                logger.warning(f"TEXTS_TOP_K ({v}) out of range (1-20). Using default: 5")
                return 5
            return val
        except (ValueError, TypeError):
            logger.warning(f"Invalid value for TEXTS_TOP_K in .env: {v}. Using default: 5")
            return 5

    @field_validator("margin_threshold", "probability_threshold", "annotators_intents_confidence_threshold", mode="before")
    @classmethod
    def validate_float_thresholds(cls, v, info):
        field_defaults = {
            "margin_threshold": 0.1,
            "probability_threshold": 0.1,
            "annotators_intents_confidence_threshold": 0.4
        }
        env_names = {
            "margin_threshold": "TEXTS_MARGIN_THRESHOLD",
            "probability_threshold": "TEXTS_PROBABILITY_THRESHOLD",
            "annotators_intents_confidence_threshold": "ANNOTATORS_INTENTS_CONFIDENCE_THRESHOLD"
        }
        field_name = info.field_name
        default = field_defaults.get(field_name)
        env_name = env_names.get(field_name)

        try:
            val = float(v)
            if not (0 <= val <= 1):
                logger.warning(f"{env_name} ({v}) out of range (0-1). Using default: {default}")
                return default
            return val
        except (ValueError, TypeError):
            logger.warning(f"Invalid value for {env_name} in .env: {v}. Using default: {default}")
            return default

    @field_validator("intents_path", "annotators_path", mode="before")
    @classmethod
    def validate_paths(cls, v, info):
        field_defaults = {
            "intents_path": "data/intents",
            "annotators_path": "data/annotators.yaml",
        }
        env_names = {
            "intents_path": "TEXTS_INTENTS_PATH",
            "annotators_path": "TEXTS_ANNOTATORS_PATH",
        }
        field_name = info.field_name
        default = field_defaults.get(field_name)
        env_name = env_names.get(field_name)

        if not v or not isinstance(v, str) or not v.strip():
            logger.warning(f"Invalid or empty value for {env_name} in .env: {repr(v)}. Using default: {default}")
            return default

        return v.strip()


    # Paths
    intents_path: str = Field(default="data/intents", validation_alias=AliasChoices("TEXTS_INTENTS_PATH", "intents_path"))
    annotators_path: str = Field(default="data/annotators.yaml", validation_alias=AliasChoices("TEXTS_ANNOTATORS_PATH", "annotators_path"))

    # Admin
    admin_password: str = Field(default="admin123", validation_alias=AliasChoices("TEXTS_ADMIN_PASSWORD", "admin_password"))

    # Logging
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL", "log_level"))
    log_file: str = Field(default="logs/app.log", validation_alias=AliasChoices("LOG_FILE", "log_file"))

    model_config = SettingsConfigDict(
        env_file = Path(__file__).parent.parent.parent / ".env",
        case_sensitive = False,
        extra = "ignore"
    )


# Global settings instance
settings = Settings()
