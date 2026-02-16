"""Pytest configuration and fixtures."""

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator, Dict

import pytest

from src.models.annotator import Annotator, AnnotatorConfig
from src.models.intent import Intent
from src.repositories.annotation_repo import AnnotationRepository
from src.repositories.intent_repo import IntentRepository
from src.repositories.text_repo import TextRepository
from src.services.auth_service import AuthService
from src.utils.database import init_database,get_connection


@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name
    
    init_database(db_path)
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_yaml_file() -> Generator[str, None, None]:
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
    
    yield yaml_path
    
    # Cleanup
    if os.path.exists(yaml_path):
        os.unlink(yaml_path)


@pytest.fixture
def sample_annotators_config() -> AnnotatorConfig:
    """Create sample annotators configuration."""
    return AnnotatorConfig(
        annotators=[
            Annotator(
                name="annotator1",
                password="password1",
                language="ru",
                clusters=["cluster1", "cluster2"]
            ),
            Annotator(
                name="annotator2",
                password="password2",
                language="en",
                clusters=["cluster3"]
            )
        ]
    )


@pytest.fixture
def sample_intents() -> Dict[str, Intent]:
    """Create sample intents."""
    return {
        "intent_a": Intent(
            label="intent_a",
            description="Intent A description",
            train=["example 1", "example 2"],
            complexity="low",
            cluster="cluster1",
            source_file="cluster1.yaml"
        ),
        "intent_b": Intent(
            label="intent_b",
            description="Intent B description",
            train=["example 3", "example 4"],
            complexity="medium",
            cluster="cluster2",
            source_file="cluster2.yaml"
        ),
    }


@pytest.fixture
def text_repo(temp_db: str) -> TextRepository:
    """Create a TextRepository instance with temporary database."""
    return TextRepository(temp_db)


@pytest.fixture
def annotation_repo(temp_db: str) -> AnnotationRepository:
    """Create an AnnotationRepository instance with temporary database."""
    return AnnotationRepository(temp_db)


@pytest.fixture
def intent_repo(temp_db: str) -> IntentRepository:
    """Create an IntentRepository instance with temporary database."""
    return IntentRepository(temp_db)
