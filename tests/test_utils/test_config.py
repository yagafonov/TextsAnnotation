"""Tests for configuration management."""

import os
from unittest.mock import patch
import pytest
from src.utils.config import Settings


class TestConfig:
    """Tests for Settings model and environment variable loading."""

    def test_default_settings(self):
        """Test that default settings are applied when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            # We pass _env_file=None to avoid loading the actual .env file for this test
            s = Settings(_env_file=None)
            assert s.db_path == "data/db/app.db"
            assert s.top_k == 5
            assert s.margin_threshold == 0.1

    def test_valid_env_vars(self):
        """Test that valid env vars are correctly loaded."""
        with patch.dict(os.environ, {
            "TEXTS_DB_PATH": "custom/db.db",
            "TEXTS_TOP_K": "10",
            "TEXTS_MARGIN_THRESHOLD": "0.5"
        }, clear=True):
            s = Settings(_env_file=None)
            assert s.db_path == "custom/db.db"
            assert s.top_k == 10
            assert s.margin_threshold == 0.5

    def test_invalid_int_falls_back_to_default(self, caplog):
        """Test that invalid integer values fall back to defaults and log a warning."""
        with caplog.at_level("WARNING"):
            with patch.dict(os.environ, {
                "TEXTS_TOP_K": "invalid_int",
                "TEXTS_DB_DUMP_INTERVAL_SEC": "5"  # Too low, should fallback
            }, clear=True):
                s = Settings(_env_file=None)
                
                # Assert fallbacks
                assert s.top_k == 5
                assert s.db_dump_interval == 60
                
                # Check log output
                assert "Invalid value for TEXTS_TOP_K in .env: invalid_int. Using default: 5" in caplog.text
                assert "TEXTS_DB_DUMP_INTERVAL_SEC (5) is too low. Using default: 60" in caplog.text

    def test_invalid_float_falls_back_to_default(self, caplog):
        """Test that invalid float values fall back to defaults and log a warning."""
        with caplog.at_level("WARNING"):
            with patch.dict(os.environ, {
                "TEXTS_MARGIN_THRESHOLD": "not_a_float",
                "TEXTS_PROBABILITY_THRESHOLD": "1.5"  # Out of range
            }, clear=True):
                s = Settings(_env_file=None)
                
                # Assert fallbacks
                assert s.margin_threshold == 0.1
                assert s.probability_threshold == 0.1
                
                # Check log output
                assert "Invalid value for TEXTS_MARGIN_THRESHOLD in .env: not_a_float. Using default: 0.1" in caplog.text
                assert "TEXTS_PROBABILITY_THRESHOLD (1.5) out of range (0-1). Using default: 0.1" in caplog.text

    def test_invalid_path_falls_back_to_default(self, caplog):
        """Test that invalid path values fall back to defaults and log a warning."""
        with caplog.at_level("WARNING"):
            with patch.dict(os.environ, {
                "TEXTS_IMPORT_CSV_PATH": "",  # Empty path
                "TEXTS_ANNOTATORS_PATH": "   "  # Whitespace path
            }, clear=True):
                s = Settings(_env_file=None)
                
                # Assert fallbacks
                assert s.import_csv_path == "data/requests.csv"
                assert s.annotators_path == "data/annotators.yaml"
                
                # Check log output
                assert "Invalid or empty value for TEXTS_IMPORT_CSV_PATH in .env: ''. Using default: data/requests.csv" in caplog.text

    def test_non_existent_path_falls_back_to_default(self, caplog):
        """Test that non-existent import paths fall back to defaults and log a warning."""
        with caplog.at_level("WARNING"):
            with patch.dict(os.environ, {
                "TEXTS_IMPORT_CSV_PATH": "non_existent_file.csv"
            }, clear=True):
                s = Settings(_env_file=None)
                assert s.import_csv_path == "data/requests.csv"
                assert "Path does not exist for TEXTS_IMPORT_CSV_PATH: non_existent_file.csv. Using default: data/requests.csv" in caplog.text

    def test_case_insensitivity(self):
        """Test that env vars are case-insensitive."""
        with patch.dict(os.environ, {
            "texts_top_k": "7"
        }, clear=True):
            s = Settings(_env_file=None)
            assert s.top_k == 7
