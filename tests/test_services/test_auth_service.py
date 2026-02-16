"""Tests for services."""

import tempfile

import pytest
import yaml

from src.services.auth_service import AuthService


class TestAuthService:
    """Tests for AuthService."""
    
    def test_load_annotators(self, temp_yaml_file):
        """Test loading annotators from YAML file."""
        # Create annotators YAML
        yaml_content = {
            "annotators": [
                {
                    "name": "user1",
                    "password": "pass1",
                    "language": "ru",
                    "clusters": ["cluster1", "cluster2"]
                },
                {
                    "name": "user2",
                    "password": "pass2",
                    "language": "en",
                    "cluster": "cluster3"  # Testing single cluster field
                }
            ]
        }
        
        with open(temp_yaml_file, "w") as f:
            yaml.dump(yaml_content, f)
        
        # Load annotators
        auth_service = AuthService(temp_yaml_file)
        config = auth_service.load_annotators()
        
        assert len(config.annotators) == 2
        assert config.annotators[0].name == "user1"
        assert config.annotators[0].clusters == ["cluster1", "cluster2"]
        assert config.annotators[1].clusters == ["cluster3"]  # Converted from 'cluster'
    
    def test_load_annotators_with_legacy_languge(self, temp_yaml_file):
        """Test loading annotators with legacy 'languge' field."""
        # Create YAML with typo
        yaml_content = {
            "annotators": [
                {
                    "name": "user1",
                    "password": "pass1",
                    "languge": "ru",  # Legacy typo
                    "clusters": []
                }
            ]
        }
        
        with open(temp_yaml_file, "w") as f:
            yaml.dump(yaml_content, f)
        
        # Load annotators
        auth_service = AuthService(temp_yaml_file)
        config = auth_service.load_annotators()
        
        # Should convert languge -> language
        assert config.annotators[0].language == "ru"
    
    def test_authenticate_success(self, temp_yaml_file):
        """Test successful authentication."""
        yaml_content = {
            "annotators": [
                {
                    "name": "test_user",
                    "password": "test_password",
                    "language": "ru",
                    "clusters": []
                }
            ]
        }
        
        with open(temp_yaml_file, "w") as f:
            yaml.dump(yaml_content, f)
        
        auth_service = AuthService(temp_yaml_file)
        
        # Authenticate
        annotator = auth_service.authenticate("test_user", "test_password")
        
        assert annotator is not None
        assert annotator.name == "test_user"
    
    def test_authenticate_failure_wrong_password(self, temp_yaml_file):
        """Test authentication fails with wrong password."""
        yaml_content = {
            "annotators": [
                {
                    "name": "test_user",
                    "password": "correct_password",
                    "language": "ru",
                    "clusters": []
                }
            ]
        }
        
        with open(temp_yaml_file, "w") as f:
            yaml.dump(yaml_content, f)
        
        auth_service = AuthService(temp_yaml_file)
        
        # Try with wrong password
        annotator = auth_service.authenticate("test_user", "wrong_password")
        
        assert annotator is None
    
    def test_authenticate_failure_nonexistent_user(self, temp_yaml_file):
        """Test authentication fails for nonexistent user."""
        yaml_content = {"annotators": []}
        
        with open(temp_yaml_file, "w") as f:
            yaml.dump(yaml_content, f)
        
        auth_service = AuthService(temp_yaml_file)
        
        annotator = auth_service.authenticate("nonexistent", "password")
        
        assert annotator is None
    
    def test_get_annotator(self, temp_yaml_file):
        """Test getting annotator by username."""
        yaml_content = {
            "annotators": [
                {
                    "name": "test_user",
                    "password": "pass",
                    "language": "ru",
                    "clusters": []
                }
            ]
        }
        
        with open(temp_yaml_file, "w") as f:
            yaml.dump(yaml_content, f)
        
        auth_service = AuthService(temp_yaml_file)
        
        annotator = auth_service.get_annotator("test_user")
        
        assert annotator is not None
        assert annotator.name == "test_user"
        
        # Non-existent user
        assert auth_service.get_annotator("nonexistent") is None
