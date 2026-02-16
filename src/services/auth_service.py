"""Authentication service."""

from typing import Optional

from src.models.annotator import Annotator, AnnotatorConfig
from src.utils.logger import logger
from src.utils.yaml_loader import load_yaml_file


class AuthService:
    """Service for authentication operations."""
    
    def __init__(self, annotators_path: str):
        """Initialize authentication service.
        
        Args:
            annotators_path: Path to annotators YAML file
        """
        self.annotators_path = annotators_path
        self._config: Optional[AnnotatorConfig] = None
    
    def load_annotators(self) -> AnnotatorConfig:
        """Load annotators from YAML file.
        
        Returns:
            AnnotatorConfig with all annotators
        """
        data = load_yaml_file(self.annotators_path)
        
        # Handle language mapping (Name -> Code)
        language_map = {
            "Русский": "ru",
            "Казахский": "kk",
            "kazakh": "kk",
            "russian": "ru"
        }
        
        if "annotators" in data:
            for annotator in data["annotators"]:
                # Handle legacy 'languge' typo - convert to 'language'
                if "languge" in annotator and "language" not in annotator:
                    annotator["language"] = annotator.pop("languge")
                    logger.warning(f"Converted legacy 'languge' to 'language' for {annotator.get('name')}")
                
                # Normalize language code if it matches a known name
                if "language" in annotator and annotator["language"] in language_map:
                    original_lang = annotator["language"]
                    annotator["language"] = language_map[original_lang]
                    logger.info(f"Mapped language '{original_lang}' to '{annotator['language']}' for {annotator.get('name')}")
                
                # Handle both 'cluster' and 'clusters' fields
                if "cluster" in annotator and "clusters" not in annotator:
                    annotator["clusters"] = annotator.pop("cluster")
        
        self._config = AnnotatorConfig(**data)
        logger.info(f"Loaded {len(self._config.annotators)} annotators")
        return self._config
    
    def authenticate(self, username: str, password: str) -> Optional[Annotator]:
        """Authenticate an annotator.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Annotator if authenticated, None otherwise
        """
        if not self._config:
            self._config = self.load_annotators()
        
        annotator = self._config.get_by_name(username)
        if annotator and annotator.password == password:
            logger.info(f"Annotator authenticated: {username}")
            return annotator
        
        logger.warning(f"Authentication failed for: {username}")
        return None
    
    def get_annotator(self, username: str) -> Optional[Annotator]:
        """Get annotator by username.
        
        Args:
            username: Username
            
        Returns:
            Annotator if found, None otherwise
        """
        if not self._config:
            self._config = self.load_annotators()
        
        return self._config.get_by_name(username)
