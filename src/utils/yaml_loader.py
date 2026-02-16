"""YAML file loading utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.logger import logger


def load_yaml_file(path: str) -> Dict[str, Any]:
    """Load and parse a YAML file with error handling.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Parsed YAML data as dictionary, empty dict if file not found or invalid
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            logger.info(f"Loaded YAML file: {path}")
            return data or {}
    except FileNotFoundError:
        logger.warning(f"YAML file not found: {path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {path}: {e}")
        return {}
    except Exception as e:
        logger.exception(f"Unexpected error loading YAML {path}: {e}")
        return {}
