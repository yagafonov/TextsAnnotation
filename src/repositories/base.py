"""Base repository with common database operations."""

import sqlite3
from typing import Any, Dict, List, Optional

from src.utils.database import get_connection
from src.utils.logger import logger


class BaseRepository:
    """Base repository with common database operations."""
    
    def __init__(self, db_path: str):
        """Initialize repository.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
    
    def _execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query and return cursor (for internal use).
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Cursor with results
        """
        with get_connection(self.db_path) as conn:
            return conn.execute(query, params)
    
    def get_setting(self, key: str, default: str = "0") -> str:
        """Get a setting value from database.
        
        Args:
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value
        """
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default
    
    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value in database.
        
        Args:
            key: Setting key
            value: Setting value
        """
        with get_connection(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, value)
            )
            conn.commit()
        logger.info(f"Setting updated: {key} = {value}")
