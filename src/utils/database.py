"""Thread-safe database connection and initialization."""

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.utils.logger import logger


# Thread safety
_db_lock = threading.RLock()
_dump_thread = None
_dump_stop = threading.Event()


def ensure_parent_dir(path: str) -> None:
    """Ensure parent directory exists for a file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


@contextmanager
def get_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Get a thread-safe database connection.
    
    Args:
        db_path: Path to SQLite database file
        
    Yields:
        SQLite connection with Row factory
    """
    ensure_parent_dir(db_path)
    with _db_lock:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            logger.debug(f"Database connection opened: {db_path}")
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
            logger.debug(f"Database connection closed: {db_path}")


def restore_from_dump(db_path: str, dump_path: str) -> None:
    """Restore database from SQL dump if database doesn't exist.
    
    Args:
        db_path: Path to SQLite database
        dump_path: Path to SQL dump file
    """
    if os.path.exists(db_path):
        logger.info(f"Database already exists: {db_path}")
        return
    
    if not os.path.exists(dump_path):
        logger.info(f"No dump file found: {dump_path}")
        return
    
    logger.info(f"Restoring database from dump: {dump_path}")
    ensure_parent_dir(db_path)
    
    try:
        with sqlite3.connect(db_path) as conn, open(dump_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
            conn.commit()
        logger.info("Database restored successfully")
    except Exception as e:
        logger.error(f"Failed to restore database: {e}")
        raise


def dump_database(db_path: str, dump_path: str) -> None:
    """Create SQL dump of database.
    
    Args:
        db_path: Path to SQLite database
        dump_path: Path to save SQL dump
    """
    ensure_parent_dir(dump_path)
    
    try:
        with sqlite3.connect(db_path) as conn, open(dump_path, "w", encoding="utf-8") as f:
            for line in conn.iterdump():
                f.write(f"{line}\n")
        logger.debug(f"Database dumped to: {dump_path}")
    except Exception as e:
        logger.error(f"Failed to dump database: {e}")


def start_auto_dump(db_path: str, dump_path: str, interval_sec: int) -> None:
    """Start automatic database dumping in background thread.
    
    Args:
        db_path: Path to SQLite database
        dump_path: Path to save SQL dumps
        interval_sec: Dump interval in seconds
    """
    global _dump_thread
    
    if _dump_thread and _dump_thread.is_alive():
        logger.warning("Auto-dump thread already running")
        return
    
    _dump_stop.clear()
    
    def _dump_loop() -> None:
        logger.info(f"Auto-dump started (interval: {interval_sec}s)")
        while not _dump_stop.wait(interval_sec):
            dump_database(db_path, dump_path)
    
    _dump_thread = threading.Thread(target=_dump_loop, daemon=True, name="DatabaseDumper")
    _dump_thread.start()
    logger.info("Auto-dump thread started")


def stop_auto_dump() -> None:
    """Stop the automatic database dumping thread."""
    _dump_stop.set()
    logger.info("Auto-dump stopped")


def init_database(db_path: str) -> None:
    """Initialize database schema with all tables and indexes.
    
    Args:
        db_path: Path to SQLite database
    """
    logger.info(f"Initializing database: {db_path}")
    ensure_parent_dir(db_path)
    
    with get_connection(db_path) as conn:
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                language TEXT,
                clusters TEXT,
                assigned_cluster TEXT,
                assigned_to TEXT,
                data_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                rank INTEGER NOT NULL,
                probability REAL NOT NULL,
                model_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(text_id) REFERENCES texts(id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id INTEGER NOT NULL,
                annotator TEXT NOT NULL,
                label TEXT NOT NULL,
                decision TEXT NOT NULL,
                is_candidate INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(text_id) REFERENCES texts(id)
            )
        """)
        
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS annotations_unique_entry
            ON annotations (text_id, annotator, label)
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS intents (
                label TEXT PRIMARY KEY,
                description TEXT,
                examples TEXT,
                complexity TEXT,
                cluster TEXT,
                source_file TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS skipped_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id INTEGER NOT NULL,
                annotator TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(text_id) REFERENCES texts(id),
                UNIQUE(text_id, annotator)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shown_intents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id INTEGER NOT NULL,
                annotator TEXT NOT NULL,
                label TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(text_id) REFERENCES texts(id)
            )
        """)
        
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS shown_intents_unique_entry
            ON shown_intents (text_id, annotator, label)
        """)
        
        # Performance indexes
        logger.info("Creating performance indexes")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_texts_assigned_cluster ON texts(assigned_cluster)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_texts_language ON texts(language)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_text_id ON annotations(text_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_annotator ON annotations(annotator)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_created ON annotations(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_text_id ON candidates(text_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_text_rank ON candidates(text_id, rank)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_skipped_annotator ON skipped_texts(annotator)")
        
        # Initialize default data
        if not conn.execute("SELECT 1 FROM model_versions LIMIT 1").fetchone():
            logger.info("Initializing default model version")
            conn.execute(
                "INSERT INTO model_versions (version, note, created_at) VALUES (?, ?, datetime('now'))",
                (0, "initial version")
            )
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                ("current_model_version", "0")
            )
        
        conn.commit()
    
    logger.info("Database initialized successfully")
