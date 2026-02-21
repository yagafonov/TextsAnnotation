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

            # Backfill assigned_to from actual annotators in imported dump
            conn.execute("""
                UPDATE texts SET assigned_to = (
                    SELECT GROUP_CONCAT(DISTINCT annotator)
                    FROM annotations WHERE text_id = texts.id
                )
                WHERE (assigned_to IS NULL OR assigned_to = '')
                  AND id IN (SELECT DISTINCT text_id FROM annotations)
            """)

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


def ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> bool:
    """Ensure a column exists in a table.
    
    Returns:
        True if column was added, False if it already existed.
    """
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row["name"] for row in cursor.fetchall()]
    if column not in columns:
        logger.info(f"Adding missing column '{column}' to table '{table}'")
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
        return True
    return False


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
                text TEXT NOT NULL UNIQUE,
                language TEXT,
                clusters TEXT,
                assigned_cluster TEXT,
                assigned_to TEXT,
                is_skipped INTEGER NOT NULL DEFAULT 0,
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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                annotator TEXT PRIMARY KEY,
                dark_mode TEXT,
                layout_mode TEXT,
                selected_label TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        
        # Schema Migrations (Compatibility with legacy dumps)
        logger.debug("Checking for schema migrations")
        migration_applied = False
        try:
            # Table 'texts' migrations
            if ensure_column(conn, "texts", "assigned_cluster", "TEXT"): migration_applied = True
            if ensure_column(conn, "texts", "language", "TEXT"): migration_applied = True
            if ensure_column(conn, "texts", "assigned_to", "TEXT"): migration_applied = True
            if ensure_column(conn, "texts", "is_skipped", "INTEGER NOT NULL DEFAULT 0"): migration_applied = True

            # Backfill is_skipped from skipped_texts table
            conn.execute("""
                UPDATE texts SET is_skipped = 1
                WHERE is_skipped = 0 AND id IN (SELECT text_id FROM skipped_texts)
            """)

            if migration_applied:
                logger.info("Schema migration applied. Incrementing data version.")
                current_v_row = conn.execute("SELECT value FROM settings WHERE key = 'current_data_version'").fetchone()
                current_v = int(current_v_row["value"]) if current_v_row else 0
                new_v = current_v + 1
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                    ("current_data_version", str(new_v))
                )
        except Exception as e:
            logger.critical(f"DATABASE MIGRATION FAILED: {e}")
            raise RuntimeError(f"Database migration failed. Manual intervention may be required: {e}")

        # Performance and Integrity indexes
        logger.debug("Creating performance and integrity indexes")
        
        # Optimization: Only clean up and create unique index if it doesn't exist
        # This prevents scanning millions of rows on every Streamlit reload.
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_texts_text_unique'")
        index_exists = cursor.fetchone() is not None
        
        if not index_exists:
            # Proactive Cleanup: Remove duplicate texts before creating UNIQUE index
            # This handles legacy dumps or "dirty" restores.
            try:
                with conn:
                    # Find duplicates and remove them (keep oldest ID)
                    duplicate_ids_query = """
                        SELECT id FROM texts 
                        WHERE id NOT IN (
                            SELECT MIN(id) FROM texts GROUP BY text
                        )
                    """
                    cursor = conn.execute(duplicate_ids_query)
                    duplicate_ids = [row["id"] for row in cursor.fetchall()]
                    
                    if duplicate_ids:
                        logger.info(f"Found {len(duplicate_ids)} duplicate texts. Cleaning up before enforcing UNIQUE constraint.")
                        # Remove linked records first to maintain consistency
                        placeholders = ", ".join("?" for _ in duplicate_ids)
                        conn.execute(f"DELETE FROM candidates WHERE text_id IN ({placeholders})", duplicate_ids)
                        conn.execute(f"DELETE FROM annotations WHERE text_id IN ({placeholders})", duplicate_ids)
                        conn.execute(f"DELETE FROM skipped_texts WHERE text_id IN ({placeholders})", duplicate_ids)
                        conn.execute(f"DELETE FROM shown_intents WHERE text_id IN ({placeholders})", duplicate_ids)
                        conn.execute(f"DELETE FROM texts WHERE id IN ({placeholders})", duplicate_ids)
                        logger.info("Duplicate cleanup completed.")

                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_texts_text_unique ON texts(text)")
            except sqlite3.IntegrityError as e:
                logger.warning(f"Could not enforce UNIQUE constraint on texts(text): {e}")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_texts_text ON texts(text)")
        
        logger.debug("Creating other performance indexes")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_texts_assigned_to ON texts(assigned_to)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_texts_assigned_cluster ON texts(assigned_cluster)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_texts_language ON texts(language)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_text_id ON annotations(text_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_annotator ON annotations(annotator)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_created ON annotations(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_text_id ON candidates(text_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_text_rank ON candidates(text_id, rank)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_skipped_annotator ON skipped_texts(annotator)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shown_intents_lookup ON shown_intents(text_id, annotator)")

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
    
    logger.debug("Database initialized successfully")
