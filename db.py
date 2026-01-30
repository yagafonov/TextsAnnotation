import os
import sqlite3
import threading
from contextlib import contextmanager

DEFAULT_DB_PATH = os.environ.get("TEXTS_DB_PATH", "data/db/app.db")
DEFAULT_DUMP_PATH = os.environ.get("TEXTS_DB_DUMP_PATH", "data/dumps/backup.sql")
DEFAULT_DUMP_INTERVAL_SEC = int(os.environ.get("TEXTS_DB_DUMP_INTERVAL_SEC", "60"))

_dump_thread = None
_dump_stop = threading.Event()


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


@contextmanager
def connect(db_path: str = DEFAULT_DB_PATH):
    ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def restore_from_dump_if_needed(db_path: str = DEFAULT_DB_PATH, dump_path: str = DEFAULT_DUMP_PATH) -> None:
    if os.path.exists(db_path):
        return
    if not os.path.exists(dump_path):
        return
    ensure_parent_dir(db_path)
    with sqlite3.connect(db_path) as conn, open(dump_path, "r", encoding="utf-8") as handle:
        conn.executescript(handle.read())
        conn.commit()


def dump_database(db_path: str = DEFAULT_DB_PATH, dump_path: str = DEFAULT_DUMP_PATH) -> None:
    ensure_parent_dir(dump_path)
    with sqlite3.connect(db_path) as conn, open(dump_path, "w", encoding="utf-8") as handle:
        for line in conn.iterdump():
            handle.write(f"{line}\n")


def start_auto_dump(
    db_path: str = DEFAULT_DB_PATH,
    dump_path: str = DEFAULT_DUMP_PATH,
    interval_sec: int = DEFAULT_DUMP_INTERVAL_SEC,
) -> None:
    global _dump_thread
    if _dump_thread and _dump_thread.is_alive():
        return
    _dump_stop.clear()

    def _loop() -> None:
        while not _dump_stop.wait(interval_sec):
            dump_database(db_path=db_path, dump_path=dump_path)

    _dump_thread = threading.Thread(target=_loop, daemon=True)
    _dump_thread.start()


def stop_auto_dump() -> None:
    _dump_stop.set()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    ensure_parent_dir(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                language TEXT,
                clusters TEXT,
                assigned_cluster TEXT,
                data_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
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
            """
        )
        conn.execute(
            """
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
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS annotations_unique_entry
            ON annotations (text_id, annotator, label)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intents (
                label TEXT PRIMARY KEY,
                description TEXT,
                examples TEXT,
                complexity TEXT,
                cluster TEXT,
                source_file TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skipped_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id INTEGER NOT NULL,
                annotator TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(text_id) REFERENCES texts(id),
                UNIQUE(text_id, annotator)
            )
            """
        )
        conn.commit()
        ensure_column(conn, "texts", "assigned_cluster", "TEXT")

        if not conn.execute("SELECT 1 FROM model_versions LIMIT 1").fetchone():
            conn.execute(
                "INSERT INTO model_versions (version, note, created_at) VALUES (?, ?, datetime('now'))",
                (0, "initial version",),
            )
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", ("current_model_version", "0"))
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", ("current_data_version", "0"))
            conn.commit()


def ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    existing = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if any(row["name"] == column for row in existing):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
    conn.commit()


def upsert_intent(
    conn: sqlite3.Connection,
    label: str,
    description: str,
    examples: str,
    complexity: str,
    cluster: str,
    source_file: str,
) -> None:
    conn.execute(
        """
        INSERT INTO intents (label, description, examples, complexity, cluster, source_file, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(label) DO UPDATE SET
            description=excluded.description,
            examples=excluded.examples,
            complexity=excluded.complexity,
            cluster=excluded.cluster,
            source_file=excluded.source_file,
            updated_at=datetime('now')
        """,
        (label, description, examples, complexity, cluster, source_file),
    )
    conn.commit()



def get_setting(conn: sqlite3.Connection, key: str, default: str = "0") -> str:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default



def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def create_model_version(conn: sqlite3.Connection, note: str) -> int:
    current = int(get_setting(conn, "current_model_version", "0"))
    new_version = current + 1
    conn.execute(
        "INSERT INTO model_versions (version, note, created_at) VALUES (?, ?, datetime('now'))",
        (new_version, note),
    )
    set_setting(conn, "current_model_version", str(new_version))
    return new_version


def create_data_version(conn: sqlite3.Connection) -> int:
    current = int(get_setting(conn, "current_data_version", "0"))
    new_version = current + 1
    set_setting(conn, "current_data_version", str(new_version))
    return new_version
