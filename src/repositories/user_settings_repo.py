"""Repository for per-user UI settings (theme, layout, label filter)."""

from typing import Optional

from src.repositories.base import BaseRepository
from src.utils.database import get_connection


class UserSettingsRepository(BaseRepository):
    """Saves and loads per-annotator UI preferences."""

    def load(self, annotator: str) -> dict:
        """Return settings dict for annotator, or empty dict if not found."""
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT dark_mode, layout_mode, selected_label "
                "FROM user_settings WHERE annotator = ?",
                (annotator,),
            ).fetchone()
        if row is None:
            return {}
        return {k: row[k] for k in ("dark_mode", "layout_mode", "selected_label") if row[k] is not None}

    def save(self, annotator: str, **kwargs) -> None:
        """Upsert one or more settings keys for annotator.

        Accepted kwargs: dark_mode, layout_mode, selected_label.
        Only keys present in kwargs are updated; others are left unchanged.
        """
        allowed = {"dark_mode", "layout_mode", "selected_label"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        with get_connection(self.db_path) as conn:
            # Ensure row exists
            conn.execute(
                "INSERT OR IGNORE INTO user_settings (annotator, updated_at) VALUES (?, datetime('now'))",
                (annotator,),
            )
            for col, val in fields.items():
                conn.execute(
                    f"UPDATE user_settings SET {col} = ?, updated_at = datetime('now') WHERE annotator = ?",
                    (val, annotator),
                )
            conn.commit()
