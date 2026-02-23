"""Intent repository for database operations."""

import os
from pathlib import Path
from typing import Dict, List, Optional

from src.models.intent import Intent, IntentDatabase
from src.repositories.base import BaseRepository
from src.utils.database import get_connection
from src.utils.logger import logger
from src.utils.yaml_loader import load_yaml_file


class IntentRepository(BaseRepository):
    """Repository for intent operations."""
    
    def load_from_yaml(self, path: str) -> Dict[str, Intent]:
        """Load intents from YAML files.
        
        Args:
            path: Directory containing YAML files or path to single file
            
        Returns:
            Dictionary mapping intent labels to Intent objects
        """
        intents: Dict[str, Intent] = {}
        
        if os.path.isdir(path):
            logger.info(f"Loading intents from directory: {path}")
            for filename in sorted(os.listdir(path)):
                if not filename.lower().endswith((".yaml", ".yml")):
                    continue
                
                file_path = os.path.join(path, filename)
                cluster_name = os.path.splitext(filename)[0]
                file_intents = load_yaml_file(file_path)
                
                for label, payload in file_intents.items():
                    payload = payload or {}
                    payload.setdefault("cluster", cluster_name)
                    
                    intent = Intent(
                        label=label,
                        description=payload.get("description", ""),
                        train=payload.get("train", []),
                        test=payload.get("test", []),
                        complexity=payload.get("complexity", ""),
                        cluster=payload["cluster"],
                        source_file=filename
                    )
                    intents[label] = intent
        else:
            logger.info(f"Loading intents from file: {path}")
            file_intents = load_yaml_file(path)
            filename = os.path.basename(path)
            
            for label, payload in file_intents.items():
                payload = payload or {}
                intent = Intent(
                    label=label,
                    description=payload.get("description", ""),
                    train=payload.get("train", []),
                    test=payload.get("test", []),
                    complexity=payload.get("complexity", ""),
                    cluster=payload.get("cluster", "unknown"),
                    source_file=filename
                )
                intents[label] = intent
        
        logger.info(f"Loaded {len(intents)} intents")
        return intents
    
    def upsert(self, intent: Intent) -> None:
        """Insert or update an intent in the database.
        
        Args:
            intent: Intent to upsert
        """
        with get_connection(self.db_path) as conn:
            train_str = ", ".join(intent.train)
            test_str = ", ".join(intent.test)
            conn.execute(
                """
                INSERT INTO intents (label, description, examples, test_examples, complexity, cluster, source_file, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(label) DO UPDATE SET
                    description=excluded.description,
                    examples=excluded.examples,
                    test_examples=excluded.test_examples,
                    complexity=excluded.complexity,
                    cluster=excluded.cluster,
                    source_file=excluded.source_file,
                    updated_at=datetime('now')
                """,
                (
                    intent.label,
                    intent.description,
                    train_str,
                    test_str,
                    intent.complexity,
                    intent.cluster,
                    intent.source_file or ""
                )
            )
            conn.commit()
    
    def get_by_label(self, label: str) -> Optional[Intent]:
        """Get intent by label.
        
        Args:
            label: Intent label
            
        Returns:
            Intent if found, None otherwise
        """
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM intents WHERE label = ?", (label,)
            ).fetchone()
            
            if not row:
                return None
            
            return Intent(
                label=row["label"],
                description=row["description"],
                train=[ex.strip() for ex in row["examples"].split(",") if ex.strip()],
                test=[ex.strip() for ex in row["test_examples"].split(",") if ex.strip()] if row["test_examples"] else [],
                complexity=row["complexity"],
                cluster=row["cluster"],
                source_file=row["source_file"]
            )
    
    def get_all(self) -> Dict[str, Intent]:
        """Get all intents from database.
        
        Returns:
            Dictionary mapping labels to Intent objects
        """
        with get_connection(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM intents").fetchall()
            
            intents = {}
            for row in rows:
                intent = Intent(
                    label=row["label"],
                    description=row["description"],
                    train=[ex.strip() for ex in row["examples"].split(",") if ex.strip()],
                    test=[ex.strip() for ex in row["test_examples"].split(",") if ex.strip()] if row["test_examples"] else [],
                    complexity=row["complexity"],
                    cluster=row["cluster"],
                    source_file=row["source_file"]
                )
                intents[intent.label] = intent
            
            return intents
    
    def get_by_cluster(self, cluster: str) -> List[Intent]:
        """Get all intents for a specific cluster.
        
        Args:
            cluster: Cluster name
            
        Returns:
            List of intents in cluster
        """
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM intents WHERE cluster = ?", (cluster,)
            ).fetchall()
            
            return [
                Intent(
                    label=row["label"],
                    description=row["description"],
                    train=[ex.strip() for ex in row["examples"].split(",") if ex.strip()],
                    test=[ex.strip() for ex in row["test_examples"].split(",") if ex.strip()] if row["test_examples"] else [],
                    complexity=row["complexity"],
                    cluster=row["cluster"],
                    source_file=row["source_file"]
                )
                for row in rows
            ]
