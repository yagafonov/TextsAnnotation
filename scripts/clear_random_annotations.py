
import sqlite3
import random
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import settings
from src.utils.database import get_connection

def clear_random_annotations(percent=10):
    db_path = settings.db_path
    print(f"Database path: {db_path}")
    
    with get_connection(db_path) as conn:
        # Get all annotated text IDs
        cursor = conn.execute("SELECT DISTINCT text_id FROM annotations")
        annotated_text_ids = [row[0] for row in cursor.fetchall()]
        
        total = len(annotated_text_ids)
        if total == 0:
            print("No annotations found.")
            return

        to_remove_count = int(total * (percent / 100))
        if to_remove_count == 0 and total > 0:
            to_remove_count = 1  # Ensure at least one is removed if requested
            
        print(f"Total annotated texts: {total}")
        print(f"Target modification: {percent}% ({to_remove_count} texts)")
        
        # Select random IDs
        ids_to_remove = random.sample(annotated_text_ids, to_remove_count)
        
        if not ids_to_remove:
            print("No IDs selected for removal.")
            return

        # Prepare SQL for safe formatting (sqlite doesn't support array parameters easily)
        # We'll do it in chunks or loop if list is huge, but for 10% of likely small dataset, string formatting is okay-ish 
        # or verify standard binding "IN (?, ?, ...)"
        
        placeholders = ','.join('?' for _ in ids_to_remove)
        
        # 1. Remove from annotations
        conn.execute(f"DELETE FROM annotations WHERE text_id IN ({placeholders})", ids_to_remove)
        
        # 2. Remove from shown_intents (cleanup)
        conn.execute(f"DELETE FROM shown_intents WHERE text_id IN ({placeholders})", ids_to_remove)
        
        conn.commit()
        
        print(f"Successfully cleared annotations for {len(ids_to_remove)} texts.")

if __name__ == "__main__":
    # Confirm
    print("WARNING: This will permanently delete random annotations.")
    clear_random_annotations(10)
