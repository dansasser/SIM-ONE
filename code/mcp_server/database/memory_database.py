import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DB_FILE = Path(__file__).parent / "persistent_memory.db"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection failed: {e}")
        return None

def _upgrade_schema(cursor):
    """Upgrades the database schema by adding new columns if they don't exist."""
    logger.info("Checking database schema...")

    cursor.execute("PRAGMA table_info(memories)")
    columns = [row['name'] for row in cursor.fetchall()]

    # Handle the salience -> emotional_salience rename first
    if 'salience' in columns and 'emotional_salience' not in columns:
        try:
            cursor.execute("ALTER TABLE memories RENAME COLUMN salience TO emotional_salience")
            logger.info("Renamed column 'salience' to 'emotional_salience'.")
            # Refresh column list after rename
            cursor.execute("PRAGMA table_info(memories)")
            columns = [row['name'] for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            logger.error(f"Could not rename 'salience' column: {e}")

    # Now, add the new columns if they are missing
    new_columns = {
        "session_id": "TEXT",
        "emotional_salience": "REAL DEFAULT 0.5",
        "rehearsal_count": "INTEGER DEFAULT 0",
        "last_accessed": "TIMESTAMP",
        "confidence_score": "REAL DEFAULT 1.0",
        "memory_type": "TEXT DEFAULT 'episodic'",
        "actors": "TEXT",
        "context_tags": "TEXT"
    }

    for col_name, col_type in new_columns.items():
        if col_name not in columns:
            try:
                cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
                logger.info(f"Added column '{col_name}' to 'memories' table.")
            except sqlite3.OperationalError as e:
                logger.error(f"Could not add column {col_name}: {e}")


def initialize_database():
    """
    Initializes the database and creates/upgrades the necessary tables.
    """
    logger.info(f"Initializing database at: {DB_FILE}")
    conn = get_db_connection()
    if conn is None:
        logger.error("Could not initialize database.")
        return

    try:
        cursor = conn.cursor()

        # --- Entities Table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # --- Memories Table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                content TEXT NOT NULL,
                emotional_state TEXT,
                source_protocol TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities (id)
            )
        """)

        # --- Relationships Table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id INTEGER,
                target_entity_id INTEGER,
                type TEXT NOT NULL,
                strength REAL,
                FOREIGN KEY (source_entity_id) REFERENCES entities (id),
                FOREIGN KEY (target_entity_id) REFERENCES entities (id)
            )
        """)

        # Upgrade schema for the memories table
        _upgrade_schema(cursor)

        conn.commit()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database tables: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # This script can be run directly to create/update the database file.
    logging.basicConfig(level=logging.INFO)
    print("Running database initialization...")
    initialize_database()
    print("Database initialization script finished.")

    # Verify file creation
    if DB_FILE.exists():
        print(f"Database file created at: {DB_FILE}")
    else:
        print("Database file was not created.")
