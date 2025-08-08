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

def initialize_database():
    """
    Initializes the database and creates the necessary tables if they don't exist.
    """
    logger.info(f"Initializing database at: {DB_FILE}")
    conn = get_db_connection()
    if conn is None:
        logger.error("Could not initialize database.")
        return

    try:
        cursor = conn.cursor()

        # --- Entities Table ---
        # Stores unique entities (people, places, concepts)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # --- Memories Table ---
        # Stores episodic and semantic memories associated with entities
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                content TEXT NOT NULL,
                emotional_state TEXT,
                salience REAL,
                source_protocol TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities (id)
            )
        """)

        # --- Relationships Table ---
        # Stores relationships between entities
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
