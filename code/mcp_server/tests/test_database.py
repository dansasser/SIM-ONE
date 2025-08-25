import unittest
import sqlite3
import os
from pathlib import Path

from mcp_server.database.memory_database import initialize_database, get_db_connection, DB_FILE

class TestDatabaseMigration(unittest.TestCase):

    def setUp(self):
        # Ensure the database file doesn't exist before a test
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)

    def tearDown(self):
        # Clean up the database file after a test
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)

    def test_schema_migration_from_old_version(self):
        """
        Tests that the initialize_database function correctly migrates an old schema.
        """
        # 1. Create an old-style database manually
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY,
                content TEXT,
                salience REAL
            )
        """)
        cursor.execute("INSERT INTO memories (content, salience) VALUES (?, ?)", ("old memory", 0.7))
        conn.commit()
        conn.close()

        # 2. Run the initialization process, which should trigger the migration
        initialize_database()

        # 3. Check that the new columns exist and salience was renamed
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(memories)")
        columns = [row['name'] for row in cursor.fetchall()]

        self.assertIn("emotional_salience", columns)
        self.assertIn("rehearsal_count", columns)
        self.assertIn("confidence_score", columns)
        self.assertNotIn("salience", columns)

        # 4. Check that data was preserved
        cursor.execute("SELECT emotional_salience FROM memories WHERE content = ?", ("old memory",))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row['emotional_salience'], 0.7)

        conn.close()

if __name__ == '__main__':
    unittest.main()
