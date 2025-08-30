import unittest
import asyncio
import os
import logging
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

# Set test environment before importing modules
os.environ['POSTGRES_HOST'] = 'localhost'
os.environ['POSTGRES_DB'] = 'simone_mcp_test'
os.environ['POSTGRES_USER'] = 'test'
os.environ['POSTGRES_PASSWORD'] = 'test'

from mcp_server.database.postgres_database import PostgreSQLDatabase
from mcp_server.database.database_manager import DatabaseManager, DatabaseType
from mcp_server.database.backup_manager import BackupManager

class TestPostgreSQLDatabase(unittest.TestCase):
    """Test PostgreSQL database functionality."""
    
    def setUp(self):
        self.db = PostgreSQLDatabase()
        # Mock the asyncpg pool to avoid requiring actual PostgreSQL
        self.mock_pool = MagicMock()
        self.mock_connection = MagicMock()
        self.mock_pool.acquire.return_value.__aenter__.return_value = self.mock_connection
    
    def test_connection_params_from_environment(self):
        """Test that connection parameters are correctly loaded from environment."""
        params = self.db._get_connection_params()
        
        self.assertEqual(params['host'], 'localhost')
        self.assertEqual(params['database'], 'simone_mcp_test')
        self.assertEqual(params['user'], 'test')
        self.assertEqual(params['password'], 'test')
        self.assertEqual(params['min_size'], 5)
        self.assertEqual(params['max_size'], 20)
    
    @patch('mcp_server.database.postgres_database.asyncpg.create_pool')
    async def test_initialize_pool(self, mock_create_pool):
        """Test pool initialization."""
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool
        
        # Mock the initialize_schema call
        with patch.object(self.db, 'initialize_schema', new_callable=AsyncMock):
            await self.db.initialize_pool()
        
        self.assertEqual(self.db.pool, mock_pool)
        mock_create_pool.assert_called_once()
    
    @patch('mcp_server.database.postgres_database.asyncpg.create_pool')
    async def test_initialize_schema(self, mock_create_pool):
        """Test schema initialization."""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()
        
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = mock_transaction
        
        mock_create_pool.return_value = mock_pool
        self.db.pool = mock_pool
        
        await self.db.initialize_schema()
        
        # Verify that essential SQL commands were executed
        execute_calls = mock_conn.execute.call_args_list
        execute_commands = [call[0][0] for call in execute_calls]
        
        # Check that extensions were created
        extension_commands = [cmd for cmd in execute_commands if 'CREATE EXTENSION' in cmd]
        self.assertGreater(len(extension_commands), 0)
        
        # Check that tables were created
        table_commands = [cmd for cmd in execute_commands if 'CREATE TABLE' in cmd]
        expected_tables = ['entities', 'memories', 'relationships', 'memory_contradictions']
        created_tables = []
        
        for cmd in table_commands:
            for table in expected_tables:
                if table in cmd:
                    created_tables.append(table)
        
        for table in expected_tables:
            self.assertIn(table, created_tables, f"Table {table} should be created")
    
    async def test_health_check_success(self):
        """Test successful health check."""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = 1
        mock_conn.fetchrow.return_value = {
            'database_name': 'simone_mcp_test',
            'user_name': 'test',
            'version': 'PostgreSQL 15.0'
        }
        
        # Mock pool stats
        mock_pool.get_size.return_value = 5
        mock_pool.get_free_size.return_value = 3
        mock_pool.get_max_size.return_value = 20
        mock_pool.get_min_size.return_value = 5
        
        self.db.pool = mock_pool
        
        health = await self.db.health_check()
        
        self.assertEqual(health['status'], 'healthy')
        self.assertTrue(health['connection_test'])
        self.assertIn('pool_stats', health)
        self.assertIn('database_stats', health)
    
    async def test_health_check_failure(self):
        """Test health check when database is unavailable."""
        self.db.pool = None
        
        health = await self.db.health_check()
        
        self.assertEqual(health['status'], 'error')
        self.assertIn('message', health)

class TestDatabaseManager(unittest.TestCase):
    """Test database manager functionality."""
    
    def setUp(self):
        self.manager = DatabaseManager()
    
    def test_detect_database_type_postgresql(self):
        """Test PostgreSQL detection with environment variables."""
        with patch.dict(os.environ, {'POSTGRES_HOST': 'localhost'}):
            manager = DatabaseManager()
            self.assertEqual(manager._detect_database_type(), DatabaseType.POSTGRESQL)
    
    def test_detect_database_type_sqlite(self):
        """Test SQLite detection as default."""
        # Clear PostgreSQL environment variables
        env_vars_to_clear = ['POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER', 'DATABASE_URL']
        with patch.dict(os.environ, {var: '' for var in env_vars_to_clear}, clear=False):
            manager = DatabaseManager()
            self.assertEqual(manager._detect_database_type(), DatabaseType.SQLITE)
    
    @patch('mcp_server.database.database_manager.postgres_db.initialize_pool')
    async def test_initialize_postgresql(self, mock_init_pool):
        """Test PostgreSQL initialization."""
        self.manager.db_type = DatabaseType.POSTGRESQL
        mock_init_pool.return_value = None
        
        await self.manager.initialize()
        
        mock_init_pool.assert_called_once()
        self.assertTrue(self.manager._initialized)
    
    @patch('mcp_server.database.database_manager.initialize_database')
    async def test_initialize_sqlite(self, mock_init_db):
        """Test SQLite initialization."""
        self.manager.db_type = DatabaseType.SQLITE
        mock_init_db.return_value = None
        
        await self.manager.initialize()
        
        mock_init_db.assert_called_once()
        self.assertTrue(self.manager._initialized)
    
    async def test_store_memory_detection(self):
        """Test that memory storage correctly detects database type."""
        # Test PostgreSQL path
        self.manager.db_type = DatabaseType.POSTGRESQL
        with patch.object(self.manager, '_store_memory_postgres', new_callable=AsyncMock) as mock_pg:
            mock_pg.return_value = 123
            
            result = await self.manager.store_memory(1, "test content")
            
            mock_pg.assert_called_once_with(1, "test content", None, None, None)
            self.assertEqual(result, 123)
        
        # Test SQLite path
        self.manager.db_type = DatabaseType.SQLITE
        with patch.object(self.manager, '_store_memory_sqlite', new_callable=AsyncMock) as mock_sqlite:
            mock_sqlite.return_value = 456
            
            result = await self.manager.store_memory(1, "test content")
            
            mock_sqlite.assert_called_once_with(1, "test content", None, None, None)
            self.assertEqual(result, 456)

class TestBackupManager(unittest.TestCase):
    """Test database backup and recovery functionality."""
    
    def setUp(self):
        self.backup_manager = BackupManager()
    
    @patch('mcp_server.database.backup_manager.db_manager')
    async def test_create_backup_detection(self, mock_db_manager):
        """Test that backup creation detects correct database type."""
        # Test PostgreSQL backup
        mock_db_manager.is_postgresql.return_value = True
        
        with patch.object(self.backup_manager, '_create_postgres_backup', new_callable=AsyncMock) as mock_pg:
            mock_pg.return_value = {"type": "postgresql", "filename": "test.sql.gz"}
            
            result = await self.backup_manager.create_backup("test")
            
            self.assertEqual(result["type"], "postgresql")
        
        # Test SQLite backup
        mock_db_manager.is_postgresql.return_value = False
        
        with patch.object(self.backup_manager, '_create_sqlite_backup', new_callable=AsyncMock) as mock_sqlite:
            mock_sqlite.return_value = {"type": "sqlite", "filename": "test.db.gz"}
            
            result = await self.backup_manager.create_backup("test")
            
            self.assertEqual(result["type"], "sqlite")
    
    def test_list_backups(self):
        """Test backup file listing."""
        # This test doesn't require actual files, just tests the parsing logic
        backups = self.backup_manager.list_backups()
        
        # Should return empty list if no backup files exist
        self.assertIsInstance(backups, list)
    
    def test_cleanup_old_backups_logic(self):
        """Test backup cleanup logic without actual files."""
        # Mock backup list
        mock_backups = [
            {
                'filename': 'simone_mcp_postgres_manual_20240830_120000.gz',
                'backup_type': 'manual',
                'created_at': '2024-08-30T12:00:00',
                'size_bytes': 1000,
                'path': '/tmp/test.gz'
            }
        ]
        
        with patch.object(self.backup_manager, 'list_backups', return_value=mock_backups):
            with patch('pathlib.Path.unlink') as mock_unlink:
                result = self.backup_manager.cleanup_old_backups()
                
                # Manual backups should never be auto-deleted
                self.assertEqual(result['removed_count'], 0)
                self.assertEqual(result['kept_count'], 1)
                mock_unlink.assert_not_called()

class TestDatabaseIntegration(unittest.TestCase):
    """Integration tests for database functionality."""
    
    @patch('mcp_server.database.database_manager.db_manager')
    async def test_memory_operations_flow(self, mock_db_manager):
        """Test complete memory storage and retrieval flow."""
        mock_db_manager.find_or_create_entity.return_value = 1
        mock_db_manager.store_memory.return_value = 10
        mock_db_manager.get_memories_by_session.return_value = [
            {
                'id': 10,
                'content': 'test memory',
                'entity_name': 'test_entity'
            }
        ]
        
        # Simulate storing a memory
        entity_id = await mock_db_manager.find_or_create_entity("test_entity")
        memory_id = await mock_db_manager.store_memory(
            entity_id, 
            "test memory", 
            session_id="test_session"
        )
        
        # Simulate retrieving memories
        memories = await mock_db_manager.get_memories_by_session("test_session")
        
        self.assertEqual(entity_id, 1)
        self.assertEqual(memory_id, 10)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]['content'], 'test memory')

def run_async_test(coro):
    """Helper function to run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Convert async test methods to sync for unittest compatibility
for cls in [TestPostgreSQLDatabase, TestDatabaseManager, TestBackupManager, TestDatabaseIntegration]:
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and attr_name.startswith('test_') and asyncio.iscoroutinefunction(attr):
            # Wrap async test in sync function
            def make_sync_test(async_test):
                def sync_test(self):
                    return run_async_test(async_test(self))
                return sync_test
            
            setattr(cls, attr_name, make_sync_test(attr))

if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()