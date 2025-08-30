import asyncio
import logging
import os
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from datetime import datetime

from mcp_server.database.memory_database import get_db_connection, initialize_database
from mcp_server.database.postgres_database import postgres_db

logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"

class DatabaseManager:
    """
    Unified database manager that supports both SQLite and PostgreSQL.
    Automatically detects and uses the appropriate database based on configuration.
    """
    
    def __init__(self):
        self.db_type = self._detect_database_type()
        self._initialized = False
    
    def _detect_database_type(self) -> DatabaseType:
        """Detect which database type to use based on environment configuration."""
        # Check for PostgreSQL environment variables
        postgres_indicators = [
            'POSTGRES_HOST',
            'POSTGRES_DB', 
            'POSTGRES_USER',
            'DATABASE_URL'  # Common for cloud deployments
        ]
        
        # If any PostgreSQL indicators are present, use PostgreSQL
        if any(os.getenv(var) for var in postgres_indicators):
            logger.info("PostgreSQL configuration detected, using PostgreSQL backend")
            return DatabaseType.POSTGRESQL
        
        # Default to SQLite for development and simple deployments
        logger.info("No PostgreSQL configuration found, using SQLite backend")
        return DatabaseType.SQLITE
    
    async def initialize(self):
        """Initialize the appropriate database backend."""
        if self._initialized:
            return
        
        try:
            if self.db_type == DatabaseType.POSTGRESQL:
                await postgres_db.initialize_pool()
                logger.info("PostgreSQL database initialized")
            else:
                # SQLite initialization (synchronous)
                initialize_database()
                logger.info("SQLite database initialized")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        if self.db_type == DatabaseType.POSTGRESQL:
            await postgres_db.close_pool()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            if self.db_type == DatabaseType.POSTGRESQL:
                return await postgres_db.health_check()
            else:
                # SQLite health check
                conn = get_db_connection()
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()
                        conn.close()
                        
                        return {
                            "status": "healthy",
                            "database_type": "sqlite",
                            "connection_test": result[0] == 1,
                            "timestamp": datetime.now().isoformat()
                        }
                    except Exception as e:
                        conn.close()
                        raise e
                else:
                    return {
                        "status": "error",
                        "database_type": "sqlite",
                        "message": "Could not establish connection",
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Memory Management Methods
    
    async def store_memory(self, entity_id: int, content: str, emotional_state: Optional[str] = None, 
                          source_protocol: Optional[str] = None, session_id: Optional[str] = None,
                          **kwargs) -> int:
        """Store a new memory in the database."""
        if self.db_type == DatabaseType.POSTGRESQL:
            return await self._store_memory_postgres(entity_id, content, emotional_state, 
                                                   source_protocol, session_id, **kwargs)
        else:
            return await self._store_memory_sqlite(entity_id, content, emotional_state, 
                                                 source_protocol, session_id, **kwargs)
    
    async def _store_memory_postgres(self, entity_id: int, content: str, emotional_state: Optional[str],
                                   source_protocol: Optional[str], session_id: Optional[str],
                                   **kwargs) -> int:
        """Store memory in PostgreSQL."""
        query = """
            INSERT INTO memories (
                entity_id, content, emotional_state, source_protocol, session_id,
                emotional_salience, rehearsal_count, confidence_score, memory_type,
                actors, context_tags, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING id
        """
        
        async with postgres_db.pool.acquire() as conn:
            memory_id = await conn.fetchval(
                query,
                entity_id,
                content,
                emotional_state,
                source_protocol,
                session_id,
                kwargs.get('emotional_salience', 0.5),
                kwargs.get('rehearsal_count', 0),
                kwargs.get('confidence_score', 1.0),
                kwargs.get('memory_type', 'episodic'),
                kwargs.get('actors', []),
                kwargs.get('context_tags', []),
                kwargs.get('metadata', {})
            )
            
        logger.debug(f"Stored memory with ID {memory_id} in PostgreSQL")
        return memory_id
    
    async def _store_memory_sqlite(self, entity_id: int, content: str, emotional_state: Optional[str],
                                 source_protocol: Optional[str], session_id: Optional[str],
                                 **kwargs) -> int:
        """Store memory in SQLite."""
        conn = get_db_connection()
        if not conn:
            raise RuntimeError("Could not connect to SQLite database")
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (
                    entity_id, content, emotional_state, source_protocol, session_id,
                    emotional_salience, rehearsal_count, confidence_score, memory_type,
                    actors, context_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity_id,
                content,
                emotional_state,
                source_protocol,
                session_id,
                kwargs.get('emotional_salience', 0.5),
                kwargs.get('rehearsal_count', 0),
                kwargs.get('confidence_score', 1.0),
                kwargs.get('memory_type', 'episodic'),
                str(kwargs.get('actors', [])) if kwargs.get('actors') else None,
                str(kwargs.get('context_tags', [])) if kwargs.get('context_tags') else None
            ))
            
            memory_id = cursor.lastrowid
            conn.commit()
            logger.debug(f"Stored memory with ID {memory_id} in SQLite")
            return memory_id
            
        finally:
            conn.close()
    
    async def get_memories_by_session(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve memories for a specific session."""
        if self.db_type == DatabaseType.POSTGRESQL:
            return await self._get_memories_by_session_postgres(session_id, limit)
        else:
            return await self._get_memories_by_session_sqlite(session_id, limit)
    
    async def _get_memories_by_session_postgres(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get session memories from PostgreSQL."""
        query = """
            SELECT m.*, e.name as entity_name
            FROM memories m
            LEFT JOIN entities e ON m.entity_id = e.id
            WHERE m.session_id = $1
            ORDER BY m.emotional_salience DESC, m.created_at DESC
            LIMIT $2
        """
        
        return await postgres_db.execute_query(query, session_id, limit)
    
    async def _get_memories_by_session_sqlite(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get session memories from SQLite."""
        conn = get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.*, e.name as entity_name
                FROM memories m
                LEFT JOIN entities e ON m.entity_id = e.id
                WHERE m.session_id = ?
                ORDER BY m.emotional_salience DESC, m.timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
            
        finally:
            conn.close()
    
    async def find_or_create_entity(self, name: str, entity_type: Optional[str] = None) -> int:
        """Find an existing entity or create a new one."""
        if self.db_type == DatabaseType.POSTGRESQL:
            return await self._find_or_create_entity_postgres(name, entity_type)
        else:
            return await self._find_or_create_entity_sqlite(name, entity_type)
    
    async def _find_or_create_entity_postgres(self, name: str, entity_type: Optional[str]) -> int:
        """Find or create entity in PostgreSQL."""
        async with postgres_db.pool.acquire() as conn:
            # Try to find existing entity
            entity_id = await conn.fetchval("SELECT id FROM entities WHERE name = $1", name)
            
            if entity_id:
                return entity_id
            
            # Create new entity
            entity_id = await conn.fetchval(
                "INSERT INTO entities (name, type) VALUES ($1, $2) RETURNING id",
                name, entity_type
            )
            
            logger.debug(f"Created new entity '{name}' with ID {entity_id}")
            return entity_id
    
    async def _find_or_create_entity_sqlite(self, name: str, entity_type: Optional[str]) -> int:
        """Find or create entity in SQLite."""
        conn = get_db_connection()
        if not conn:
            raise RuntimeError("Could not connect to database")
        
        try:
            cursor = conn.cursor()
            
            # Try to find existing entity
            cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
            result = cursor.fetchone()
            
            if result:
                return result[0]
            
            # Create new entity
            cursor.execute("INSERT INTO entities (name, type) VALUES (?, ?)", (name, entity_type))
            entity_id = cursor.lastrowid
            conn.commit()
            
            logger.debug(f"Created new entity '{name}' with ID {entity_id}")
            return entity_id
            
        finally:
            conn.close()
    
    async def update_memory_access(self, memory_id: int):
        """Update the last_accessed timestamp for a memory."""
        if self.db_type == DatabaseType.POSTGRESQL:
            await postgres_db.execute_command(
                "UPDATE memories SET last_accessed = NOW(), rehearsal_count = rehearsal_count + 1 WHERE id = $1",
                memory_id
            )
        else:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE memories SET last_accessed = CURRENT_TIMESTAMP, rehearsal_count = rehearsal_count + 1 WHERE id = ?",
                        (memory_id,)
                    )
                    conn.commit()
                finally:
                    conn.close()
    
    def get_database_type(self) -> DatabaseType:
        """Get the current database type."""
        return self.db_type
    
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL."""
        return self.db_type == DatabaseType.POSTGRESQL
    
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self.db_type == DatabaseType.SQLITE

# Global database manager instance
db_manager = DatabaseManager()

async def initialize_database_manager():
    """Initialize the global database manager."""
    await db_manager.initialize()

async def get_database_manager() -> DatabaseManager:
    """Get the initialized database manager."""
    if not db_manager._initialized:
        await db_manager.initialize()
    return db_manager