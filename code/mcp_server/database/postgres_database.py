import asyncio
import asyncpg
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import os
from pathlib import Path

from mcp_server.config import settings

logger = logging.getLogger(__name__)

class PostgreSQLDatabase:
    """
    Production PostgreSQL database connection and operations.
    Provides async connection pooling and transaction management.
    """
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.connection_params = self._get_connection_params()
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Get PostgreSQL connection parameters from environment or config."""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'simone_mcp'),
            'user': os.getenv('POSTGRES_USER', 'simone'),
            'password': os.getenv('POSTGRES_PASSWORD', ''),
            'min_size': int(os.getenv('POSTGRES_MIN_CONNECTIONS', 5)),
            'max_size': int(os.getenv('POSTGRES_MAX_CONNECTIONS', 20)),
            'command_timeout': int(os.getenv('POSTGRES_COMMAND_TIMEOUT', 60)),
            'server_settings': {
                'application_name': f'simone_mcp_{settings.APP_VERSION}',
                'jit': 'off'  # Disable JIT for smaller queries
            }
        }
    
    async def initialize_pool(self):
        """Initialize the connection pool."""
        try:
            logger.info(f"Initializing PostgreSQL connection pool to {self.connection_params['host']}:{self.connection_params['port']}")
            
            self.pool = await asyncpg.create_pool(
                host=self.connection_params['host'],
                port=self.connection_params['port'],
                database=self.connection_params['database'],
                user=self.connection_params['user'],
                password=self.connection_params['password'],
                min_size=self.connection_params['min_size'],
                max_size=self.connection_params['max_size'],
                command_timeout=self.connection_params['command_timeout'],
                server_settings=self.connection_params['server_settings']
            )
            
            logger.info("PostgreSQL connection pool initialized successfully")
            
            # Test connection and initialize schema
            await self.initialize_schema()
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    async def close_pool(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    async def initialize_schema(self):
        """Initialize database schema with tables and indexes."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Enable necessary extensions
                await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
                await conn.execute("CREATE EXTENSION IF NOT EXISTS \"pg_trgm\"")
                
                # Create entities table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        id SERIAL PRIMARY KEY,
                        uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
                        name TEXT UNIQUE NOT NULL,
                        type TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create memories table with enhanced schema
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id SERIAL PRIMARY KEY,
                        uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
                        entity_id INTEGER REFERENCES entities(id),
                        content TEXT NOT NULL,
                        emotional_state TEXT,
                        source_protocol TEXT,
                        session_id TEXT,
                        emotional_salience REAL DEFAULT 0.5,
                        rehearsal_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        confidence_score REAL DEFAULT 1.0,
                        memory_type TEXT DEFAULT 'episodic',
                        actors TEXT[],
                        context_tags TEXT[],
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create relationships table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS relationships (
                        id SERIAL PRIMARY KEY,
                        uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
                        source_entity_id INTEGER REFERENCES entities(id),
                        target_entity_id INTEGER REFERENCES entities(id),
                        type TEXT NOT NULL,
                        strength REAL DEFAULT 1.0,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create memory contradictions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_contradictions (
                        id SERIAL PRIMARY KEY,
                        uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
                        memory_id_1 INTEGER REFERENCES memories(id),
                        memory_id_2 INTEGER REFERENCES memories(id),
                        reason TEXT,
                        confidence REAL DEFAULT 1.0,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolution_notes TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create indexes for performance
                await self._create_indexes(conn)
                
                # Create updated_at triggers
                await self._create_update_triggers(conn)
                
                logger.info("PostgreSQL schema initialized successfully")
    
    async def _create_indexes(self, conn):
        """Create performance indexes."""
        indexes = [
            # Entities indexes
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities USING gin(name gin_trgm_ops)",
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)",
            "CREATE INDEX IF NOT EXISTS idx_entities_created_at ON entities(created_at)",
            
            # Memories indexes
            "CREATE INDEX IF NOT EXISTS idx_memories_entity_id ON memories(entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_content ON memories USING gin(content gin_trgm_ops)",
            "CREATE INDEX IF NOT EXISTS idx_memories_emotional_state ON memories(emotional_state)",
            "CREATE INDEX IF NOT EXISTS idx_memories_source_protocol ON memories(source_protocol)",
            "CREATE INDEX IF NOT EXISTS idx_memories_emotional_salience ON memories(emotional_salience DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_actors ON memories USING gin(actors)",
            "CREATE INDEX IF NOT EXISTS idx_memories_context_tags ON memories USING gin(context_tags)",
            "CREATE INDEX IF NOT EXISTS idx_memories_metadata ON memories USING gin(metadata)",
            "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC)",
            
            # Relationships indexes
            "CREATE INDEX IF NOT EXISTS idx_relationships_source_entity ON relationships(source_entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_target_entity ON relationships(target_entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(type)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_strength ON relationships(strength DESC)",
            
            # Contradictions indexes
            "CREATE INDEX IF NOT EXISTS idx_contradictions_memory_1 ON memory_contradictions(memory_id_1)",
            "CREATE INDEX IF NOT EXISTS idx_contradictions_memory_2 ON memory_contradictions(memory_id_2)",
            "CREATE INDEX IF NOT EXISTS idx_contradictions_resolved ON memory_contradictions(resolved)",
        ]
        
        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Could not create index: {e}")
    
    async def _create_update_triggers(self, conn):
        """Create triggers to automatically update updated_at timestamps."""
        # Create the trigger function
        await conn.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql'
        """)
        
        # Create triggers for each table
        tables = ['entities', 'memories', 'relationships', 'memory_contradictions']
        for table in tables:
            await conn.execute(f"""
                DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table};
                CREATE TRIGGER update_{table}_updated_at
                    BEFORE UPDATE ON {table}
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column()
            """)
    
    async def get_connection(self):
        """Get a connection from the pool."""
        if not self.pool:
            await self.initialize_pool()
        return self.pool.acquire()
    
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dicts."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute an INSERT/UPDATE/DELETE command and return status."""
        async with self.pool.acquire() as conn:
            return await conn.execute(command, *args)
    
    async def execute_transaction(self, commands: List[tuple]) -> bool:
        """Execute multiple commands in a transaction."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                try:
                    for command, args in commands:
                        await conn.execute(command, *args)
                    return True
                except Exception as e:
                    logger.error(f"Transaction failed: {e}")
                    return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database connection."""
        if not self.pool:
            return {"status": "error", "message": "Pool not initialized"}
        
        try:
            async with self.pool.acquire() as conn:
                # Test basic connectivity
                result = await conn.fetchval("SELECT 1")
                
                # Get pool stats
                pool_stats = {
                    "size": self.pool.get_size(),
                    "free_connections": self.pool.get_free_size(),
                    "max_size": self.pool.get_max_size(),
                    "min_size": self.pool.get_min_size()
                }
                
                # Get database stats
                db_stats = await conn.fetchrow("""
                    SELECT 
                        current_database() as database_name,
                        current_user as user_name,
                        version() as version
                """)
                
                return {
                    "status": "healthy",
                    "connection_test": result == 1,
                    "pool_stats": pool_stats,
                    "database_stats": dict(db_stats),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error", 
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global database instance
postgres_db = PostgreSQLDatabase()

async def get_postgres_connection():
    """Get a PostgreSQL database connection."""
    return await postgres_db.get_connection()

async def initialize_postgres():
    """Initialize the PostgreSQL database."""
    await postgres_db.initialize_pool()

async def close_postgres():
    """Close the PostgreSQL database connection pool."""
    await postgres_db.close_pool()