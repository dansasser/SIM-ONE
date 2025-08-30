import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

from mcp_server.database.database_manager import db_manager

logger = logging.getLogger(__name__)

class MigrationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"

@dataclass
class Migration:
    """Represents a database migration."""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    checksum: str
    dependencies: List[str]
    created_at: datetime
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate migration checksum for integrity verification."""
        content = f"{self.version}:{self.name}:{self.up_sql}:{self.down_sql}"
        return hashlib.sha256(content.encode()).hexdigest()

@dataclass
class MigrationRecord:
    """Represents a migration execution record."""
    version: str
    name: str
    status: MigrationStatus
    checksum: str
    executed_at: datetime
    execution_time_ms: int
    error_message: Optional[str] = None

class DatabaseSchemaManager:
    """
    Comprehensive database schema management and migration system.
    Handles version control, migration execution, rollbacks, and schema validation.
    """
    
    def __init__(self, migrations_dir: Optional[str] = None):
        self.migrations_dir = Path(migrations_dir or "./database/migrations")
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        
        self.migrations: Dict[str, Migration] = {}
        self.migration_history: List[MigrationRecord] = []
        self.current_version = "0.0.0"
        
        # Schema validation settings
        self.validate_checksums = True
        self.auto_create_migration_table = True
        
        # Migration execution settings
        self.max_execution_time = 300  # 5 minutes
        self.backup_before_migration = True
    
    async def initialize(self):
        """Initialize schema manager and create migration tracking table."""
        try:
            await self._create_migration_table()
            await self.load_migrations()
            await self._load_migration_history()
            await self._determine_current_version()
            
            logger.info(f"Schema manager initialized. Current version: {self.current_version}")
            
        except Exception as e:
            logger.error(f"Failed to initialize schema manager: {e}")
            raise
    
    async def _create_migration_table(self):
        """Create the migration tracking table."""
        if db_manager.is_postgresql():
            await self._create_postgresql_migration_table()
        else:
            await self._create_sqlite_migration_table()
    
    async def _create_postgresql_migration_table(self):
        """Create PostgreSQL migration tracking table."""
        from mcp_server.database.postgres_database import postgres_db
        
        async with postgres_db.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        id SERIAL PRIMARY KEY,
                        version VARCHAR(50) NOT NULL UNIQUE,
                        name VARCHAR(255) NOT NULL,
                        status VARCHAR(20) NOT NULL DEFAULT 'pending',
                        checksum VARCHAR(64) NOT NULL,
                        executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        execution_time_ms INTEGER DEFAULT 0,
                        error_message TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create index for version lookups
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_schema_migrations_version 
                    ON schema_migrations(version)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_schema_migrations_status 
                    ON schema_migrations(status)
                """)
    
    async def _create_sqlite_migration_table(self):
        """Create SQLite migration tracking table."""
        conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
        if not conn:
            from mcp_server.database.memory_database import get_db_connection
            conn = get_db_connection()
        
        if not conn:
            raise RuntimeError("Could not establish SQLite connection")
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    checksum TEXT NOT NULL,
                    executed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    execution_time_ms INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_schema_migrations_version 
                ON schema_migrations(version)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_schema_migrations_status 
                ON schema_migrations(status)
            """)
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def load_migrations(self):
        """Load migration files from the migrations directory."""
        self.migrations.clear()
        
        migration_files = sorted(self.migrations_dir.glob("*.sql"))
        
        for migration_file in migration_files:
            try:
                migration = await self._parse_migration_file(migration_file)
                self.migrations[migration.version] = migration
                
            except Exception as e:
                logger.error(f"Failed to parse migration file {migration_file}: {e}")
        
        logger.info(f"Loaded {len(self.migrations)} migration(s)")
    
    async def _parse_migration_file(self, file_path: Path) -> Migration:
        """Parse a migration SQL file."""
        content = file_path.read_text()
        
        # Extract metadata from comments at the top of the file
        lines = content.split('\n')
        metadata = {}
        up_sql_lines = []
        down_sql_lines = []
        
        current_section = 'metadata'
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('-- @'):
                # Parse metadata
                key_value = line[4:].split(':', 1)
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    metadata[key] = value
            
            elif line.upper() == '-- UP':
                current_section = 'up'
                continue
            
            elif line.upper() == '-- DOWN':
                current_section = 'down'
                continue
            
            elif current_section == 'up' and line and not line.startswith('--'):
                up_sql_lines.append(line)
            
            elif current_section == 'down' and line and not line.startswith('--'):
                down_sql_lines.append(line)
        
        # Extract version from filename (e.g., "001_initial_schema.sql" -> "001")
        filename = file_path.stem
        version = metadata.get('version', filename.split('_')[0])
        name = metadata.get('name', filename)
        description = metadata.get('description', '')
        dependencies = metadata.get('dependencies', '').split(',') if metadata.get('dependencies') else []
        
        up_sql = '\n'.join(up_sql_lines)
        down_sql = '\n'.join(down_sql_lines)
        
        if not up_sql:
            raise ValueError(f"Migration {filename} missing UP section")
        
        return Migration(
            version=version,
            name=name,
            description=description,
            up_sql=up_sql,
            down_sql=down_sql,
            checksum='',  # Will be calculated in __post_init__
            dependencies=dependencies,
            created_at=datetime.fromtimestamp(file_path.stat().st_mtime)
        )
    
    async def _load_migration_history(self):
        """Load migration execution history from database."""
        self.migration_history.clear()
        
        try:
            if db_manager.is_postgresql():
                from mcp_server.database.postgres_database import postgres_db
                
                rows = await postgres_db.execute_query("""
                    SELECT version, name, status, checksum, executed_at, 
                           execution_time_ms, error_message
                    FROM schema_migrations 
                    ORDER BY executed_at ASC
                """)
                
            else:
                conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
                if not conn:
                    from mcp_server.database.memory_database import get_db_connection
                    conn = get_db_connection()
                
                if not conn:
                    return
                
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT version, name, status, checksum, executed_at, 
                               execution_time_ms, error_message
                        FROM schema_migrations 
                        ORDER BY executed_at ASC
                    """)
                    
                    columns = [desc[0] for desc in cursor.description]
                    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    
                finally:
                    conn.close()
            
            for row in rows:
                executed_at = row['executed_at']
                if isinstance(executed_at, str):
                    try:
                        executed_at = datetime.fromisoformat(executed_at.replace('Z', '+00:00'))
                    except ValueError:
                        executed_at = datetime.now()
                
                record = MigrationRecord(
                    version=row['version'],
                    name=row['name'],
                    status=MigrationStatus(row['status']),
                    checksum=row['checksum'],
                    executed_at=executed_at,
                    execution_time_ms=row['execution_time_ms'] or 0,
                    error_message=row['error_message']
                )
                
                self.migration_history.append(record)
                
        except Exception as e:
            logger.warning(f"Could not load migration history: {e}")
    
    async def _determine_current_version(self):
        """Determine the current database version."""
        successful_migrations = [
            record for record in self.migration_history 
            if record.status == MigrationStatus.SUCCESS
        ]
        
        if successful_migrations:
            # Get the latest successful migration version
            latest = max(successful_migrations, key=lambda x: x.executed_at)
            self.current_version = latest.version
        else:
            self.current_version = "0.0.0"
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations that need to be executed."""
        executed_versions = {
            record.version for record in self.migration_history 
            if record.status == MigrationStatus.SUCCESS
        }
        
        pending = [
            migration for version, migration in self.migrations.items()
            if version not in executed_versions
        ]
        
        # Sort by version
        pending.sort(key=lambda x: x.version)
        return pending
    
    async def validate_migration_integrity(self, migration: Migration) -> bool:
        """Validate migration integrity by checking checksums."""
        if not self.validate_checksums:
            return True
        
        # Check if migration was previously executed
        for record in self.migration_history:
            if record.version == migration.version:
                if record.checksum != migration.checksum:
                    logger.error(f"Migration {migration.version} checksum mismatch! "
                               f"Expected: {record.checksum}, Got: {migration.checksum}")
                    return False
                break
        
        return True
    
    async def check_migration_dependencies(self, migration: Migration) -> bool:
        """Check if migration dependencies are satisfied."""
        if not migration.dependencies:
            return True
        
        executed_versions = {
            record.version for record in self.migration_history 
            if record.status == MigrationStatus.SUCCESS
        }
        
        for dependency in migration.dependencies:
            if dependency not in executed_versions:
                logger.error(f"Migration {migration.version} depends on {dependency} "
                           "which has not been executed")
                return False
        
        return True
    
    async def execute_migration(self, migration: Migration) -> bool:
        """Execute a single migration."""
        logger.info(f"Executing migration {migration.version}: {migration.name}")
        
        # Validate migration
        if not await self.validate_migration_integrity(migration):
            return False
        
        if not await self.check_migration_dependencies(migration):
            return False
        
        # Record migration start
        await self._record_migration_start(migration)
        
        start_time = datetime.now()
        
        try:
            # Create backup if configured
            if self.backup_before_migration:
                await self._create_pre_migration_backup(migration)
            
            # Execute the migration
            if db_manager.is_postgresql():
                await self._execute_postgresql_migration(migration)
            else:
                await self._execute_sqlite_migration(migration)
            
            # Calculate execution time
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Record success
            await self._record_migration_success(migration, execution_time)
            
            logger.info(f"Migration {migration.version} completed successfully in {execution_time}ms")
            return True
            
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            error_message = str(e)
            
            # Record failure
            await self._record_migration_failure(migration, execution_time, error_message)
            
            logger.error(f"Migration {migration.version} failed: {error_message}")
            return False
    
    async def _execute_postgresql_migration(self, migration: Migration):
        """Execute PostgreSQL migration."""
        from mcp_server.database.postgres_database import postgres_db
        
        async with postgres_db.pool.acquire() as conn:
            async with conn.transaction():
                # Split SQL into individual statements
                statements = [stmt.strip() for stmt in migration.up_sql.split(';') 
                             if stmt.strip()]
                
                for statement in statements:
                    await conn.execute(statement)
    
    async def _execute_sqlite_migration(self, migration: Migration):
        """Execute SQLite migration."""
        conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
        if not conn:
            from mcp_server.database.memory_database import get_db_connection
            conn = get_db_connection()
        
        if not conn:
            raise RuntimeError("Could not establish SQLite connection")
        
        try:
            cursor = conn.cursor()
            
            # Split SQL into individual statements
            statements = [stmt.strip() for stmt in migration.up_sql.split(';') 
                         if stmt.strip()]
            
            for statement in statements:
                cursor.execute(statement)
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def _record_migration_start(self, migration: Migration):
        """Record migration execution start."""
        if db_manager.is_postgresql():
            from mcp_server.database.postgres_database import postgres_db
            
            await postgres_db.execute_command("""
                INSERT INTO schema_migrations (version, name, status, checksum)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (version) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = NOW()
            """, migration.version, migration.name, MigrationStatus.RUNNING.value, migration.checksum)
            
        else:
            conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
            if not conn:
                from mcp_server.database.memory_database import get_db_connection
                conn = get_db_connection()
            
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO schema_migrations 
                        (version, name, status, checksum, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (migration.version, migration.name, MigrationStatus.RUNNING.value, migration.checksum))
                    conn.commit()
                finally:
                    conn.close()
    
    async def _record_migration_success(self, migration: Migration, execution_time_ms: int):
        """Record successful migration execution."""
        if db_manager.is_postgresql():
            from mcp_server.database.postgres_database import postgres_db
            
            await postgres_db.execute_command("""
                UPDATE schema_migrations 
                SET status = $1, execution_time_ms = $2, executed_at = NOW(), updated_at = NOW()
                WHERE version = $3
            """, MigrationStatus.SUCCESS.value, execution_time_ms, migration.version)
            
        else:
            conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
            if not conn:
                from mcp_server.database.memory_database import get_db_connection
                conn = get_db_connection()
            
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE schema_migrations 
                        SET status = ?, execution_time_ms = ?, executed_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE version = ?
                    """, (MigrationStatus.SUCCESS.value, execution_time_ms, migration.version))
                    conn.commit()
                finally:
                    conn.close()
    
    async def _record_migration_failure(self, migration: Migration, execution_time_ms: int, error_message: str):
        """Record failed migration execution."""
        if db_manager.is_postgresql():
            from mcp_server.database.postgres_database import postgres_db
            
            await postgres_db.execute_command("""
                UPDATE schema_migrations 
                SET status = $1, execution_time_ms = $2, error_message = $3,
                    executed_at = NOW(), updated_at = NOW()
                WHERE version = $4
            """, MigrationStatus.FAILED.value, execution_time_ms, error_message, migration.version)
            
        else:
            conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
            if not conn:
                from mcp_server.database.memory_database import get_db_connection
                conn = get_db_connection()
            
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE schema_migrations 
                        SET status = ?, execution_time_ms = ?, error_message = ?,
                            executed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                        WHERE version = ?
                    """, (MigrationStatus.FAILED.value, execution_time_ms, error_message, migration.version))
                    conn.commit()
                finally:
                    conn.close()
    
    async def _create_pre_migration_backup(self, migration: Migration):
        """Create database backup before migration (if configured)."""
        try:
            from mcp_server.database.backup_manager import get_backup_manager
            backup_manager = await get_backup_manager()
            
            backup_result = await backup_manager.create_backup(f'pre_migration_{migration.version}')
            logger.info(f"Created pre-migration backup: {backup_result.get('backup_path')}")
            
        except Exception as e:
            logger.warning(f"Could not create pre-migration backup: {e}")
    
    async def migrate_to_latest(self) -> Dict[str, Any]:
        """Execute all pending migrations to bring database to latest version."""
        pending_migrations = await self.get_pending_migrations()
        
        if not pending_migrations:
            return {
                "message": "Database is already up to date",
                "current_version": self.current_version,
                "executed_migrations": 0
            }
        
        executed_count = 0
        failed_migrations = []
        
        for migration in pending_migrations:
            success = await self.execute_migration(migration)
            
            if success:
                executed_count += 1
            else:
                failed_migrations.append(migration.version)
                # Stop on first failure for safety
                break
        
        # Reload history to update current version
        await self._load_migration_history()
        await self._determine_current_version()
        
        result = {
            "executed_migrations": executed_count,
            "failed_migrations": failed_migrations,
            "current_version": self.current_version,
            "total_pending": len(pending_migrations)
        }
        
        if failed_migrations:
            result["message"] = f"Migration failed at version {failed_migrations[0]}"
            result["status"] = "partial"
        else:
            result["message"] = "All migrations executed successfully"
            result["status"] = "success"
        
        return result
    
    async def rollback_migration(self, target_version: str) -> Dict[str, Any]:
        """Rollback database to a specific version."""
        if target_version not in self.migrations:
            return {"error": f"Migration version {target_version} not found"}
        
        # Find migrations to rollback (in reverse order)
        migrations_to_rollback = []
        
        for record in reversed(self.migration_history):
            if record.status == MigrationStatus.SUCCESS:
                if record.version == target_version:
                    break
                migrations_to_rollback.append(record.version)
        
        if not migrations_to_rollback:
            return {
                "message": f"Already at version {target_version}",
                "current_version": self.current_version
            }
        
        rollback_count = 0
        failed_rollbacks = []
        
        for version in migrations_to_rollback:
            if version in self.migrations:
                migration = self.migrations[version]
                success = await self._execute_rollback(migration)
                
                if success:
                    rollback_count += 1
                else:
                    failed_rollbacks.append(version)
                    break
        
        # Reload history
        await self._load_migration_history()
        await self._determine_current_version()
        
        result = {
            "rolled_back": rollback_count,
            "failed_rollbacks": failed_rollbacks,
            "current_version": self.current_version,
            "target_version": target_version
        }
        
        if failed_rollbacks:
            result["message"] = f"Rollback failed at version {failed_rollbacks[0]}"
            result["status"] = "partial"
        else:
            result["message"] = f"Successfully rolled back to version {target_version}"
            result["status"] = "success"
        
        return result
    
    async def _execute_rollback(self, migration: Migration) -> bool:
        """Execute rollback for a specific migration."""
        if not migration.down_sql:
            logger.error(f"Migration {migration.version} has no rollback SQL")
            return False
        
        logger.info(f"Rolling back migration {migration.version}: {migration.name}")
        
        start_time = datetime.now()
        
        try:
            if db_manager.is_postgresql():
                from mcp_server.database.postgres_database import postgres_db
                
                async with postgres_db.pool.acquire() as conn:
                    async with conn.transaction():
                        statements = [stmt.strip() for stmt in migration.down_sql.split(';') 
                                     if stmt.strip()]
                        
                        for statement in statements:
                            await conn.execute(statement)
            else:
                conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
                if not conn:
                    from mcp_server.database.memory_database import get_db_connection
                    conn = get_db_connection()
                
                if not conn:
                    raise RuntimeError("Could not establish SQLite connection")
                
                try:
                    cursor = conn.cursor()
                    
                    statements = [stmt.strip() for stmt in migration.down_sql.split(';') 
                                 if stmt.strip()]
                    
                    for statement in statements:
                        cursor.execute(statement)
                    
                    conn.commit()
                    
                finally:
                    conn.close()
            
            # Record rollback
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await self._record_rollback(migration, execution_time)
            
            logger.info(f"Migration {migration.version} rolled back successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback of migration {migration.version} failed: {e}")
            return False
    
    async def _record_rollback(self, migration: Migration, execution_time_ms: int):
        """Record migration rollback."""
        if db_manager.is_postgresql():
            from mcp_server.database.postgres_database import postgres_db
            
            await postgres_db.execute_command("""
                UPDATE schema_migrations 
                SET status = $1, execution_time_ms = $2, executed_at = NOW(), updated_at = NOW()
                WHERE version = $3
            """, MigrationStatus.ROLLBACK.value, execution_time_ms, migration.version)
            
        else:
            conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
            if not conn:
                from mcp_server.database.memory_database import get_db_connection
                conn = get_db_connection()
            
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE schema_migrations 
                        SET status = ?, execution_time_ms = ?, executed_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE version = ?
                    """, (MigrationStatus.ROLLBACK.value, execution_time_ms, migration.version))
                    conn.commit()
                finally:
                    conn.close()
    
    def get_schema_status(self) -> Dict[str, Any]:
        """Get current schema status and migration information."""
        pending = len([m for m in self.migrations.values() 
                      if m.version not in {r.version for r in self.migration_history 
                                         if r.status == MigrationStatus.SUCCESS}])
        
        failed = len([r for r in self.migration_history 
                     if r.status == MigrationStatus.FAILED])
        
        return {
            "current_version": self.current_version,
            "total_migrations": len(self.migrations),
            "pending_migrations": pending,
            "failed_migrations": failed,
            "migration_history": [
                {
                    "version": record.version,
                    "name": record.name,
                    "status": record.status.value,
                    "executed_at": record.executed_at.isoformat(),
                    "execution_time_ms": record.execution_time_ms,
                    "error_message": record.error_message
                }
                for record in self.migration_history
            ]
        }
    
    async def create_migration_template(self, version: str, name: str, description: str = "") -> str:
        """Create a new migration template file."""
        filename = f"{version}_{name.replace(' ', '_').lower()}.sql"
        filepath = self.migrations_dir / filename
        
        template = f"""-- @version: {version}
-- @name: {name}
-- @description: {description}
-- @dependencies: 

-- UP
-- Add your migration SQL here


-- DOWN
-- Add your rollback SQL here

"""
        
        filepath.write_text(template)
        logger.info(f"Created migration template: {filepath}")
        
        return str(filepath)

# Global schema manager instance
schema_manager = DatabaseSchemaManager()

async def get_schema_manager() -> DatabaseSchemaManager:
    """Get the schema manager instance."""
    return schema_manager