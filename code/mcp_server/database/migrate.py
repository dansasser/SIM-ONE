#!/usr/bin/env python3
"""
Database migration utility for SIM-ONE Framework MCP Server.
Handles both PostgreSQL migrations via Alembic and SQLite schema upgrades.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_server.database.database_manager import db_manager, DatabaseType
from mcp_server.config import settings

logger = logging.getLogger(__name__)

class MigrationManager:
    """Handles database migrations for both SQLite and PostgreSQL."""
    
    def __init__(self):
        self.migrations_dir = Path(__file__).parent / "migrations"
        
    async def run_migrations(self, direction: str = "upgrade"):
        """Run database migrations."""
        await db_manager.initialize()
        
        if db_manager.get_database_type() == DatabaseType.POSTGRESQL:
            await self._run_postgres_migrations(direction)
        else:
            await self._run_sqlite_migrations(direction)
    
    async def _run_postgres_migrations(self, direction: str):
        """Run PostgreSQL migrations using Alembic."""
        logger.info("Running PostgreSQL migrations with Alembic")
        
        # Set up Alembic configuration
        alembic_cfg = Config(str(self.migrations_dir / "alembic.ini"))
        
        # Set database URL from environment
        database_url = self._get_postgres_url()
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        
        try:
            if direction == "upgrade":
                logger.info("Upgrading database to latest version")
                command.upgrade(alembic_cfg, "head")
            elif direction == "downgrade":
                logger.info("Downgrading database by one version")
                command.downgrade(alembic_cfg, "-1")
            elif direction.startswith("revision"):
                message = direction.split(":", 1)[1] if ":" in direction else "Auto-generated migration"
                logger.info(f"Creating new revision: {message}")
                command.revision(alembic_cfg, message=message, autogenerate=True)
            else:
                raise ValueError(f"Unknown migration direction: {direction}")
            
            logger.info("PostgreSQL migration completed successfully")
            
        except Exception as e:
            logger.error(f"PostgreSQL migration failed: {e}")
            raise
    
    async def _run_sqlite_migrations(self, direction: str):
        """Run SQLite migrations using the existing upgrade system."""
        logger.info("Running SQLite schema upgrade")
        
        try:
            from mcp_server.database.memory_database import initialize_database
            
            if direction == "upgrade":
                initialize_database()
                logger.info("SQLite schema upgrade completed")
            else:
                logger.warning(f"SQLite doesn't support '{direction}' migrations - only upgrades")
                
        except Exception as e:
            logger.error(f"SQLite migration failed: {e}")
            raise
    
    def _get_postgres_url(self) -> str:
        """Construct PostgreSQL URL from environment variables."""
        user = os.getenv('POSTGRES_USER', 'simone')
        password = os.getenv('POSTGRES_PASSWORD', '')
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        database = os.getenv('POSTGRES_DB', 'simone_mcp')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def check_migration_status(self):
        """Check current migration status."""
        await db_manager.initialize()
        
        if db_manager.get_database_type() == DatabaseType.POSTGRESQL:
            return await self._check_postgres_status()
        else:
            return await self._check_sqlite_status()
    
    async def _check_postgres_status(self):
        """Check PostgreSQL migration status."""
        try:
            alembic_cfg = Config(str(self.migrations_dir / "alembic.ini"))
            database_url = self._get_postgres_url()
            alembic_cfg.set_main_option("sqlalchemy.url", database_url)
            
            # Get current revision
            engine = create_engine(database_url)
            with engine.connect() as conn:
                context = command.config.Config.from_config(alembic_cfg)
                context.configure(connection=conn, target_metadata=None)
                
                current_rev = context.get_current_revision()
                
            # Get available revisions
            script = command.ScriptDirectory.from_config(alembic_cfg)
            head_rev = script.get_current_head()
            
            return {
                "database_type": "postgresql",
                "current_revision": current_rev,
                "head_revision": head_rev,
                "up_to_date": current_rev == head_rev,
                "migration_files": len(list(script.walk_revisions()))
            }
            
        except Exception as e:
            logger.error(f"Failed to check PostgreSQL status: {e}")
            return {
                "database_type": "postgresql",
                "error": str(e),
                "up_to_date": False
            }
    
    async def _check_sqlite_status(self):
        """Check SQLite schema status."""
        try:
            from mcp_server.database.memory_database import get_db_connection
            
            conn = get_db_connection()
            if not conn:
                raise Exception("Could not connect to SQLite database")
            
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            # Check if enhanced columns exist in memories table
            enhanced_columns = []
            if 'memories' in tables:
                cursor.execute("PRAGMA table_info(memories)")
                columns = [row[1] for row in cursor.fetchall()]
                
                expected_columns = [
                    'session_id', 'emotional_salience', 'rehearsal_count',
                    'last_accessed', 'confidence_score', 'memory_type',
                    'actors', 'context_tags'
                ]
                
                enhanced_columns = [col for col in expected_columns if col in columns]
            
            conn.close()
            
            return {
                "database_type": "sqlite",
                "tables": tables,
                "enhanced_columns": enhanced_columns,
                "up_to_date": len(enhanced_columns) >= 6,  # Most enhanced columns present
                "table_count": len(tables)
            }
            
        except Exception as e:
            logger.error(f"Failed to check SQLite status: {e}")
            return {
                "database_type": "sqlite",
                "error": str(e),
                "up_to_date": False
            }

async def main():
    """Main migration command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SIM-ONE Database Migration Utility")
    parser.add_argument(
        "action",
        choices=["upgrade", "downgrade", "status", "revision"],
        help="Migration action to perform"
    )
    parser.add_argument(
        "--message", "-m",
        help="Message for new revision (used with 'revision' action)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    migration_manager = MigrationManager()
    
    try:
        if args.action == "status":
            status = await migration_manager.check_migration_status()
            print("\n=== Database Migration Status ===")
            for key, value in status.items():
                print(f"{key}: {value}")
            
        elif args.action == "revision":
            if not args.message:
                print("Error: --message is required for creating revisions")
                return 1
            await migration_manager.run_migrations(f"revision:{args.message}")
            
        else:
            await migration_manager.run_migrations(args.action)
        
        print(f"\n✅ Migration '{args.action}' completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        logger.exception("Migration failed with exception")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))