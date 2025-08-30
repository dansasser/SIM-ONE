import os
import subprocess
import logging
import shutil
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio

from mcp_server.database.database_manager import db_manager

logger = logging.getLogger(__name__)

class BackupManager:
    """
    Database backup and recovery manager for both SQLite and PostgreSQL.
    Supports automated backups, retention policies, and point-in-time recovery.
    """
    
    def __init__(self):
        self.backup_dir = Path(os.getenv('BACKUP_DIR', './backups'))
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup retention settings
        self.daily_retention = int(os.getenv('BACKUP_DAILY_RETENTION', '7'))  # 7 days
        self.weekly_retention = int(os.getenv('BACKUP_WEEKLY_RETENTION', '4'))  # 4 weeks
        self.monthly_retention = int(os.getenv('BACKUP_MONTHLY_RETENTION', '3'))  # 3 months
    
    async def create_backup(self, backup_type: str = 'manual') -> Dict[str, Any]:
        """Create a database backup."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if db_manager.is_postgresql():
            return await self._create_postgres_backup(timestamp, backup_type)
        else:
            return await self._create_sqlite_backup(timestamp, backup_type)
    
    async def _create_postgres_backup(self, timestamp: str, backup_type: str) -> Dict[str, Any]:
        """Create PostgreSQL backup using pg_dump."""
        backup_filename = f"simone_mcp_postgres_{backup_type}_{timestamp}.sql.gz"
        backup_path = self.backup_dir / backup_filename
        
        # PostgreSQL connection parameters
        pg_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'simone_mcp'),
            'username': os.getenv('POSTGRES_USER', 'simone'),
        }
        
        # Set PGPASSWORD environment variable for pg_dump
        env = os.environ.copy()
        env['PGPASSWORD'] = os.getenv('POSTGRES_PASSWORD', '')
        
        try:
            # Create pg_dump command
            cmd = [
                'pg_dump',
                '-h', pg_params['host'],
                '-p', pg_params['port'],
                '-U', pg_params['username'],
                '-d', pg_params['database'],
                '--verbose',
                '--no-password',
                '--format=custom',
                '--compress=9',
                '--no-privileges',
                '--no-owner'
            ]
            
            logger.info(f"Creating PostgreSQL backup: {backup_filename}")
            
            # Run pg_dump and compress output
            with gzip.open(backup_path, 'wb') as gz_file:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise RuntimeError(f"pg_dump failed: {stderr.decode()}")
                
                gz_file.write(stdout)
            
            backup_info = {
                'type': 'postgresql',
                'backup_type': backup_type,
                'filename': backup_filename,
                'path': str(backup_path),
                'size_bytes': backup_path.stat().st_size,
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'database_params': {k: v for k, v in pg_params.items() if k != 'password'}
            }
            
            logger.info(f"PostgreSQL backup completed: {backup_info['size_bytes']} bytes")
            return backup_info
            
        except Exception as e:
            logger.error(f"PostgreSQL backup failed: {e}")
            if backup_path.exists():
                backup_path.unlink()
            raise
    
    async def _create_sqlite_backup(self, timestamp: str, backup_type: str) -> Dict[str, Any]:
        """Create SQLite backup using file copy and compression."""
        from mcp_server.database.memory_database import DB_FILE
        
        if not DB_FILE.exists():
            raise FileNotFoundError(f"SQLite database file not found: {DB_FILE}")
        
        backup_filename = f"simone_mcp_sqlite_{backup_type}_{timestamp}.db.gz"
        backup_path = self.backup_dir / backup_filename
        
        try:
            logger.info(f"Creating SQLite backup: {backup_filename}")
            
            # Create compressed backup
            with open(DB_FILE, 'rb') as src, gzip.open(backup_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            
            backup_info = {
                'type': 'sqlite',
                'backup_type': backup_type,
                'filename': backup_filename,
                'path': str(backup_path),
                'size_bytes': backup_path.stat().st_size,
                'original_size': DB_FILE.stat().st_size,
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'source_file': str(DB_FILE)
            }
            
            logger.info(f"SQLite backup completed: {backup_info['size_bytes']} bytes (compressed from {backup_info['original_size']})")
            return backup_info
            
        except Exception as e:
            logger.error(f"SQLite backup failed: {e}")
            if backup_path.exists():
                backup_path.unlink()
            raise
    
    async def restore_backup(self, backup_filename: str) -> Dict[str, Any]:
        """Restore database from a backup file."""
        backup_path = self.backup_dir / backup_filename
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Determine backup type from filename
        if 'postgres' in backup_filename:
            return await self._restore_postgres_backup(backup_path)
        elif 'sqlite' in backup_filename:
            return await self._restore_sqlite_backup(backup_path)
        else:
            raise ValueError(f"Cannot determine backup type from filename: {backup_filename}")
    
    async def _restore_postgres_backup(self, backup_path: Path) -> Dict[str, Any]:
        """Restore PostgreSQL backup using pg_restore."""
        # PostgreSQL connection parameters
        pg_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'simone_mcp'),
            'username': os.getenv('POSTGRES_USER', 'simone'),
        }
        
        # Set PGPASSWORD environment variable
        env = os.environ.copy()
        env['PGPASSWORD'] = os.getenv('POSTGRES_PASSWORD', '')
        
        try:
            logger.info(f"Restoring PostgreSQL backup: {backup_path.name}")
            
            # Decompress backup first
            temp_file = backup_path.with_suffix('')
            with gzip.open(backup_path, 'rb') as gz_file, open(temp_file, 'wb') as out_file:
                shutil.copyfileobj(gz_file, out_file)
            
            # Create pg_restore command
            cmd = [
                'pg_restore',
                '-h', pg_params['host'],
                '-p', pg_params['port'],
                '-U', pg_params['username'],
                '-d', pg_params['database'],
                '--verbose',
                '--no-password',
                '--clean',
                '--if-exists',
                '--no-privileges',
                '--no-owner',
                str(temp_file)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await process.communicate()
            
            # Clean up temporary file
            temp_file.unlink()
            
            if process.returncode != 0:
                raise RuntimeError(f"pg_restore failed: {stderr.decode()}")
            
            restore_info = {
                'type': 'postgresql',
                'backup_file': backup_path.name,
                'restored_at': datetime.now().isoformat(),
                'stdout': stdout.decode(),
                'success': True
            }
            
            logger.info(f"PostgreSQL restore completed: {backup_path.name}")
            return restore_info
            
        except Exception as e:
            logger.error(f"PostgreSQL restore failed: {e}")
            # Clean up temporary file if it exists
            temp_file = backup_path.with_suffix('')
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    async def _restore_sqlite_backup(self, backup_path: Path) -> Dict[str, Any]:
        """Restore SQLite backup by decompressing and replacing database file."""
        from mcp_server.database.memory_database import DB_FILE
        
        try:
            logger.info(f"Restoring SQLite backup: {backup_path.name}")
            
            # Create backup of current database
            current_backup = DB_FILE.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
            if DB_FILE.exists():
                shutil.copy2(DB_FILE, current_backup)
                logger.info(f"Current database backed up to: {current_backup}")
            
            # Restore from compressed backup
            with gzip.open(backup_path, 'rb') as src, open(DB_FILE, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            
            restore_info = {
                'type': 'sqlite',
                'backup_file': backup_path.name,
                'restored_at': datetime.now().isoformat(),
                'current_backup': str(current_backup) if current_backup.exists() else None,
                'restored_size': DB_FILE.stat().st_size,
                'success': True
            }
            
            logger.info(f"SQLite restore completed: {restore_info['restored_size']} bytes")
            return restore_info
            
        except Exception as e:
            logger.error(f"SQLite restore failed: {e}")
            raise
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backup files."""
        backups = []
        
        for backup_file in self.backup_dir.glob("simone_mcp_*.gz"):
            try:
                stat = backup_file.stat()
                
                # Parse backup info from filename
                parts = backup_file.stem.split('_')
                if len(parts) >= 4:
                    db_type = parts[2]  # postgres or sqlite
                    backup_type = parts[3]  # manual, daily, weekly, etc.
                    timestamp = '_'.join(parts[4:]) if len(parts) > 4 else 'unknown'
                else:
                    db_type = 'unknown'
                    backup_type = 'unknown'
                    timestamp = 'unknown'
                
                backups.append({
                    'filename': backup_file.name,
                    'path': str(backup_file),
                    'size_bytes': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'db_type': db_type,
                    'backup_type': backup_type,
                    'timestamp': timestamp,
                    'age_days': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days
                })
            except Exception as e:
                logger.warning(f"Could not parse backup file {backup_file}: {e}")
        
        # Sort by creation time, newest first
        backups.sort(key=lambda x: x['created_at'], reverse=True)
        return backups
    
    def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up old backups according to retention policy."""
        backups = self.list_backups()
        removed = []
        kept = []
        
        now = datetime.now()
        
        for backup in backups:
            backup_date = datetime.fromisoformat(backup['created_at'].replace('Z', '+00:00'))
            age_days = (now - backup_date).days
            
            # Determine if backup should be kept
            should_keep = False
            reason = ""
            
            if backup['backup_type'] == 'manual':
                should_keep = True
                reason = "Manual backup - never auto-deleted"
            elif age_days <= self.daily_retention:
                should_keep = True
                reason = f"Within daily retention ({self.daily_retention} days)"
            elif age_days <= self.weekly_retention * 7 and backup_date.weekday() == 6:  # Sunday
                should_keep = True
                reason = f"Weekly backup within retention ({self.weekly_retention} weeks)"
            elif age_days <= self.monthly_retention * 30 and backup_date.day == 1:  # First of month
                should_keep = True
                reason = f"Monthly backup within retention ({self.monthly_retention} months)"
            else:
                reason = f"Exceeds retention policy ({age_days} days old)"
            
            if should_keep:
                kept.append({
                    'filename': backup['filename'],
                    'age_days': age_days,
                    'reason': reason
                })
            else:
                try:
                    backup_path = Path(backup['path'])
                    backup_path.unlink()
                    removed.append({
                        'filename': backup['filename'],
                        'age_days': age_days,
                        'reason': reason,
                        'size_bytes': backup['size_bytes']
                    })
                    logger.info(f"Removed old backup: {backup['filename']} ({age_days} days old)")
                except Exception as e:
                    logger.error(f"Failed to remove backup {backup['filename']}: {e}")
        
        total_removed_size = sum(b['size_bytes'] for b in removed)
        
        return {
            'removed_count': len(removed),
            'kept_count': len(kept),
            'total_removed_size': total_removed_size,
            'removed_backups': removed,
            'kept_backups': kept,
            'cleanup_date': now.isoformat()
        }

# Global backup manager instance
backup_manager = BackupManager()