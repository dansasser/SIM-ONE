"""
Embedding Cache for Semantic Encoding Protocol

High-performance caching system for embeddings that respects SIM-ONE's
energy stewardship principles while providing fast retrieval.

Author: SIM-ONE Framework / Manus AI Enhancement
"""

import logging
import asyncio
import json
import pickle
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import aiosqlite

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    High-performance embedding cache with TTL and persistence.
    
    Implements energy-efficient caching strategies:
    - In-memory LRU cache for hot data
    - SQLite persistence for cold storage
    - Automatic TTL management
    - Compression for large embeddings
    """
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24, cache_dir: str = "./cache"):
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache (LRU)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []  # For LRU tracking
        
        # Persistent storage
        self.db_path = self.cache_dir / "embeddings.db"
        self.db_connection = None
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'evictions': 0,
            'total_size': 0
        }
        
        # Configuration
        self.memory_cache_ratio = 0.3  # 30% of max_size in memory
        self.memory_cache_size = int(max_size * self.memory_cache_ratio)
        self.compression_threshold = 1000  # Compress vectors larger than this
        
        logger.info(f"EmbeddingCache initialized: max_size={max_size}, ttl={ttl_hours}h, memory_ratio={self.memory_cache_ratio}")
    
    async def initialize(self):
        """Initialize the cache system."""
        try:
            await self._initialize_database()
            await self._load_hot_cache()
            await self._cleanup_expired()
            
            logger.info("EmbeddingCache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingCache: {e}")
            raise
    
    async def _initialize_database(self):
        """Initialize SQLite database for persistent storage."""
        self.db_connection = await aiosqlite.connect(str(self.db_path))
        
        await self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                vector_data BLOB,
                metadata TEXT,
                created_at TIMESTAMP,
                accessed_at TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                compressed BOOLEAN DEFAULT FALSE
            )
        """)
        
        await self.db_connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_accessed_at 
            ON embeddings(accessed_at)
        """)
        
        await self.db_connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_created_at 
            ON embeddings(created_at)
        """)
        
        await self.db_connection.commit()
    
    async def _load_hot_cache(self):
        """Load frequently accessed embeddings into memory cache."""
        try:
            # Load most recently accessed items into memory
            cursor = await self.db_connection.execute("""
                SELECT key, vector_data, metadata, compressed
                FROM embeddings 
                ORDER BY access_count DESC, accessed_at DESC 
                LIMIT ?
            """, (self.memory_cache_size,))
            
            rows = await cursor.fetchall()
            
            for key, vector_data, metadata, compressed in rows:
                try:
                    # Deserialize vector data
                    if compressed:
                        vector_data = self._decompress_vector(vector_data)
                    
                    embedding_data = pickle.loads(vector_data)
                    metadata_dict = json.loads(metadata)
                    
                    cache_entry = {
                        'data': embedding_data,
                        'metadata': metadata_dict,
                        'timestamp': datetime.now()
                    }
                    
                    self.memory_cache[key] = cache_entry
                    self.access_order.append(key)
                    
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {key}: {e}")
            
            logger.info(f"Loaded {len(self.memory_cache)} entries into memory cache")
            
        except Exception as e:
            logger.error(f"Failed to load hot cache: {e}")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve embedding from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached embedding data or None if not found/expired
        """
        try:
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Check TTL
                if self._is_expired(entry['timestamp']):
                    await self._remove_from_memory(key)
                    self.stats['misses'] += 1
                    return None
                
                # Update access order for LRU
                self._update_access_order(key)
                
                self.stats['hits'] += 1
                self.stats['memory_hits'] += 1
                
                return entry['data']
            
            # Check persistent storage
            disk_result = await self._get_from_disk(key)
            if disk_result:
                # Promote to memory cache if frequently accessed
                await self._maybe_promote_to_memory(key, disk_result)
                
                self.stats['hits'] += 1
                self.stats['disk_hits'] += 1
                
                return disk_result
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            self.stats['misses'] += 1
            return None
    
    async def set(self, key: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """
        Store embedding in cache.
        
        Args:
            key: Cache key
            data: Embedding data to store
            metadata: Optional metadata
        """
        try:
            timestamp = datetime.now()
            metadata = metadata or {}
            
            cache_entry = {
                'data': data,
                'metadata': metadata,
                'timestamp': timestamp
            }
            
            # Store in memory cache if there's space
            if len(self.memory_cache) < self.memory_cache_size:
                self.memory_cache[key] = cache_entry
                self.access_order.append(key)
            else:
                # Check if this should replace an existing entry
                if await self._should_cache_in_memory(key, data):
                    await self._evict_lru_memory()
                    self.memory_cache[key] = cache_entry
                    self.access_order.append(key)
            
            # Always store in persistent storage
            await self._store_to_disk(key, data, metadata, timestamp)
            
            self.stats['total_size'] += 1
            
        except Exception as e:
            logger.error(f"Error storing to cache: {e}")
    
    async def _get_from_disk(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve embedding from persistent storage."""
        try:
            cursor = await self.db_connection.execute("""
                SELECT vector_data, metadata, created_at, compressed
                FROM embeddings 
                WHERE key = ?
            """, (key,))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            vector_data, metadata, created_at, compressed = row
            
            # Check TTL
            created_time = datetime.fromisoformat(created_at)
            if self._is_expired(created_time):
                await self._remove_from_disk(key)
                return None
            
            # Deserialize data
            if compressed:
                vector_data = self._decompress_vector(vector_data)
            
            embedding_data = pickle.loads(vector_data)
            
            # Update access statistics
            await self.db_connection.execute("""
                UPDATE embeddings 
                SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE key = ?
            """, (key,))
            await self.db_connection.commit()
            
            return embedding_data
            
        except Exception as e:
            logger.error(f"Error retrieving from disk: {e}")
            return None
    
    async def _store_to_disk(self, key: str, data: Dict[str, Any], metadata: Dict[str, Any], timestamp: datetime):
        """Store embedding to persistent storage."""
        try:
            # Serialize data
            vector_data = pickle.dumps(data)
            
            # Compress if large
            compressed = False
            if len(vector_data) > self.compression_threshold:
                vector_data = self._compress_vector(vector_data)
                compressed = True
            
            metadata_json = json.dumps(metadata)
            
            # Store or update
            await self.db_connection.execute("""
                INSERT OR REPLACE INTO embeddings 
                (key, vector_data, metadata, created_at, accessed_at, compressed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (key, vector_data, metadata_json, timestamp.isoformat(), timestamp.isoformat(), compressed))
            
            await self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing to disk: {e}")
    
    def _compress_vector(self, data: bytes) -> bytes:
        """Compress vector data for storage efficiency."""
        import gzip
        return gzip.compress(data)
    
    def _decompress_vector(self, data: bytes) -> bytes:
        """Decompress vector data."""
        import gzip
        return gzip.decompress(data)
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired."""
        expiry_time = timestamp + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def _update_access_order(self, key: str):
        """Update LRU access order."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    async def _evict_lru_memory(self):
        """Evict least recently used item from memory cache."""
        if not self.access_order:
            return
        
        lru_key = self.access_order.pop(0)
        if lru_key in self.memory_cache:
            del self.memory_cache[lru_key]
            self.stats['evictions'] += 1
    
    async def _remove_from_memory(self, key: str):
        """Remove item from memory cache."""
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        if key in self.access_order:
            self.access_order.remove(key)
    
    async def _remove_from_disk(self, key: str):
        """Remove expired item from disk storage."""
        try:
            await self.db_connection.execute("DELETE FROM embeddings WHERE key = ?", (key,))
            await self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error removing from disk: {e}")
    
    async def _should_cache_in_memory(self, key: str, data: Dict[str, Any]) -> bool:
        """Determine if item should be cached in memory."""
        # Simple heuristic: cache if it's a reasonably sized embedding
        if 'vector' in data and isinstance(data['vector'], np.ndarray):
            vector_size = data['vector'].nbytes
            # Cache vectors under 100KB in memory
            return vector_size < 100 * 1024
        
        return True
    
    async def _maybe_promote_to_memory(self, key: str, data: Dict[str, Any]):
        """Promote frequently accessed item to memory cache."""
        try:
            # Check access count
            cursor = await self.db_connection.execute("""
                SELECT access_count FROM embeddings WHERE key = ?
            """, (key,))
            
            row = await cursor.fetchone()
            if row and row[0] > 3:  # Promote if accessed more than 3 times
                if len(self.memory_cache) >= self.memory_cache_size:
                    await self._evict_lru_memory()
                
                cache_entry = {
                    'data': data,
                    'metadata': {},
                    'timestamp': datetime.now()
                }
                
                self.memory_cache[key] = cache_entry
                self.access_order.append(key)
                
        except Exception as e:
            logger.error(f"Error promoting to memory: {e}")
    
    async def _cleanup_expired(self):
        """Remove expired entries from storage."""
        try:
            expiry_time = datetime.now() - timedelta(hours=self.ttl_hours)
            
            # Clean memory cache
            expired_keys = []
            for key, entry in self.memory_cache.items():
                if self._is_expired(entry['timestamp']):
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self._remove_from_memory(key)
            
            # Clean disk storage
            await self.db_connection.execute("""
                DELETE FROM embeddings 
                WHERE created_at < ?
            """, (expiry_time.isoformat(),))
            
            await self.db_connection.commit()
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def size(self) -> int:
        """Get current cache size."""
        try:
            cursor = await self.db_connection.execute("SELECT COUNT(*) FROM embeddings")
            row = await cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            return 0
    
    async def clear(self):
        """Clear all cache entries."""
        try:
            self.memory_cache.clear()
            self.access_order.clear()
            
            await self.db_connection.execute("DELETE FROM embeddings")
            await self.db_connection.commit()
            
            self.stats = {
                'hits': 0,
                'misses': 0,
                'memory_hits': 0,
                'disk_hits': 0,
                'evictions': 0,
                'total_size': 0
            }
            
            logger.info("Cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        memory_hit_rate = self.stats['memory_hits'] / self.stats['hits'] if self.stats['hits'] > 0 else 0.0
        
        return {
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_hit_rate': memory_hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_capacity': self.memory_cache_size,
            'evictions': self.stats['evictions'],
            'total_entries': self.stats['total_size']
        }
    
    async def shutdown(self):
        """Clean shutdown of cache system."""
        try:
            logger.info("Shutting down EmbeddingCache...")
            
            # Final cleanup
            await self._cleanup_expired()
            
            # Close database connection
            if self.db_connection:
                await self.db_connection.close()
            
            # Clear memory
            self.memory_cache.clear()
            self.access_order.clear()
            
            logger.info("EmbeddingCache shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during cache shutdown: {e}")

