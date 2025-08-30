import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json
import hashlib
import pickle
import os
from pathlib import Path

from mcp_server.database.database_manager import db_manager

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    memory_id: int
    content: str
    entity_name: str
    similarity_score: float
    embedding_vector: Optional[np.ndarray]
    metadata: Dict[str, Any]
    emotional_salience: float
    session_id: str
    timestamp: datetime

@dataclass
class EmbeddingModel:
    """Configuration for embedding model."""
    name: str
    dimensions: int
    max_tokens: int
    batch_size: int

class VectorSimilarityEngine:
    """
    Vector-based similarity search engine for the SIM-ONE memory system.
    Supports semantic search using embeddings with fallback to lightweight alternatives.
    """
    
    def __init__(self):
        self.embedding_cache = {}
        self.cache_dir = Path("./embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Available embedding models (lightweight options)
        self.models = {
            'mock': EmbeddingModel('mock', 384, 512, 32),  # Mock for testing
            'tfidf': EmbeddingModel('tfidf', 1000, 10000, 100),  # TF-IDF vectors
            'word2vec_lite': EmbeddingModel('word2vec_lite', 100, 1000, 50)  # Simplified word embeddings
        }
        
        self.current_model = 'mock'  # Default to mock for compatibility
        self.vector_cache_size = 10000
        self.similarity_threshold = 0.7
        
        # Initialize vector storage
        self.memory_vectors = {}  # memory_id -> vector mapping
        self.vector_metadata = {}  # memory_id -> metadata mapping
    
    async def initialize_vector_storage(self):
        """Initialize vector storage capabilities."""
        try:
            if db_manager.is_postgresql():
                await self._initialize_postgresql_vectors()
            else:
                await self._initialize_sqlite_vectors()
            
            logger.info("Vector storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector storage: {e}")
            raise
    
    async def _initialize_postgresql_vectors(self):
        """Initialize PostgreSQL vector storage with pgvector extension."""
        from mcp_server.database.postgres_database import postgres_db
        
        async with postgres_db.pool.acquire() as conn:
            async with conn.transaction():
                # Try to create pgvector extension (may require superuser privileges)
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    logger.info("PostgreSQL vector extension enabled")
                    vector_support = True
                except Exception as e:
                    logger.warning(f"Could not enable pgvector extension: {e}")
                    logger.info("Falling back to JSONB vector storage")
                    vector_support = False
                
                if vector_support:
                    # Create vector column with pgvector
                    await conn.execute("""
                        ALTER TABLE memories 
                        ADD COLUMN IF NOT EXISTS embedding_vector vector(384)
                    """)
                    
                    # Create vector similarity index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_memories_embedding_vector 
                        ON memories USING ivfflat (embedding_vector vector_cosine_ops)
                    """)
                else:
                    # Fallback to JSONB for vector storage
                    await conn.execute("""
                        ALTER TABLE memories 
                        ADD COLUMN IF NOT EXISTS embedding_vector JSONB
                    """)
                    
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_memories_embedding_vector_gin 
                        ON memories USING gin(embedding_vector)
                    """)
                
                # Add vector metadata columns
                await conn.execute("""
                    ALTER TABLE memories 
                    ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'mock',
                    ADD COLUMN IF NOT EXISTS embedding_created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    ADD COLUMN IF NOT EXISTS vector_hash TEXT
                """)
    
    async def _initialize_sqlite_vectors(self):
        """Initialize SQLite vector storage using JSONB-like TEXT columns."""
        conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
        if not conn:
            from mcp_server.database.memory_database import get_db_connection
            conn = get_db_connection()
        
        if not conn:
            raise RuntimeError("Could not establish SQLite connection")
        
        try:
            cursor = conn.cursor()
            
            # Add vector columns to memories table
            try:
                cursor.execute("""
                    ALTER TABLE memories 
                    ADD COLUMN embedding_vector TEXT
                """)
            except Exception:
                pass  # Column might already exist
            
            try:
                cursor.execute("""
                    ALTER TABLE memories 
                    ADD COLUMN embedding_model TEXT DEFAULT 'mock'
                """)
            except Exception:
                pass
            
            try:
                cursor.execute("""
                    ALTER TABLE memories 
                    ADD COLUMN embedding_created_at TEXT DEFAULT CURRENT_TIMESTAMP
                """)
            except Exception:
                pass
            
            try:
                cursor.execute("""
                    ALTER TABLE memories 
                    ADD COLUMN vector_hash TEXT
                """)
            except Exception:
                pass
            
            # Create index on vector_hash for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_vector_hash 
                ON memories(vector_hash)
            """)
            
            conn.commit()
            
        finally:
            conn.close()
    
    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """Generate a mock embedding vector based on text characteristics."""
        # Create a deterministic but varied embedding based on text content
        hash_val = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numeric values
        numeric_values = [int(hash_val[i:i+2], 16) for i in range(0, 32, 2)]
        
        # Pad or truncate to desired dimensions
        dimensions = self.models[self.current_model].dimensions
        
        if len(numeric_values) < dimensions:
            # Repeat values to reach desired length
            multiplier = dimensions // len(numeric_values) + 1
            extended_values = (numeric_values * multiplier)[:dimensions]
        else:
            extended_values = numeric_values[:dimensions]
        
        # Normalize to create unit vector
        vector = np.array(extended_values, dtype=np.float32)
        
        # Add some text-based features for better similarity
        text_lower = text.lower()
        
        # Length feature
        vector[0] = min(len(text) / 1000.0, 1.0) * 255
        
        # Word count feature
        word_count = len(text.split())
        vector[1] = min(word_count / 100.0, 1.0) * 255
        
        # Character frequency features
        for i, char in enumerate('etaoinshrdlu'):  # Common English letters
            if i + 2 < dimensions:
                freq = text_lower.count(char) / max(len(text), 1)
                vector[i + 2] = freq * 255
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _generate_tfidf_embedding(self, text: str, vocabulary: Optional[Dict] = None) -> np.ndarray:
        """Generate TF-IDF based embedding (simplified implementation)."""
        import re
        from collections import Counter
        
        # Simple tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        
        # Use a basic vocabulary or create one
        if vocabulary is None:
            # Create a simple vocabulary from common English words + current text
            common_words = [
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
                'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
                'memory', 'entity', 'emotion', 'cognitive', 'protocol', 'system',
                'ai', 'learning', 'data', 'information', 'process', 'analysis'
            ]
            vocabulary = {word: i for i, word in enumerate(common_words + list(set(words)))}
        
        # Create TF vector
        dimensions = min(self.models['tfidf'].dimensions, len(vocabulary))
        vector = np.zeros(dimensions)
        
        total_words = len(words)
        for word, count in word_counts.items():
            if word in vocabulary and vocabulary[word] < dimensions:
                tf = count / total_words if total_words > 0 else 0
                vector[vocabulary[word]] = tf
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> np.ndarray:
        """Generate embedding vector for given text."""
        if model is None:
            model = self.current_model
        
        # Check cache first
        cache_key = f"{model}:{hashlib.md5(text.encode()).hexdigest()}"
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            if model == 'mock':
                vector = self._generate_mock_embedding(text)
            elif model == 'tfidf':
                vector = self._generate_tfidf_embedding(text)
            else:
                # Fallback to mock
                logger.warning(f"Unknown embedding model '{model}', using mock")
                vector = self._generate_mock_embedding(text)
            
            # Cache the result
            if len(self.embedding_cache) < self.vector_cache_size:
                self.embedding_cache[cache_key] = vector
            
            return vector
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.models[model].dimensions)
    
    def calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Handle zero vectors
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(vector1, vector2) / (norm1 * norm2)
            
            # Clamp to [-1, 1] range
            similarity = max(-1.0, min(1.0, similarity))
            
            # Convert to [0, 1] range for easier interpretation
            return (similarity + 1.0) / 2.0
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    async def store_memory_embedding(self, memory_id: int, content: str) -> bool:
        """Generate and store embedding for a memory."""
        try:
            # Generate embedding
            vector = await self.generate_embedding(content)
            
            # Create vector hash for deduplication
            vector_hash = hashlib.md5(vector.tobytes()).hexdigest()
            
            # Store in database
            if db_manager.is_postgresql():
                await self._store_postgresql_embedding(memory_id, vector, vector_hash)
            else:
                await self._store_sqlite_embedding(memory_id, vector, vector_hash)
            
            # Update in-memory cache
            self.memory_vectors[memory_id] = vector
            self.vector_metadata[memory_id] = {
                'hash': vector_hash,
                'model': self.current_model,
                'created_at': datetime.now()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding for memory {memory_id}: {e}")
            return False
    
    async def _store_postgresql_embedding(self, memory_id: int, vector: np.ndarray, vector_hash: str):
        """Store embedding in PostgreSQL."""
        from mcp_server.database.postgres_database import postgres_db
        
        # Convert vector to appropriate format
        try:
            # Try pgvector format first
            vector_str = f"[{','.join(map(str, vector))}]"
            
            await postgres_db.execute_command("""
                UPDATE memories 
                SET embedding_vector = $1,
                    embedding_model = $2,
                    embedding_created_at = NOW(),
                    vector_hash = $3
                WHERE id = $4
            """, vector_str, self.current_model, vector_hash, memory_id)
            
        except Exception:
            # Fallback to JSONB format
            vector_json = json.dumps(vector.tolist())
            
            await postgres_db.execute_command("""
                UPDATE memories 
                SET embedding_vector = $1,
                    embedding_model = $2,
                    embedding_created_at = NOW(),
                    vector_hash = $3
                WHERE id = $4
            """, vector_json, self.current_model, vector_hash, memory_id)
    
    async def _store_sqlite_embedding(self, memory_id: int, vector: np.ndarray, vector_hash: str):
        """Store embedding in SQLite."""
        conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
        if not conn:
            from mcp_server.database.memory_database import get_db_connection
            conn = get_db_connection()
        
        if not conn:
            raise RuntimeError("Could not establish SQLite connection")
        
        try:
            vector_json = json.dumps(vector.tolist())
            
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE memories 
                SET embedding_vector = ?,
                    embedding_model = ?,
                    embedding_created_at = CURRENT_TIMESTAMP,
                    vector_hash = ?
                WHERE id = ?
            """, (vector_json, self.current_model, vector_hash, memory_id))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def load_memory_embeddings(self, memory_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """Load memory embeddings from database."""
        try:
            if db_manager.is_postgresql():
                return await self._load_postgresql_embeddings(memory_ids)
            else:
                return await self._load_sqlite_embeddings(memory_ids)
                
        except Exception as e:
            logger.error(f"Failed to load memory embeddings: {e}")
            return {}
    
    async def _load_postgresql_embeddings(self, memory_ids: Optional[List[int]]) -> Dict[int, np.ndarray]:
        """Load embeddings from PostgreSQL."""
        from mcp_server.database.postgres_database import postgres_db
        
        if memory_ids:
            placeholders = ','.join([f'${i+1}' for i in range(len(memory_ids))])
            query = f"""
                SELECT id, embedding_vector, embedding_model
                FROM memories 
                WHERE id IN ({placeholders}) 
                AND embedding_vector IS NOT NULL
            """
            rows = await postgres_db.execute_query(query, *memory_ids)
        else:
            query = """
                SELECT id, embedding_vector, embedding_model
                FROM memories 
                WHERE embedding_vector IS NOT NULL
            """
            rows = await postgres_db.execute_query(query)
        
        embeddings = {}
        for row in rows:
            try:
                memory_id = row['id']
                vector_data = row['embedding_vector']
                
                # Parse vector data (handle both pgvector and JSON formats)
                if isinstance(vector_data, str):
                    if vector_data.startswith('[') and vector_data.endswith(']'):
                        # pgvector format
                        values = [float(x.strip()) for x in vector_data[1:-1].split(',')]
                        vector = np.array(values)
                    else:
                        # JSON format
                        values = json.loads(vector_data)
                        vector = np.array(values)
                elif isinstance(vector_data, list):
                    # Direct list format
                    vector = np.array(vector_data)
                else:
                    continue  # Skip invalid format
                
                embeddings[memory_id] = vector
                
            except Exception as e:
                logger.warning(f"Could not parse embedding for memory {row['id']}: {e}")
                continue
        
        return embeddings
    
    async def _load_sqlite_embeddings(self, memory_ids: Optional[List[int]]) -> Dict[int, np.ndarray]:
        """Load embeddings from SQLite."""
        conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
        if not conn:
            from mcp_server.database.memory_database import get_db_connection
            conn = get_db_connection()
        
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            if memory_ids:
                placeholders = ','.join(['?' for _ in memory_ids])
                query = f"""
                    SELECT id, embedding_vector, embedding_model
                    FROM memories 
                    WHERE id IN ({placeholders}) 
                    AND embedding_vector IS NOT NULL
                """
                cursor.execute(query, memory_ids)
            else:
                query = """
                    SELECT id, embedding_vector, embedding_model
                    FROM memories 
                    WHERE embedding_vector IS NOT NULL
                """
                cursor.execute(query)
            
            embeddings = {}
            for row in cursor.fetchall():
                try:
                    memory_id = row[0]
                    vector_data = row[1]
                    
                    if vector_data:
                        values = json.loads(vector_data)
                        vector = np.array(values)
                        embeddings[memory_id] = vector
                        
                except Exception as e:
                    logger.warning(f"Could not parse embedding for memory {memory_id}: {e}")
                    continue
            
            return embeddings
            
        finally:
            conn.close()
    
    async def similarity_search(self, query_text: str, session_id: Optional[str] = None,
                              top_k: int = 10, similarity_threshold: Optional[float] = None) -> List[VectorSearchResult]:
        """Perform similarity search using vector embeddings."""
        try:
            # Generate query embedding
            query_vector = await self.generate_embedding(query_text)
            
            # Load memory embeddings (could be optimized with caching)
            memory_embeddings = await self.load_memory_embeddings()
            
            if not memory_embeddings:
                logger.warning("No memory embeddings found")
                return []
            
            # Calculate similarities
            similarities = []
            for memory_id, memory_vector in memory_embeddings.items():
                similarity = self.calculate_cosine_similarity(query_vector, memory_vector)
                similarities.append((memory_id, similarity))
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Apply threshold
            if similarity_threshold is None:
                similarity_threshold = self.similarity_threshold
            
            filtered_similarities = [(mid, score) for mid, score in similarities 
                                   if score >= similarity_threshold]
            
            # Limit results
            top_similarities = filtered_similarities[:top_k]
            
            if not top_similarities:
                return []
            
            # Fetch memory details
            memory_ids = [mid for mid, _ in top_similarities]
            memory_details = await self._get_memory_details(memory_ids, session_id)
            
            # Create results
            results = []
            similarity_dict = dict(top_similarities)
            
            for memory in memory_details:
                memory_id = memory['id']
                if memory_id in similarity_dict:
                    results.append(VectorSearchResult(
                        memory_id=memory_id,
                        content=memory['content'],
                        entity_name=memory.get('entity_name', ''),
                        similarity_score=similarity_dict[memory_id],
                        embedding_vector=memory_embeddings.get(memory_id),
                        metadata=memory.get('metadata', {}),
                        emotional_salience=memory.get('emotional_salience', 0.5),
                        session_id=memory.get('session_id', ''),
                        timestamp=memory.get('created_at', datetime.now())
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def _get_memory_details(self, memory_ids: List[int], 
                                session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get detailed memory information for given IDs."""
        if not memory_ids:
            return []
        
        try:
            if db_manager.is_postgresql():
                from mcp_server.database.postgres_database import postgres_db
                
                placeholders = ','.join([f'${i+1}' for i in range(len(memory_ids))])
                query = f"""
                    SELECT m.*, e.name as entity_name
                    FROM memories m
                    LEFT JOIN entities e ON m.entity_id = e.id
                    WHERE m.id IN ({placeholders})
                """
                params = memory_ids
                
                if session_id:
                    query += f" AND m.session_id = ${len(memory_ids) + 1}"
                    params = memory_ids + [session_id]
                
                return await postgres_db.execute_query(query, *params)
                
            else:
                conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
                if not conn:
                    from mcp_server.database.memory_database import get_db_connection
                    conn = get_db_connection()
                
                if not conn:
                    return []
                
                try:
                    placeholders = ','.join(['?' for _ in memory_ids])
                    query = f"""
                        SELECT m.*, e.name as entity_name
                        FROM memories m
                        LEFT JOIN entities e ON m.entity_id = e.id
                        WHERE m.id IN ({placeholders})
                    """
                    params = memory_ids
                    
                    if session_id:
                        query += " AND m.session_id = ?"
                        params = memory_ids + [session_id]
                    
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    
                    return [dict(zip(columns, row)) for row in rows]
                    
                finally:
                    conn.close()
                    
        except Exception as e:
            logger.error(f"Failed to get memory details: {e}")
            return []
    
    async def batch_generate_embeddings(self, memory_contents: List[Tuple[int, str]]) -> Dict[int, bool]:
        """Generate embeddings for multiple memories in batch."""
        results = {}
        
        for memory_id, content in memory_contents:
            try:
                success = await self.store_memory_embedding(memory_id, content)
                results[memory_id] = success
                
                # Add small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for memory {memory_id}: {e}")
                results[memory_id] = False
        
        return results
    
    async def rebuild_embeddings(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Rebuild all embeddings for memories."""
        try:
            # Get memories that need embeddings
            if db_manager.is_postgresql():
                from mcp_server.database.postgres_database import postgres_db
                
                query = """
                    SELECT id, content 
                    FROM memories 
                    WHERE (embedding_vector IS NULL OR embedding_model != $1)
                """
                params = [self.current_model]
                
                if session_id:
                    query += " AND session_id = $2"
                    params.append(session_id)
                
                rows = await postgres_db.execute_query(query, *params)
                
            else:
                conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
                if not conn:
                    from mcp_server.database.memory_database import get_db_connection
                    conn = get_db_connection()
                
                if not conn:
                    return {"error": "Could not establish database connection"}
                
                try:
                    query = """
                        SELECT id, content 
                        FROM memories 
                        WHERE (embedding_vector IS NULL OR embedding_model != ?)
                    """
                    params = [self.current_model]
                    
                    if session_id:
                        query += " AND session_id = ?"
                        params.append(session_id)
                    
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    
                    columns = [desc[0] for desc in cursor.description]
                    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    
                finally:
                    conn.close()
            
            # Generate embeddings in batches
            memory_contents = [(row['id'], row['content']) for row in rows]
            
            if not memory_contents:
                return {
                    "message": "No memories need embedding generation",
                    "processed": 0,
                    "successful": 0,
                    "failed": 0
                }
            
            # Process in batches
            batch_size = self.models[self.current_model].batch_size
            total_processed = 0
            total_successful = 0
            total_failed = 0
            
            for i in range(0, len(memory_contents), batch_size):
                batch = memory_contents[i:i + batch_size]
                batch_results = await self.batch_generate_embeddings(batch)
                
                total_processed += len(batch)
                total_successful += sum(batch_results.values())
                total_failed += len(batch) - sum(batch_results.values())
                
                logger.info(f"Processed embedding batch {i//batch_size + 1}: "
                          f"{sum(batch_results.values())}/{len(batch)} successful")
            
            return {
                "message": "Embedding generation completed",
                "processed": total_processed,
                "successful": total_successful,
                "failed": total_failed,
                "model": self.current_model
            }
            
        except Exception as e:
            logger.error(f"Failed to rebuild embeddings: {e}")
            return {"error": str(e)}
    
    def set_model(self, model_name: str) -> bool:
        """Set the current embedding model."""
        if model_name in self.models:
            self.current_model = model_name
            logger.info(f"Embedding model set to: {model_name}")
            return True
        else:
            logger.error(f"Unknown embedding model: {model_name}")
            return False
    
    def get_available_models(self) -> Dict[str, EmbeddingModel]:
        """Get list of available embedding models."""
        return self.models.copy()
    
    def get_vector_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored vectors."""
        return {
            "cached_vectors": len(self.memory_vectors),
            "cached_embeddings": len(self.embedding_cache),
            "current_model": self.current_model,
            "available_models": list(self.models.keys()),
            "cache_size_limit": self.vector_cache_size,
            "similarity_threshold": self.similarity_threshold
        }

# Global vector search engine instance
vector_engine = VectorSimilarityEngine()

async def get_vector_engine() -> VectorSimilarityEngine:
    """Get the vector search engine instance."""
    return vector_engine