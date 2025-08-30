import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json
import math

from mcp_server.database.database_manager import db_manager

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with relevance scoring."""
    memory_id: int
    content: str
    entity_name: str
    relevance_score: float
    match_type: str  # 'exact', 'partial', 'semantic', 'fuzzy'
    matched_terms: List[str]
    emotional_salience: float
    session_id: str
    timestamp: datetime

class AdvancedSearchEngine:
    """
    Advanced search engine with full-text search, semantic search capabilities,
    and intelligent query processing for the SIM-ONE memory system.
    """
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'be', 'is', 'are', 'was', 'were',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall'
        }
        self.search_cache = {}  # Simple cache for frequent searches
        self.cache_ttl = 300  # 5 minutes cache TTL
    
    async def initialize_search_indexes(self):
        """Initialize advanced search indexes for both PostgreSQL and SQLite."""
        try:
            if db_manager.is_postgresql():
                await self._initialize_postgresql_search()
            else:
                await self._initialize_sqlite_search()
            
            logger.info("Advanced search indexes initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize search indexes: {e}")
            raise
    
    async def _initialize_postgresql_search(self):
        """Initialize PostgreSQL full-text search capabilities."""
        from mcp_server.database.postgres_database import postgres_db
        
        async with postgres_db.pool.acquire() as conn:
            async with conn.transaction():
                # Create full-text search indexes
                await conn.execute("""
                    -- Add tsvector columns for full-text search
                    ALTER TABLE memories 
                    ADD COLUMN IF NOT EXISTS content_tsv tsvector;
                """)
                
                # Create function to update tsvector
                await conn.execute("""
                    CREATE OR REPLACE FUNCTION update_memory_tsv()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.content_tsv = to_tsvector('english', COALESCE(NEW.content, ''));
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                
                # Create trigger for automatic tsvector updates
                await conn.execute("""
                    DROP TRIGGER IF EXISTS memory_content_tsv_update ON memories;
                    CREATE TRIGGER memory_content_tsv_update
                        BEFORE INSERT OR UPDATE ON memories
                        FOR EACH ROW EXECUTE FUNCTION update_memory_tsv();
                """)
                
                # Update existing records
                await conn.execute("""
                    UPDATE memories 
                    SET content_tsv = to_tsvector('english', COALESCE(content, ''))
                    WHERE content_tsv IS NULL;
                """)
                
                # Create GIN index for full-text search
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_content_tsv 
                    ON memories USING gin(content_tsv);
                """)
                
                # Create advanced search indexes
                advanced_indexes = [
                    # Composite indexes for complex queries
                    "CREATE INDEX IF NOT EXISTS idx_memories_session_salience ON memories(session_id, emotional_salience DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_memories_entity_type ON memories(entity_id, memory_type)",
                    "CREATE INDEX IF NOT EXISTS idx_memories_temporal ON memories(created_at DESC, last_accessed DESC)",
                    
                    # Partial indexes for performance
                    "CREATE INDEX IF NOT EXISTS idx_memories_high_salience ON memories(emotional_salience) WHERE emotional_salience > 0.7",
                    "CREATE INDEX IF NOT EXISTS idx_memories_recent ON memories(created_at) WHERE created_at >= NOW() - INTERVAL '30 days'",
                    
                    # Array search indexes
                    "CREATE INDEX IF NOT EXISTS idx_memories_actors_gin ON memories USING gin(actors)",
                    "CREATE INDEX IF NOT EXISTS idx_memories_context_tags_gin ON memories USING gin(context_tags)",
                    
                    # Metadata search
                    "CREATE INDEX IF NOT EXISTS idx_memories_metadata_gin ON memories USING gin(metadata)",
                ]
                
                for index_sql in advanced_indexes:
                    try:
                        await conn.execute(index_sql)
                    except Exception as e:
                        logger.warning(f"Could not create index: {e}")
    
    async def _initialize_sqlite_search(self):
        """Initialize SQLite FTS (Full-Text Search) capabilities."""
        conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
        if not conn:
            from mcp_server.database.memory_database import get_db_connection
            conn = get_db_connection()
        
        if not conn:
            raise RuntimeError("Could not establish SQLite connection")
        
        try:
            cursor = conn.cursor()
            
            # Create FTS virtual table for content search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    content,
                    entity_name,
                    actors,
                    context_tags,
                    content=memories,
                    content_rowid=id
                );
            """)
            
            # Create triggers to keep FTS table in sync
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_insert 
                AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(memory_id, content, entity_name, actors, context_tags)
                    SELECT NEW.id, NEW.content, 
                           COALESCE((SELECT name FROM entities WHERE id = NEW.entity_id), ''),
                           NEW.actors, NEW.context_tags;
                END;
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_update 
                AFTER UPDATE ON memories BEGIN
                    UPDATE memories_fts 
                    SET content = NEW.content,
                        entity_name = COALESCE((SELECT name FROM entities WHERE id = NEW.entity_id), ''),
                        actors = NEW.actors,
                        context_tags = NEW.context_tags
                    WHERE memory_id = NEW.id;
                END;
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_delete 
                AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE memory_id = OLD.id;
                END;
            """)
            
            # Populate FTS table with existing data
            cursor.execute("""
                INSERT OR REPLACE INTO memories_fts(memory_id, content, entity_name, actors, context_tags)
                SELECT m.id, m.content, 
                       COALESCE(e.name, ''),
                       m.actors, m.context_tags
                FROM memories m
                LEFT JOIN entities e ON m.entity_id = e.id;
            """)
            
            conn.commit()
            logger.info("SQLite FTS indexes created successfully")
            
        finally:
            conn.close()
    
    def _preprocess_query(self, query: str) -> Tuple[str, List[str], List[str]]:
        """Preprocess search query to extract terms and operators."""
        # Remove extra whitespace and convert to lowercase
        clean_query = re.sub(r'\s+', ' ', query.strip().lower())
        
        # Extract quoted phrases
        phrases = re.findall(r'"([^"]*)"', clean_query)
        
        # Remove quoted phrases from query for word extraction
        query_without_phrases = re.sub(r'"[^"]*"', '', clean_query)
        
        # Extract individual words, removing stop words
        words = [word for word in re.findall(r'\b\w+\b', query_without_phrases) 
                 if word not in self.stop_words and len(word) > 2]
        
        return clean_query, words, phrases
    
    async def search_memories(self, query: str, session_id: Optional[str] = None,
                            entity_filter: Optional[str] = None,
                            memory_type_filter: Optional[str] = None,
                            date_range: Optional[Tuple[datetime, datetime]] = None,
                            min_salience: float = 0.0,
                            limit: int = 20,
                            offset: int = 0) -> List[SearchResult]:
        """
        Advanced memory search with multiple search strategies and filters.
        """
        # Check cache first
        cache_key = self._generate_cache_key(query, session_id, entity_filter, 
                                           memory_type_filter, date_range, min_salience, limit, offset)
        
        if cache_key in self.search_cache:
            cached_result, timestamp = self.search_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return cached_result
        
        try:
            if db_manager.is_postgresql():
                results = await self._search_postgresql(query, session_id, entity_filter,
                                                      memory_type_filter, date_range,
                                                      min_salience, limit, offset)
            else:
                results = await self._search_sqlite(query, session_id, entity_filter,
                                                  memory_type_filter, date_range,
                                                  min_salience, limit, offset)
            
            # Cache results
            self.search_cache[cache_key] = (results, datetime.now())
            
            # Clean old cache entries
            if len(self.search_cache) > 1000:
                self._clean_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _search_postgresql(self, query: str, session_id: Optional[str],
                               entity_filter: Optional[str], memory_type_filter: Optional[str],
                               date_range: Optional[Tuple[datetime, datetime]],
                               min_salience: float, limit: int, offset: int) -> List[SearchResult]:
        """PostgreSQL full-text search implementation."""
        from mcp_server.database.postgres_database import postgres_db
        
        clean_query, words, phrases = self._preprocess_query(query)
        
        # Build tsquery for full-text search
        search_terms = []
        
        # Add phrases (exact matches)
        for phrase in phrases:
            if phrase.strip():
                search_terms.append(f"'{phrase.replace(' ', ' & ')}'")
        
        # Add individual words with OR logic
        if words:
            word_query = ' | '.join([f"'{word}'" for word in words])
            search_terms.append(f"({word_query})")
        
        if not search_terms:
            # If no valid search terms, return empty results
            return []
        
        tsquery = ' & '.join(search_terms)
        
        # Build the main query
        base_query = """
            SELECT 
                m.id,
                m.content,
                e.name as entity_name,
                m.emotional_salience,
                m.session_id,
                m.created_at,
                m.memory_type,
                m.actors,
                m.context_tags,
                ts_rank(m.content_tsv, to_tsquery('english', $1)) as rank,
                'full_text' as match_type
            FROM memories m
            JOIN entities e ON m.entity_id = e.id
            WHERE m.content_tsv @@ to_tsquery('english', $1)
        """
        
        params = [tsquery]
        param_count = 1
        
        # Add filters
        if session_id:
            param_count += 1
            base_query += f" AND m.session_id = ${param_count}"
            params.append(session_id)
        
        if entity_filter:
            param_count += 1
            base_query += f" AND e.name ILIKE ${param_count}"
            params.append(f"%{entity_filter}%")
        
        if memory_type_filter:
            param_count += 1
            base_query += f" AND m.memory_type = ${param_count}"
            params.append(memory_type_filter)
        
        if date_range:
            param_count += 1
            base_query += f" AND m.created_at >= ${param_count}"
            params.append(date_range[0])
            param_count += 1
            base_query += f" AND m.created_at <= ${param_count}"
            params.append(date_range[1])
        
        if min_salience > 0:
            param_count += 1
            base_query += f" AND m.emotional_salience >= ${param_count}"
            params.append(min_salience)
        
        # Add ordering and pagination
        base_query += """
            ORDER BY rank DESC, m.emotional_salience DESC, m.created_at DESC
            LIMIT $%d OFFSET $%d
        """ % (param_count + 1, param_count + 2)
        
        params.extend([limit, offset])
        
        try:
            rows = await postgres_db.execute_query(base_query, *params)
            return self._process_search_results(rows, words, phrases)
            
        except Exception as e:
            logger.error(f"PostgreSQL search failed: {e}")
            # Fallback to simple LIKE search
            return await self._fallback_search(query, session_id, entity_filter,
                                             memory_type_filter, date_range,
                                             min_salience, limit, offset)
    
    async def _search_sqlite(self, query: str, session_id: Optional[str],
                           entity_filter: Optional[str], memory_type_filter: Optional[str],
                           date_range: Optional[Tuple[datetime, datetime]],
                           min_salience: float, limit: int, offset: int) -> List[SearchResult]:
        """SQLite FTS search implementation."""
        clean_query, words, phrases = self._preprocess_query(query)
        
        # Build FTS query
        fts_terms = []
        
        # Add phrases
        for phrase in phrases:
            if phrase.strip():
                fts_terms.append(f'"{phrase}"')
        
        # Add words
        for word in words:
            fts_terms.append(word)
        
        if not fts_terms:
            return []
        
        fts_query = ' OR '.join(fts_terms)
        
        # Build main query using FTS
        base_query = """
            SELECT 
                m.id,
                m.content,
                e.name as entity_name,
                m.emotional_salience,
                m.session_id,
                m.timestamp as created_at,
                m.memory_type,
                m.actors,
                m.context_tags,
                fts.rank as rank,
                'full_text' as match_type
            FROM memories_fts fts
            JOIN memories m ON fts.memory_id = m.id
            JOIN entities e ON m.entity_id = e.id
            WHERE memories_fts MATCH ?
        """
        
        params = [fts_query]
        
        # Add filters
        if session_id:
            base_query += " AND m.session_id = ?"
            params.append(session_id)
        
        if entity_filter:
            base_query += " AND e.name LIKE ?"
            params.append(f"%{entity_filter}%")
        
        if memory_type_filter:
            base_query += " AND m.memory_type = ?"
            params.append(memory_type_filter)
        
        if date_range:
            base_query += " AND m.timestamp >= ? AND m.timestamp <= ?"
            params.extend([date_range[0], date_range[1]])
        
        if min_salience > 0:
            base_query += " AND m.emotional_salience >= ?"
            params.append(min_salience)
        
        # Add ordering and pagination
        base_query += """
            ORDER BY fts.rank, m.emotional_salience DESC, m.timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        try:
            conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
            if not conn:
                from mcp_server.database.memory_database import get_db_connection
                conn = get_db_connection()
            
            if not conn:
                raise RuntimeError("Could not establish SQLite connection")
            
            cursor = conn.cursor()
            cursor.execute(base_query, params)
            
            columns = [desc[0] for desc in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return self._process_search_results(rows, words, phrases)
            
        except Exception as e:
            logger.error(f"SQLite FTS search failed: {e}")
            # Fallback to simple LIKE search
            return await self._fallback_search(query, session_id, entity_filter,
                                             memory_type_filter, date_range,
                                             min_salience, limit, offset)
        finally:
            if conn:
                conn.close()
    
    async def _fallback_search(self, query: str, session_id: Optional[str],
                             entity_filter: Optional[str], memory_type_filter: Optional[str],
                             date_range: Optional[Tuple[datetime, datetime]],
                             min_salience: float, limit: int, offset: int) -> List[SearchResult]:
        """Fallback search using simple LIKE queries."""
        clean_query, words, phrases = self._preprocess_query(query)
        
        # Create a simple LIKE-based search
        search_terms = phrases + words
        if not search_terms:
            return []
        
        if db_manager.is_postgresql():
            from mcp_server.database.postgres_database import postgres_db
            
            like_conditions = []
            params = []
            param_count = 0
            
            for term in search_terms:
                param_count += 1
                like_conditions.append(f"m.content ILIKE ${param_count}")
                params.append(f"%{term}%")
            
            base_query = f"""
                SELECT 
                    m.id, m.content, e.name as entity_name, m.emotional_salience,
                    m.session_id, m.created_at, m.memory_type, m.actors, m.context_tags,
                    0.5 as rank, 'fallback' as match_type
                FROM memories m
                JOIN entities e ON m.entity_id = e.id
                WHERE ({' OR '.join(like_conditions)})
            """
            
            # Add filters (similar to above)
            if session_id:
                param_count += 1
                base_query += f" AND m.session_id = ${param_count}"
                params.append(session_id)
            
            base_query += f" ORDER BY m.emotional_salience DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
            params.extend([limit, offset])
            
            rows = await postgres_db.execute_query(base_query, *params)
            
        else:
            # SQLite fallback
            conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
            if not conn:
                from mcp_server.database.memory_database import get_db_connection
                conn = get_db_connection()
            
            if not conn:
                return []
            
            try:
                like_conditions = []
                params = []
                
                for term in search_terms:
                    like_conditions.append("m.content LIKE ?")
                    params.append(f"%{term}%")
                
                base_query = f"""
                    SELECT 
                        m.id, m.content, e.name as entity_name, m.emotional_salience,
                        m.session_id, m.timestamp as created_at, m.memory_type, m.actors, m.context_tags,
                        0.5 as rank, 'fallback' as match_type
                    FROM memories m
                    JOIN entities e ON m.entity_id = e.id
                    WHERE ({' OR '.join(like_conditions)})
                """
                
                if session_id:
                    base_query += " AND m.session_id = ?"
                    params.append(session_id)
                
                base_query += " ORDER BY m.emotional_salience DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor = conn.cursor()
                cursor.execute(base_query, params)
                
                columns = [desc[0] for desc in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
            finally:
                conn.close()
        
        return self._process_search_results(rows, words, phrases)
    
    def _process_search_results(self, rows: List[Dict], words: List[str], phrases: List[str]) -> List[SearchResult]:
        """Process raw database results into SearchResult objects."""
        results = []
        
        for row in rows:
            # Determine matched terms
            content_lower = row['content'].lower()
            matched_terms = []
            
            for phrase in phrases:
                if phrase in content_lower:
                    matched_terms.append(phrase)
            
            for word in words:
                if word in content_lower:
                    matched_terms.append(word)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                row, matched_terms, words, phrases
            )
            
            # Convert timestamp if needed
            created_at = row['created_at']
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = datetime.now()
            
            results.append(SearchResult(
                memory_id=row['id'],
                content=row['content'],
                entity_name=row['entity_name'],
                relevance_score=relevance_score,
                match_type=row.get('match_type', 'unknown'),
                matched_terms=matched_terms,
                emotional_salience=row.get('emotional_salience', 0.5),
                session_id=row.get('session_id', ''),
                timestamp=created_at
            ))
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _calculate_relevance_score(self, row: Dict, matched_terms: List[str], 
                                 words: List[str], phrases: List[str]) -> float:
        """Calculate relevance score for a search result."""
        base_score = row.get('rank', 0.5)
        
        # Boost for phrase matches (more important than word matches)
        phrase_boost = len([term for term in matched_terms if term in phrases]) * 0.3
        
        # Boost for word matches
        word_boost = len([term for term in matched_terms if term in words]) * 0.1
        
        # Boost for emotional salience
        salience_boost = row.get('emotional_salience', 0.5) * 0.2
        
        # Boost for recent memories (recency bias)
        recency_boost = 0.1  # Simplified - could calculate based on actual age
        
        total_score = base_score + phrase_boost + word_boost + salience_boost + recency_boost
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, total_score))
    
    def _generate_cache_key(self, query: str, session_id: Optional[str],
                          entity_filter: Optional[str], memory_type_filter: Optional[str],
                          date_range: Optional[Tuple[datetime, datetime]],
                          min_salience: float, limit: int, offset: int) -> str:
        """Generate cache key for search parameters."""
        key_parts = [
            query.lower().strip(),
            str(session_id or ''),
            str(entity_filter or ''),
            str(memory_type_filter or ''),
            str(date_range) if date_range else '',
            str(min_salience),
            str(limit),
            str(offset)
        ]
        return '|'.join(key_parts)
    
    def _clean_cache(self):
        """Clean old cache entries."""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, (_, timestamp) in self.search_cache.items():
            if (current_time - timestamp).total_seconds() > self.cache_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.search_cache[key]
    
    async def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions based on partial query."""
        if len(partial_query) < 2:
            return []
        
        try:
            if db_manager.is_postgresql():
                from mcp_server.database.postgres_database import postgres_db
                
                # Get common terms from memory content
                query = """
                    SELECT DISTINCT unnest(string_to_array(lower(content), ' ')) as term
                    FROM memories
                    WHERE lower(content) LIKE $1
                    AND length(unnest(string_to_array(lower(content), ' '))) > 2
                    ORDER BY term
                    LIMIT $2
                """
                
                rows = await postgres_db.execute_query(query, f"%{partial_query.lower()}%", limit)
                suggestions = [row['term'] for row in rows if partial_query.lower() in row['term']]
                
            else:
                # SQLite suggestions
                conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
                if not conn:
                    from mcp_server.database.memory_database import get_db_connection
                    conn = get_db_connection()
                
                if not conn:
                    return []
                
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT DISTINCT content
                        FROM memories
                        WHERE lower(content) LIKE ?
                        LIMIT ?
                    """, (f"%{partial_query.lower()}%", limit * 5))
                    
                    # Extract words from content
                    suggestions = set()
                    for row in cursor.fetchall():
                        words = re.findall(r'\b\w+\b', row[0].lower())
                        for word in words:
                            if partial_query.lower() in word and len(word) > 2:
                                suggestions.add(word)
                                if len(suggestions) >= limit:
                                    break
                        if len(suggestions) >= limit:
                            break
                    
                    suggestions = list(suggestions)[:limit]
                    
                finally:
                    conn.close()
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []

# Global search engine instance
search_engine = AdvancedSearchEngine()

async def get_search_engine() -> AdvancedSearchEngine:
    """Get the search engine instance."""
    return search_engine