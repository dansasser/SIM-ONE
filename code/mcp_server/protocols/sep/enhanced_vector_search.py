"""
Enhanced Vector Search Integration with Semantic Encoding Protocol

Integrates SEP with the existing VectorSimilarityEngine to provide
semantic embeddings while maintaining backward compatibility.

Author: SIM-ONE Framework / Manus AI Enhancement
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from mcp_server.database.vector_search import VectorSimilarityEngine, VectorSearchResult
from .semantic_encoding_protocol import SemanticEncodingProtocol

logger = logging.getLogger(__name__)

class EnhancedVectorSimilarityEngine(VectorSimilarityEngine):
    """
    Enhanced vector search engine with SEP integration.
    
    Maintains backward compatibility while providing superior
    semantic understanding through the Semantic Encoding Protocol.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize SEP
        self.sep = None
        self.sep_enabled = False
        self.fallback_to_original = True
        
        # Enhanced configuration
        self.semantic_similarity_weight = 0.7  # Weight for semantic similarity
        self.original_similarity_weight = 0.3  # Weight for original similarity methods
        
        # Performance tracking
        self.enhanced_stats = {
            'sep_encodings': 0,
            'semantic_searches': 0,
            'fallback_uses': 0,
            'quality_improvements': 0
        }
        
        logger.info("EnhancedVectorSimilarityEngine initialized")
    
    async def initialize_sep(self, sep_config: Optional[Dict[str, Any]] = None):
        """Initialize the Semantic Encoding Protocol."""
        try:
            logger.info("Initializing Semantic Encoding Protocol...")
            
            self.sep = SemanticEncodingProtocol(sep_config)
            success = await self.sep.initialize()
            
            if success:
                self.sep_enabled = True
                logger.info("SEP integration enabled successfully")
            else:
                logger.warning("SEP initialization failed, using fallback methods")
                self.sep_enabled = False
                
        except Exception as e:
            logger.error(f"Failed to initialize SEP: {e}")
            self.sep_enabled = False
    
    async def generate_embedding(self, text: str, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate embedding using SEP or fallback to original methods.
        
        Args:
            text: Text to encode
            context: Optional context for encoding
            
        Returns:
            Embedding vector
        """
        try:
            if self.sep_enabled and self.sep:
                # Use SEP for semantic encoding
                result = await self.sep.encode_text(text, context)
                
                if not result.get('error'):
                    self.enhanced_stats['sep_encodings'] += 1
                    return result['vector']
                else:
                    logger.warning(f"SEP encoding failed: {result.get('message')}")
                    
            # Fallback to original methods
            if self.fallback_to_original:
                self.enhanced_stats['fallback_uses'] += 1
                
                if self.current_model == 'mock':
                    return self._generate_mock_embedding(text)
                elif self.current_model == 'tfidf':
                    return self._generate_tfidf_embedding(text)
                else:
                    # Default to mock
                    return self._generate_mock_embedding(text)
            else:
                raise RuntimeError("SEP encoding failed and fallback disabled")
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            
            # Last resort fallback
            if self.fallback_to_original:
                return self._generate_mock_embedding(text)
            else:
                raise
    
    async def semantic_similarity_search(
        self, 
        query_text: str, 
        session_id: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform semantic similarity search with enhanced capabilities.
        
        Args:
            query_text: Query text to search for
            session_id: Session identifier
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            context: Optional search context
            
        Returns:
            List of search results ranked by semantic similarity
        """
        try:
            self.enhanced_stats['semantic_searches'] += 1
            
            # Generate query embedding
            query_embedding = await self.generate_embedding(query_text, context)
            
            # Get all memories for the session
            memories = await self._get_session_memories(session_id)
            
            if not memories:
                return []
            
            # Calculate similarities
            similarities = []
            
            for memory in memories:
                try:
                    # Get or generate memory embedding
                    memory_embedding = await self._get_memory_embedding(memory, context)
                    
                    if memory_embedding is not None:
                        # Calculate semantic similarity
                        semantic_sim = self._calculate_cosine_similarity(query_embedding, memory_embedding)
                        
                        # Combine with other factors if available
                        final_score = self._calculate_enhanced_similarity_score(
                            semantic_sim, memory, query_text, context
                        )
                        
                        if final_score >= similarity_threshold:
                            similarities.append({
                                'memory': memory,
                                'similarity': final_score,
                                'semantic_similarity': semantic_sim
                            })
                            
                except Exception as e:
                    logger.warning(f"Error processing memory {memory.get('id', 'unknown')}: {e}")
                    continue
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Convert to VectorSearchResult objects
            results = []
            for item in similarities[:top_k]:
                memory = item['memory']
                
                result = VectorSearchResult(
                    memory_id=memory.get('id', 0),
                    content=memory.get('content', ''),
                    entity_name=memory.get('entity_name', ''),
                    similarity_score=item['similarity'],
                    embedding_vector=query_embedding,
                    metadata={
                        'semantic_similarity': item['semantic_similarity'],
                        'search_method': 'enhanced_semantic',
                        'sep_enabled': self.sep_enabled
                    },
                    emotional_salience=memory.get('emotional_salience', 0.5),
                    session_id=session_id,
                    timestamp=datetime.now()
                )
                
                results.append(result)
            
            # Track quality improvements
            if len(results) > 0:
                self.enhanced_stats['quality_improvements'] += 1
            
            logger.debug(f"Semantic search returned {len(results)} results for query: {query_text[:50]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic similarity search: {e}")
            
            # Fallback to original search if available
            if hasattr(super(), 'similarity_search'):
                logger.info("Falling back to original similarity search")
                return await super().similarity_search(query_text, session_id, top_k, similarity_threshold)
            else:
                return []
    
    async def _get_memory_embedding(self, memory: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Get or generate embedding for a memory."""
        try:
            # Check if memory already has an embedding
            if 'embedding_vector' in memory and memory['embedding_vector'] is not None:
                # Convert from stored format if necessary
                embedding = memory['embedding_vector']
                if isinstance(embedding, list):
                    return np.array(embedding, dtype=np.float32)
                elif isinstance(embedding, np.ndarray):
                    return embedding.astype(np.float32)
            
            # Generate new embedding
            content = memory.get('content', '')
            if content:
                embedding = await self.generate_embedding(content, context)
                
                # Store embedding for future use (optional optimization)
                await self._store_memory_embedding(memory.get('id'), embedding)
                
                return embedding
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting memory embedding: {e}")
            return None
    
    async def _store_memory_embedding(self, memory_id: int, embedding: np.ndarray):
        """Store embedding for a memory (optional optimization)."""
        try:
            # This would integrate with the existing database storage
            # Implementation depends on the current database schema
            pass
        except Exception as e:
            logger.warning(f"Error storing memory embedding: {e}")
    
    def _calculate_enhanced_similarity_score(
        self, 
        semantic_similarity: float, 
        memory: Dict[str, Any], 
        query_text: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate enhanced similarity score combining semantic and other factors.
        
        Args:
            semantic_similarity: Semantic similarity score (0-1)
            memory: Memory object with metadata
            query_text: Original query text
            context: Search context
            
        Returns:
            Enhanced similarity score
        """
        try:
            # Start with semantic similarity
            score = semantic_similarity * self.semantic_similarity_weight
            
            # Add keyword matching bonus (original method)
            keyword_similarity = self._calculate_keyword_similarity(query_text, memory.get('content', ''))
            score += keyword_similarity * self.original_similarity_weight
            
            # Emotional salience boost
            emotional_salience = memory.get('emotional_salience', 0.5)
            salience_boost = 1 + (emotional_salience * 0.2)  # Up to 20% boost
            score *= salience_boost
            
            # Rehearsal count boost
            rehearsal_count = memory.get('rehearsal_count', 0)
            rehearsal_boost = min(rehearsal_count * 0.05, 0.3)  # Up to 30% boost
            score += rehearsal_boost
            
            # Recency boost
            if memory.get('last_accessed'):
                # Simple recency boost (could be more sophisticated)
                score += 0.1
            
            # Actor relevance boost
            if context and 'actors' in context:
                memory_actors = memory.get('actors', [])
                if any(actor in memory_actors for actor in context['actors']):
                    score += 0.2  # 20% boost for actor relevance
            
            # Ensure score stays in reasonable range
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating enhanced similarity score: {e}")
            return semantic_similarity
    
    def _calculate_keyword_similarity(self, query: str, content: str) -> float:
        """Calculate simple keyword-based similarity."""
        try:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words:
                return 0.0
            
            intersection = query_words.intersection(content_words)
            return len(intersection) / len(query_words)
            
        except Exception as e:
            logger.warning(f"Error calculating keyword similarity: {e}")
            return 0.0
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Ensure vectors are the same length
            if len(vec1) != len(vec2):
                logger.warning(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in valid range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def _get_session_memories(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a session."""
        try:
            # This would integrate with the existing memory manager
            # For now, return empty list as placeholder
            # In real implementation, this would call the memory manager
            return []
            
        except Exception as e:
            logger.error(f"Error getting session memories: {e}")
            return []
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including SEP performance."""
        base_stats = self.get_stats() if hasattr(super(), 'get_stats') else {}
        
        enhanced_stats = {
            'sep_enabled': self.sep_enabled,
            'sep_stats': self.sep.get_stats() if self.sep else {},
            'enhanced_stats': self.enhanced_stats,
            'similarity_weights': {
                'semantic': self.semantic_similarity_weight,
                'original': self.original_similarity_weight
            }
        }
        
        return {**base_stats, **enhanced_stats}
    
    async def optimize_performance(self):
        """Optimize performance based on usage patterns."""
        try:
            if self.sep and self.sep_enabled:
                # Get SEP performance stats
                sep_stats = self.sep.get_stats()
                
                # Optimize model selection if needed
                if hasattr(self.sep.model_manager, 'optimize_model_selection'):
                    await self.sep.model_manager.optimize_model_selection()
                
                logger.info("Performance optimization completed")
                
        except Exception as e:
            logger.error(f"Error during performance optimization: {e}")
    
    async def shutdown(self):
        """Clean shutdown of enhanced vector search."""
        try:
            logger.info("Shutting down EnhancedVectorSimilarityEngine...")
            
            if self.sep:
                await self.sep.shutdown()
            
            # Call parent shutdown if available
            if hasattr(super(), 'shutdown'):
                await super().shutdown()
            
            logger.info("EnhancedVectorSimilarityEngine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during enhanced vector search shutdown: {e}")

# Factory function for easy integration
async def create_enhanced_vector_search(sep_config: Optional[Dict[str, Any]] = None) -> EnhancedVectorSimilarityEngine:
    """
    Create and initialize an enhanced vector search engine.
    
    Args:
        sep_config: Optional SEP configuration
        
    Returns:
        Initialized EnhancedVectorSimilarityEngine
    """
    engine = EnhancedVectorSimilarityEngine()
    
    # Initialize base vector search
    await engine.initialize_vector_storage()
    
    # Initialize SEP integration
    await engine.initialize_sep(sep_config)
    
    return engine

