"""
Semantic Encoding Protocol (SEP)

Lightweight transformer-based semantic encoding for the SIM-ONE Framework.
Provides superior embeddings while maintaining energy efficiency and architectural purity.

Author: SIM-ONE Framework / Manus AI Enhancement
"""

import logging
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path

from .base_protocol import BaseProtocol
from .embedding_cache import EmbeddingCache
from .encoder_models import EncoderModelManager

logger = logging.getLogger(__name__)

class SemanticEncodingProtocol(BaseProtocol):
    """
    Semantic Encoding Protocol for SIM-ONE Framework.
    
    Provides lightweight transformer-based embeddings while respecting
    the Five Laws of Cognitive Governance:
    
    - Law 1: Coordination over complexity
    - Law 2: Governed encoding processes  
    - Law 3: Truth-grounded representations
    - Law 4: Energy-efficient operations
    - Law 5: Deterministic outputs
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or self._get_default_config()
        self.protocol_name = "SemanticEncodingProtocol"
        self.version = "1.0.0"
        
        # Initialize components
        self.model_manager = EncoderModelManager(self.config.get('model_config', {}))
        self.cache = EmbeddingCache(
            max_size=self.config.get('cache_size', 10000),
            ttl_hours=self.config.get('cache_ttl_hours', 24)
        )
        
        # Performance tracking
        self.stats = {
            'total_encodings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        # Governance settings
        self.max_text_length = self.config.get('max_text_length', 8192)
        self.min_text_length = self.config.get('min_text_length', 10)
        self.quality_threshold = self.config.get('quality_threshold', 0.1)
        
        logger.info(f"SemanticEncodingProtocol initialized with model: {self.model_manager.current_model}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration respecting energy stewardship."""
        return {
            'model_config': {
                'primary_model': 'all-MiniLM-L6-v2',  # 22MB, fast, efficient
                'fallback_model': 'tfidf',  # Ultra-lightweight fallback
                'batch_size': 32,
                'max_sequence_length': 512
            },
            'cache_size': 10000,
            'cache_ttl_hours': 24,
            'max_text_length': 8192,
            'min_text_length': 10,
            'quality_threshold': 0.1,
            'energy_optimization': True,
            'deterministic_mode': True
        }
    
    async def initialize(self) -> bool:
        """Initialize the protocol and load models."""
        try:
            logger.info("Initializing Semantic Encoding Protocol...")
            
            # Initialize model manager
            await self.model_manager.initialize()
            
            # Initialize cache
            await self.cache.initialize()
            
            # Validate configuration
            if not self._validate_configuration():
                raise ValueError("Invalid protocol configuration")
            
            logger.info("Semantic Encoding Protocol initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SemanticEncodingProtocol: {e}")
            return False
    
    async def encode_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate semantic embedding for text with governance validation.
        
        Args:
            text: Input text to encode
            context: Optional context for encoding optimization
            
        Returns:
            Dict containing embedding vector, metadata, and quality metrics
        """
        start_time = datetime.now()
        
        try:
            # Law 2: Cognitive Governance - validate input
            validation_result = self._validate_input(text, context)
            if not validation_result['valid']:
                return self._create_error_response(validation_result['error'])
            
            # Normalize text for consistency (Law 5: Deterministic Reliability)
            normalized_text = self._normalize_text(text)
            
            # Check cache first (Law 4: Energy Stewardship)
            cache_key = self._generate_cache_key(normalized_text, context)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return self._add_metadata(cached_result, from_cache=True)
            
            # Generate embedding
            self.stats['cache_misses'] += 1
            embedding_result = await self._generate_embedding(normalized_text, context)
            
            # Law 3: Truth Foundation - validate embedding quality
            quality_score = self._assess_embedding_quality(embedding_result['vector'], normalized_text)
            
            if quality_score < self.quality_threshold:
                logger.warning(f"Low quality embedding detected: {quality_score}")
                # Fallback to alternative model or method
                embedding_result = await self._fallback_encoding(normalized_text, context)
            
            # Cache the result
            await self.cache.set(cache_key, embedding_result)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time)
            
            # Add metadata and return
            return self._add_metadata(embedding_result, processing_time=processing_time, quality_score=quality_score)
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return self._create_error_response(str(e))
    
    async def encode_batch(self, texts: List[str], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Encode multiple texts efficiently with batch processing.
        
        Args:
            texts: List of texts to encode
            context: Optional context for batch optimization
            
        Returns:
            List of encoding results
        """
        logger.info(f"Batch encoding {len(texts)} texts")
        
        # Validate batch size for energy efficiency
        batch_size = self.config['model_config']['batch_size']
        if len(texts) > batch_size:
            # Process in chunks
            results = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                chunk_results = await self._process_batch_chunk(chunk, context)
                results.extend(chunk_results)
            return results
        else:
            return await self._process_batch_chunk(texts, context)
    
    async def _process_batch_chunk(self, texts: List[str], context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a chunk of texts for batch encoding."""
        # Check cache for all texts first
        cache_results = {}
        uncached_texts = []
        
        for i, text in enumerate(texts):
            normalized_text = self._normalize_text(text)
            cache_key = self._generate_cache_key(normalized_text, context)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                cache_results[i] = cached_result
                self.stats['cache_hits'] += 1
            else:
                uncached_texts.append((i, normalized_text))
                self.stats['cache_misses'] += 1
        
        # Batch encode uncached texts
        batch_results = {}
        if uncached_texts:
            indices, texts_to_encode = zip(*uncached_texts)
            embeddings = await self.model_manager.encode_batch(list(texts_to_encode))
            
            for idx, (original_idx, text) in enumerate(uncached_texts):
                embedding_result = {
                    'vector': embeddings[idx],
                    'text': text,
                    'model': self.model_manager.current_model,
                    'dimensions': len(embeddings[idx])
                }
                
                # Cache the result
                cache_key = self._generate_cache_key(text, context)
                await self.cache.set(cache_key, embedding_result)
                
                batch_results[original_idx] = embedding_result
        
        # Combine cached and newly generated results
        final_results = []
        for i in range(len(texts)):
            if i in cache_results:
                result = cache_results[i]
                result = self._add_metadata(result, from_cache=True)
            else:
                result = batch_results[i]
                result = self._add_metadata(result, from_cache=False)
            
            final_results.append(result)
        
        return final_results
    
    def _validate_input(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate input text according to governance rules."""
        if not isinstance(text, str):
            return {'valid': False, 'error': 'Input must be a string'}
        
        if len(text.strip()) < self.min_text_length:
            return {'valid': False, 'error': f'Text too short (minimum {self.min_text_length} characters)'}
        
        if len(text) > self.max_text_length:
            return {'valid': False, 'error': f'Text too long (maximum {self.max_text_length} characters)'}
        
        # Check for potentially problematic content
        if self._contains_problematic_content(text):
            return {'valid': False, 'error': 'Text contains problematic content'}
        
        return {'valid': True}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent encoding (Law 5: Deterministic Reliability)."""
        # Basic normalization for consistency
        normalized = text.strip()
        
        # Remove excessive whitespace
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Ensure consistent encoding
        normalized = normalized.encode('utf-8', errors='ignore').decode('utf-8')
        
        return normalized
    
    def _generate_cache_key(self, text: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate deterministic cache key."""
        # Include model name and relevant context in key
        key_components = [
            text,
            self.model_manager.current_model,
            json.dumps(context or {}, sort_keys=True)
        ]
        
        combined = '|'.join(key_components)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    async def _generate_embedding(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embedding using current model."""
        try:
            vector = await self.model_manager.encode_single(text)
            
            return {
                'vector': vector,
                'text': text,
                'model': self.model_manager.current_model,
                'dimensions': len(vector),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _assess_embedding_quality(self, vector: np.ndarray, text: str) -> float:
        """Assess the quality of generated embedding (Law 3: Truth Foundation)."""
        # Basic quality metrics
        
        # Check for zero or near-zero vectors
        vector_norm = np.linalg.norm(vector)
        if vector_norm < 1e-6:
            return 0.0
        
        # Check for reasonable variance
        vector_std = np.std(vector)
        if vector_std < 1e-6:
            return 0.1  # Very low variance indicates poor embedding
        
        # Check for NaN or infinite values
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            return 0.0
        
        # Text-based quality indicators
        text_quality = self._assess_text_quality(text)
        
        # Combine metrics (simple weighted average)
        quality_score = (
            min(vector_norm / 10.0, 1.0) * 0.3 +  # Normalized norm
            min(vector_std * 10.0, 1.0) * 0.3 +   # Normalized std
            text_quality * 0.4                     # Text quality
        )
        
        return min(quality_score, 1.0)
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess input text quality."""
        # Basic text quality metrics
        word_count = len(text.split())
        char_count = len(text)
        
        # Reasonable length
        length_score = min(word_count / 50.0, 1.0) if word_count > 0 else 0.0
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        diversity_score = min(unique_chars / 20.0, 1.0)
        
        # Not just repeated characters
        repetition_penalty = 1.0
        if char_count > 0:
            most_common_char_count = max([text.count(c) for c in set(text)])
            repetition_ratio = most_common_char_count / char_count
            if repetition_ratio > 0.5:
                repetition_penalty = 0.5
        
        return (length_score + diversity_score) * 0.5 * repetition_penalty
    
    async def _fallback_encoding(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback encoding method for low-quality primary results."""
        logger.info("Using fallback encoding method")
        
        try:
            # Switch to fallback model
            original_model = self.model_manager.current_model
            await self.model_manager.switch_model('tfidf')
            
            # Generate fallback embedding
            vector = await self.model_manager.encode_single(text)
            
            result = {
                'vector': vector,
                'text': text,
                'model': 'tfidf_fallback',
                'dimensions': len(vector),
                'timestamp': datetime.now().isoformat(),
                'fallback': True
            }
            
            # Switch back to original model
            await self.model_manager.switch_model(original_model)
            
            return result
            
        except Exception as e:
            logger.error(f"Fallback encoding failed: {e}")
            raise
    
    def _contains_problematic_content(self, text: str) -> bool:
        """Check for potentially problematic content."""
        # Basic content validation
        # This could be expanded based on specific requirements
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.5:
            return True
        
        # Check for binary or encoded content
        try:
            text.encode('ascii')
        except UnicodeEncodeError:
            # Contains non-ASCII characters - not necessarily problematic
            pass
        
        return False
    
    def _add_metadata(self, result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Add metadata to encoding result."""
        metadata = {
            'protocol': self.protocol_name,
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'from_cache': kwargs.get('from_cache', False),
            'processing_time': kwargs.get('processing_time'),
            'quality_score': kwargs.get('quality_score'),
            'stats': self.get_stats()
        }
        
        result['metadata'] = metadata
        return result
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'error': True,
            'message': error_message,
            'vector': None,
            'metadata': {
                'protocol': self.protocol_name,
                'version': self.version,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _update_stats(self, processing_time: float):
        """Update performance statistics."""
        self.stats['total_encodings'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['average_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['total_encodings']
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        cache_hit_rate = 0.0
        if self.stats['total_encodings'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (
                self.stats['cache_hits'] + self.stats['cache_misses']
            )
        
        return {
            'total_encodings': self.stats['total_encodings'],
            'cache_hit_rate': cache_hit_rate,
            'average_processing_time': self.stats['average_processing_time'],
            'current_model': self.model_manager.current_model,
            'cache_size': await self.cache.size(),
            'energy_efficiency_score': self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate energy efficiency score (Law 4: Energy Stewardship)."""
        # Higher cache hit rate = better efficiency
        cache_hit_rate = 0.0
        if self.stats['total_encodings'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (
                self.stats['cache_hits'] + self.stats['cache_misses']
            )
        
        # Faster processing = better efficiency
        speed_score = 1.0
        if self.stats['average_processing_time'] > 0:
            # Normalize to 0-1 scale (assuming 1 second is baseline)
            speed_score = min(1.0 / self.stats['average_processing_time'], 1.0)
        
        # Combine metrics
        efficiency_score = (cache_hit_rate * 0.6) + (speed_score * 0.4)
        return efficiency_score
    
    def _validate_configuration(self) -> bool:
        """Validate protocol configuration."""
        required_keys = ['model_config', 'cache_size', 'max_text_length']
        
        for key in required_keys:
            if key not in self.config:
                logger.error(f"Missing required configuration key: {key}")
                return False
        
        return True
    
    async def shutdown(self):
        """Clean shutdown of the protocol."""
        logger.info("Shutting down Semantic Encoding Protocol...")
        
        try:
            await self.cache.shutdown()
            await self.model_manager.shutdown()
            logger.info("Semantic Encoding Protocol shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Protocol registration for SIM-ONE framework
def create_protocol(config: Optional[Dict[str, Any]] = None) -> SemanticEncodingProtocol:
    """Factory function for creating SemanticEncodingProtocol instances."""
    return SemanticEncodingProtocol(config)

