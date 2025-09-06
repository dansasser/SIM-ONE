"""
Encoder Model Manager for Semantic Encoding Protocol

Manages lightweight transformer encoders while respecting SIM-ONE's
energy stewardship and architectural intelligence principles.

Author: SIM-ONE Framework / Manus AI Enhancement
"""

import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class EncoderModelManager:
    """
    Manages multiple encoder models with automatic fallback and optimization.
    
    Supports:
    - Lightweight transformer models (sentence-transformers)
    - TF-IDF fallback for energy efficiency
    - Model switching and optimization
    - Batch processing for efficiency
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.current_model = config.get('primary_model', 'all-MiniLM-L6-v2')
        self.fallback_model = config.get('fallback_model', 'tfidf')
        
        # Model configurations
        self.model_configs = {
            'all-MiniLM-L6-v2': {
                'type': 'sentence_transformer',
                'size_mb': 22,
                'dimensions': 384,
                'max_sequence_length': 256,
                'speed_score': 9,  # 1-10 scale
                'quality_score': 7
            },
            'all-distilroberta-v1': {
                'type': 'sentence_transformer',
                'size_mb': 82,
                'dimensions': 768,
                'max_sequence_length': 512,
                'speed_score': 6,
                'quality_score': 8
            },
            'tfidf': {
                'type': 'tfidf',
                'size_mb': 1,
                'dimensions': 1000,
                'max_sequence_length': 10000,
                'speed_score': 10,
                'quality_score': 4
            },
            'word2vec_lite': {
                'type': 'word2vec',
                'size_mb': 5,
                'dimensions': 100,
                'max_sequence_length': 1000,
                'speed_score': 8,
                'quality_score': 5
            }
        }
        
        # Performance tracking
        self.stats = {
            'model_usage': {},
            'encoding_times': {},
            'error_counts': {},
            'total_encodings': 0
        }
        
        # Batch processing settings
        self.batch_size = config.get('batch_size', 32)
        self.max_sequence_length = config.get('max_sequence_length', 512)
        
        logger.info(f"EncoderModelManager initialized with primary model: {self.current_model}")
    
    async def initialize(self):
        """Initialize the model manager and load primary model."""
        try:
            logger.info("Initializing EncoderModelManager...")
            
            # Load primary model
            await self._load_model(self.current_model)
            
            # Initialize fallback model
            await self._load_model(self.fallback_model)
            
            # Initialize statistics
            for model_name in self.model_configs:
                self.stats['model_usage'][model_name] = 0
                self.stats['encoding_times'][model_name] = []
                self.stats['error_counts'][model_name] = 0
            
            logger.info("EncoderModelManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EncoderModelManager: {e}")
            raise
    
    async def _load_model(self, model_name: str):
        """Load a specific encoder model."""
        if model_name in self.models:
            logger.debug(f"Model {model_name} already loaded")
            return
        
        try:
            model_config = self.model_configs.get(model_name)
            if not model_config:
                raise ValueError(f"Unknown model: {model_name}")
            
            logger.info(f"Loading model: {model_name} ({model_config['size_mb']}MB)")
            
            if model_config['type'] == 'sentence_transformer':
                model = await self._load_sentence_transformer(model_name)
            elif model_config['type'] == 'tfidf':
                model = await self._load_tfidf_model()
            elif model_config['type'] == 'word2vec':
                model = await self._load_word2vec_model()
            else:
                raise ValueError(f"Unsupported model type: {model_config['type']}")
            
            self.models[model_name] = {
                'model': model,
                'config': model_config,
                'loaded_at': datetime.now()
            }
            
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    async def _load_sentence_transformer(self, model_name: str):
        """Load sentence transformer model."""
        try:
            # Import sentence-transformers (install if needed)
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
            
            # Load model (will download if not cached)
            model = SentenceTransformer(model_name)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer {model_name}: {e}")
            raise
    
    async def _load_tfidf_model(self):
        """Load TF-IDF model (lightweight fallback)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create TF-IDF vectorizer with reasonable parameters
        model = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        
        return model
    
    async def _load_word2vec_model(self):
        """Load simplified Word2Vec model."""
        # This is a placeholder for a lightweight word2vec implementation
        # In practice, you might use gensim or a custom implementation
        
        class SimpleWord2Vec:
            def __init__(self):
                self.dimensions = 100
                self.word_vectors = {}
                self.fitted = False
            
            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                
                vectors = []
                for text in texts:
                    # Simple averaging of word vectors (placeholder)
                    words = text.lower().split()
                    if not words:
                        vectors.append(np.zeros(self.dimensions))
                        continue
                    
                    # Generate simple hash-based vectors for words
                    word_vecs = []
                    for word in words:
                        hash_val = hashlib.md5(word.encode()).hexdigest()
                        vec = np.array([int(hash_val[i:i+2], 16) for i in range(0, min(32, len(hash_val)), 2)])
                        
                        # Pad or truncate to desired dimensions
                        if len(vec) < self.dimensions:
                            vec = np.pad(vec, (0, self.dimensions - len(vec)))
                        else:
                            vec = vec[:self.dimensions]
                        
                        word_vecs.append(vec)
                    
                    # Average word vectors
                    if word_vecs:
                        avg_vec = np.mean(word_vecs, axis=0)
                        # Normalize
                        norm = np.linalg.norm(avg_vec)
                        if norm > 0:
                            avg_vec = avg_vec / norm
                        vectors.append(avg_vec)
                    else:
                        vectors.append(np.zeros(self.dimensions))
                
                return np.array(vectors)
        
        return SimpleWord2Vec()
    
    async def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text using the current model."""
        start_time = datetime.now()
        
        try:
            model_info = self.models[self.current_model]
            model = model_info['model']
            config = model_info['config']
            
            # Truncate text if too long
            if len(text) > config['max_sequence_length']:
                text = text[:config['max_sequence_length']]
            
            # Encode based on model type
            if config['type'] == 'sentence_transformer':
                vector = model.encode([text])[0]
            elif config['type'] == 'tfidf':
                # TF-IDF requires fitting first, use simple approach
                vector = await self._encode_tfidf_single(text, model)
            elif config['type'] == 'word2vec':
                vector = model.encode([text])[0]
            else:
                raise ValueError(f"Unsupported model type: {config['type']}")
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_model_stats(self.current_model, processing_time)
            
            return vector.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error encoding text with {self.current_model}: {e}")
            self.stats['error_counts'][self.current_model] += 1
            
            # Try fallback model
            if self.current_model != self.fallback_model:
                logger.info(f"Falling back to {self.fallback_model}")
                original_model = self.current_model
                self.current_model = self.fallback_model
                try:
                    result = await self.encode_single(text)
                    self.current_model = original_model
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback encoding also failed: {fallback_error}")
                    self.current_model = original_model
            
            raise
    
    async def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts efficiently."""
        if not texts:
            return []
        
        start_time = datetime.now()
        
        try:
            model_info = self.models[self.current_model]
            model = model_info['model']
            config = model_info['config']
            
            # Truncate texts if too long
            truncated_texts = []
            for text in texts:
                if len(text) > config['max_sequence_length']:
                    text = text[:config['max_sequence_length']]
                truncated_texts.append(text)
            
            # Process in batches if necessary
            all_vectors = []
            batch_size = min(self.batch_size, len(truncated_texts))
            
            for i in range(0, len(truncated_texts), batch_size):
                batch = truncated_texts[i:i + batch_size]
                
                # Encode based on model type
                if config['type'] == 'sentence_transformer':
                    batch_vectors = model.encode(batch)
                elif config['type'] == 'tfidf':
                    batch_vectors = await self._encode_tfidf_batch(batch, model)
                elif config['type'] == 'word2vec':
                    batch_vectors = model.encode(batch)
                else:
                    raise ValueError(f"Unsupported model type: {config['type']}")
                
                all_vectors.extend(batch_vectors)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_model_stats(self.current_model, processing_time, len(texts))
            
            return [vec.astype(np.float32) for vec in all_vectors]
            
        except Exception as e:
            logger.error(f"Error batch encoding with {self.current_model}: {e}")
            self.stats['error_counts'][self.current_model] += 1
            raise
    
    async def _encode_tfidf_single(self, text: str, model) -> np.ndarray:
        """Encode single text with TF-IDF (simplified approach)."""
        # For single text, create a simple TF vector
        from collections import Counter
        import re
        
        # Simple tokenization
        words = re.findall(r'\\b\\w+\\b', text.lower())
        word_counts = Counter(words)
        
        # Create basic vocabulary
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'memory', 'entity', 'emotion', 'cognitive', 'protocol', 'system',
            'ai', 'learning', 'data', 'information', 'process', 'analysis'
        ]
        
        vocabulary = {word: i for i, word in enumerate(common_words + list(set(words)))}
        
        # Create TF vector
        dimensions = min(1000, len(vocabulary))
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
    
    async def _encode_tfidf_batch(self, texts: List[str], model) -> List[np.ndarray]:
        """Encode batch of texts with TF-IDF."""
        vectors = []
        for text in texts:
            vector = await self._encode_tfidf_single(text, model)
            vectors.append(vector)
        return vectors
    
    async def switch_model(self, model_name: str):
        """Switch to a different model."""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        if model_name not in self.models:
            await self._load_model(model_name)
        
        old_model = self.current_model
        self.current_model = model_name
        
        logger.info(f"Switched from {old_model} to {model_name}")
    
    def _update_model_stats(self, model_name: str, processing_time: float, batch_size: int = 1):
        """Update model performance statistics."""
        self.stats['model_usage'][model_name] += batch_size
        self.stats['encoding_times'][model_name].append(processing_time)
        self.stats['total_encodings'] += batch_size
        
        # Keep only recent timing data (last 1000 encodings per model)
        if len(self.stats['encoding_times'][model_name]) > 1000:
            self.stats['encoding_times'][model_name] = self.stats['encoding_times'][model_name][-1000:]
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        if model_name not in self.stats['model_usage']:
            return {}
        
        times = self.stats['encoding_times'][model_name]
        config = self.model_configs[model_name]
        
        avg_time = np.mean(times) if times else 0.0
        error_rate = self.stats['error_counts'][model_name] / max(self.stats['model_usage'][model_name], 1)
        
        # Calculate efficiency score
        speed_score = config['speed_score']
        quality_score = config['quality_score']
        reliability_score = max(0, 10 - (error_rate * 10))
        
        efficiency_score = (speed_score * 0.4 + quality_score * 0.4 + reliability_score * 0.2) / 10.0
        
        return {
            'model_name': model_name,
            'usage_count': self.stats['model_usage'][model_name],
            'average_time': avg_time,
            'error_rate': error_rate,
            'efficiency_score': efficiency_score,
            'size_mb': config['size_mb'],
            'dimensions': config['dimensions'],
            'speed_score': speed_score,
            'quality_score': quality_score
        }
    
    def get_all_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all models."""
        return {
            model_name: self.get_model_performance(model_name)
            for model_name in self.model_configs
        }
    
    def recommend_optimal_model(self, priority: str = 'balanced') -> str:
        """
        Recommend optimal model based on priority.
        
        Args:
            priority: 'speed', 'quality', 'efficiency', or 'balanced'
        """
        performances = self.get_all_model_performance()
        
        if not performances:
            return self.current_model
        
        if priority == 'speed':
            # Prioritize speed score
            best_model = max(performances.items(), 
                           key=lambda x: x[1].get('speed_score', 0) if x[1] else 0)
        elif priority == 'quality':
            # Prioritize quality score
            best_model = max(performances.items(),
                           key=lambda x: x[1].get('quality_score', 0) if x[1] else 0)
        elif priority == 'efficiency':
            # Prioritize efficiency score
            best_model = max(performances.items(),
                           key=lambda x: x[1].get('efficiency_score', 0) if x[1] else 0)
        else:  # balanced
            # Balance speed, quality, and efficiency
            def balanced_score(perf):
                if not perf:
                    return 0
                return (perf.get('speed_score', 0) * 0.3 + 
                       perf.get('quality_score', 0) * 0.4 + 
                       perf.get('efficiency_score', 0) * 10 * 0.3)
            
            best_model = max(performances.items(), key=lambda x: balanced_score(x[1]))
        
        return best_model[0]
    
    async def optimize_model_selection(self):
        """Automatically optimize model selection based on performance."""
        try:
            recommended_model = self.recommend_optimal_model('efficiency')
            
            if recommended_model != self.current_model:
                logger.info(f"Optimizing: switching to {recommended_model} for better efficiency")
                await self.switch_model(recommended_model)
                
        except Exception as e:
            logger.error(f"Error during model optimization: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall manager statistics."""
        return {
            'current_model': self.current_model,
            'total_encodings': self.stats['total_encodings'],
            'loaded_models': list(self.models.keys()),
            'model_performances': self.get_all_model_performance(),
            'recommended_model': self.recommend_optimal_model()
        }
    
    async def shutdown(self):
        """Clean shutdown of model manager."""
        try:
            logger.info("Shutting down EncoderModelManager...")
            
            # Clear models from memory
            self.models.clear()
            
            logger.info("EncoderModelManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during model manager shutdown: {e}")

