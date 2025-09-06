"""
Semantic Encoding Protocol (SEP)

A lightweight protocol for generating semantic embeddings within the SIM-ONE Framework.
Maintains architectural purity by keeping encoding separate from the MVLM while providing
superior semantic understanding for RAG operations.

This protocol embodies the Five Laws:
- Law 1: Intelligence through coordination, not model complexity
- Law 2: Governed encoding with quality validation
- Law 3: Truth-grounded semantic representations
- Law 4: Energy-efficient lightweight transformers
- Law 5: Deterministic, reproducible embeddings
"""

from .semantic_encoding_protocol import SemanticEncodingProtocol
from .embedding_cache import EmbeddingCache
from .encoder_models import EncoderModelManager

__all__ = [
    'SemanticEncodingProtocol',
    'EmbeddingCache', 
    'EncoderModelManager'
]

