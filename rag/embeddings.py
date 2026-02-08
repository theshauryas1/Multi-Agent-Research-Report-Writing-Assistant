"""
Embedding Model - Generates vector embeddings for text chunks.
Uses sentence-transformers for high-quality semantic embeddings.
"""

import os
from typing import List, Union, Optional
import numpy as np

# Configuration
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
ALTERNATIVE_MODEL = "all-mpnet-base-v2"  # Better quality, 768 dimensions


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.
    
    Provides efficient batch embedding with caching support.
    """
    
    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None for auto-detect
            cache_dir: Directory for model cache
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._dimension = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_dir,
                )
                self._dimension = self._model.get_sentence_embedding_dimension()
                print(f"✓ Loaded embedding model: {self.model_name} ({self._dimension}D)")
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy load
        return self._dimension
    
    def embed(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (N, D)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed(query)[0]
    
    def embed_documents(
        self, 
        documents: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Embed multiple documents."""
        return self.embed(documents, batch_size=batch_size, show_progress=True)


# Global singleton for efficiency
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> EmbeddingModel:
    """Get or create the global embedding model instance."""
    global _embedding_model
    
    if _embedding_model is None or _embedding_model.model_name != model_name:
        _embedding_model = EmbeddingModel(model_name=model_name)
    
    return _embedding_model


def get_embeddings(
    texts: Union[str, List[str]], 
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """
    Convenience function to get embeddings.
    
    Args:
        texts: Text(s) to embed
        model_name: Embedding model to use
        
    Returns:
        Numpy array of embeddings
    """
    model = get_embedding_model(model_name)
    return model.embed(texts)


if __name__ == "__main__":
    # Test embeddings
    print("Testing embedding model...")
    
    model = EmbeddingModel()
    
    texts = [
        "Renewable energy is the future of power generation.",
        "Solar panels convert sunlight into electricity.",
        "Machine learning models can analyze large datasets.",
    ]
    
    embeddings = model.embed(texts)
    print(f"Generated {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
    
    # Test similarity
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    
    print(f"\nSimilarity scores:")
    print(f"  Text 1 vs 2 (related): {cosine_similarity(embeddings[0], embeddings[1]):.3f}")
    print(f"  Text 1 vs 3 (unrelated): {cosine_similarity(embeddings[0], embeddings[2]):.3f}")
