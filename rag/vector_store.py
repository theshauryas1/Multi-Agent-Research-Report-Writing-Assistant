"""
Vector Store - FAISS-based vector database for RAG retrieval.
Stores document embeddings and performs similarity search.
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from .document_processor import DocumentChunk
from .embeddings import EmbeddingModel, get_embedding_model


class VectorStore:
    """
    FAISS-powered vector store for document retrieval.
    
    Features:
    - Efficient similarity search
    - Persistent storage
    - Metadata filtering
    """
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        index_path: Optional[str] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: Embedding model to use
            index_path: Path to load/save the index
        """
        self.embedding_model = embedding_model or get_embedding_model()
        self.index_path = index_path
        
        self._index = None
        self._documents: List[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        
        # Load existing index if path provided
        if index_path and Path(index_path).exists():
            self.load(index_path)
    
    @property
    def index(self):
        """Lazy load FAISS index."""
        if self._index is None:
            try:
                import faiss
                
                # Create empty index
                dimension = self.embedding_model.dimension
                self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)
                
            except ImportError:
                raise ImportError(
                    "faiss is required. Install with: pip install faiss-cpu"
                )
        return self._index
    
    def add_documents(
        self, 
        documents: List[DocumentChunk],
        batch_size: int = 32,
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of DocumentChunk objects
            batch_size: Batch size for embedding
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Extract text for embedding
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.embed_documents(texts, batch_size=batch_size)
        
        # Add to FAISS index
        import faiss
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents
        self._documents.extend(documents)
        
        # Update stored embeddings
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
        
        print(f"✓ Added {len(documents)} documents to vector store")
        return len(documents)
    
    def add_texts(
        self,
        texts: List[str],
        source: str = "manual",
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Add raw texts to the vector store.
        
        Args:
            texts: List of text strings
            source: Source identifier
            metadata: Optional metadata for all texts
            
        Returns:
            Number of texts added
        """
        documents = [
            DocumentChunk(
                content=text,
                source=source,
                chunk_id=i,
                metadata=metadata or {},
            )
            for i, text in enumerate(texts)
        ]
        return self.add_documents(documents)
    
    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of (document, score) tuples
        """
        if len(self._documents) == 0:
            return []
        
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        k = min(top_k, len(self._documents))
        scores, indices = self.index.search(query_embedding, k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= score_threshold:
                results.append((self._documents[idx], float(score)))
        
        return results
    
    def get_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get relevant chunks as dictionaries.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of chunk dictionaries with scores
        """
        results = self.similarity_search(query, top_k=top_k)
        
        return [
            {
                **doc.to_dict(),
                "relevance_score": score,
            }
            for doc, score in results
        ]
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Save path (uses index_path if not provided)
        """
        import faiss
        
        save_path = Path(path or self.index_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save documents
        with open(save_path / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)
        
        # Save metadata
        metadata = {
            "num_documents": len(self._documents),
            "embedding_model": self.embedding_model.model_name,
            "dimension": self.embedding_model.dimension,
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved vector store to {save_path}")
    
    def load(self, path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            path: Load path
        """
        import faiss
        
        load_path = Path(path)
        
        # Load FAISS index
        self._index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load documents
        with open(load_path / "documents.pkl", "rb") as f:
            self._documents = pickle.load(f)
        
        print(f"✓ Loaded vector store from {load_path} ({len(self._documents)} documents)")
    
    def __len__(self) -> int:
        """Return number of documents."""
        return len(self._documents)
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self._index = None
        self._documents = []
        self._embeddings = None


def create_store(
    documents_path: Optional[str] = None,
    index_path: Optional[str] = None,
) -> VectorStore:
    """
    Create and optionally populate a vector store.
    
    Args:
        documents_path: Path to documents directory
        index_path: Path to save/load index
        
    Returns:
        VectorStore instance
    """
    from .document_processor import DocumentProcessor
    
    store = VectorStore(index_path=index_path)
    
    if documents_path and Path(documents_path).exists():
        processor = DocumentProcessor()
        chunks = processor.process_directory(documents_path)
        store.add_documents(chunks)
    
    return store


def similarity_search(
    query: str,
    store: VectorStore,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Convenience function for similarity search.
    
    Args:
        query: Search query
        store: VectorStore instance
        top_k: Number of results
        
    Returns:
        List of relevant chunks
    """
    return store.get_relevant_chunks(query, top_k=top_k)


if __name__ == "__main__":
    # Test vector store
    print("Testing vector store...")
    
    store = VectorStore()
    
    # Add some test documents
    test_texts = [
        "Renewable energy sources include solar, wind, and hydroelectric power.",
        "Solar panels convert sunlight directly into electricity using photovoltaic cells.",
        "Wind turbines harness the kinetic energy of moving air to generate power.",
        "Climate change is driven by greenhouse gas emissions from fossil fuels.",
        "Machine learning models can predict energy consumption patterns.",
    ]
    
    store.add_texts(test_texts, source="test")
    
    # Test search
    query = "How does solar energy work?"
    print(f"\nSearching for: '{query}'")
    
    results = store.get_relevant_chunks(query, top_k=3)
    
    print("\nTop results:")
    for r in results:
        print(f"  [{r['relevance_score']:.3f}] {r['content'][:80]}...")
