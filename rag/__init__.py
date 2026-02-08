"""
RAG (Retrieval-Augmented Generation) Module.

Provides vector-based document retrieval for grounding generated content.
"""

from .vector_store import VectorStore, create_store, similarity_search
from .document_processor import DocumentProcessor, chunk_document
from .embeddings import EmbeddingModel, get_embeddings

__all__ = [
    "VectorStore",
    "create_store", 
    "similarity_search",
    "DocumentProcessor",
    "chunk_document",
    "EmbeddingModel",
    "get_embeddings",
]
