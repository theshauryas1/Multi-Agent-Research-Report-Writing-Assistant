"""
Document Processor - Parses and chunks documents for RAG.
Supports PDF, TXT, and Markdown files with semantic chunking.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentChunk:
    """A chunk of text with metadata for retrieval."""
    content: str
    source: str
    chunk_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields
    page: Optional[int] = None
    section: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "section": self.section,
            **self.metadata,
        }


class DocumentProcessor:
    """
    Processes documents into chunks suitable for RAG.
    
    Features:
    - Semantic chunking with sentence boundaries
    - Configurable chunk size and overlap
    - Metadata preservation
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a single file into chunks.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of DocumentChunk objects
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read content based on file type
        extension = path.suffix.lower()
        
        if extension == ".pdf":
            content, metadata = self._read_pdf(path)
        elif extension in [".txt", ".md", ".markdown"]:
            content, metadata = self._read_text(path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        # Chunk the content
        chunks = self._chunk_text(content, str(path), metadata)
        
        return chunks
    
    def process_directory(
        self, 
        directory: str,
        extensions: List[str] = [".pdf", ".txt", ".md"],
    ) -> List[DocumentChunk]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Directory path
            extensions: File extensions to process
            
        Returns:
            List of all DocumentChunk objects
        """
        all_chunks = []
        dir_path = Path(directory)
        
        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                try:
                    chunks = self.process_file(str(file_path))
                    all_chunks.extend(chunks)
                    print(f"✓ Processed: {file_path.name} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"✗ Failed to process {file_path.name}: {e}")
        
        return all_chunks
    
    def _read_pdf(self, path: Path) -> Tuple[str, Dict]:
        """Read PDF file content."""
        try:
            import pypdf
            
            reader = pypdf.PdfReader(str(path))
            
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text_parts.append(f"[Page {i+1}]\n{page_text}")
            
            content = "\n\n".join(text_parts)
            metadata = {
                "file_type": "pdf",
                "page_count": len(reader.pages),
                "title": reader.metadata.title if reader.metadata else None,
                "author": reader.metadata.author if reader.metadata else None,
            }
            
            return content, metadata
            
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with: pip install pypdf")
    
    def _read_text(self, path: Path) -> Tuple[str, Dict]:
        """Read text/markdown file content."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        metadata = {
            "file_type": path.suffix[1:],  # Remove the dot
        }
        
        return content, metadata
    
    def _chunk_text(
        self, 
        text: str, 
        source: str,
        metadata: Dict,
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Uses sentence boundaries for cleaner splits.
        """
        # Clean the text
        text = self._clean_text(text)
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_chunk)
                
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        source=source,
                        chunk_id=chunk_id,
                        metadata=metadata.copy(),
                    ))
                    chunk_id += 1
                
                # Keep overlap
                overlap_length = 0
                overlap_sentences = []
                
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    metadata=metadata.copy(),
                ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove weird characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    text: str, 
    source: str = "unknown",
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[DocumentChunk]:
    """
    Convenience function to chunk a document.
    
    Args:
        text: Document text
        source: Source identifier
        chunk_size: Target chunk size
        overlap: Chunk overlap
        
    Returns:
        List of DocumentChunk objects
    """
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return processor._chunk_text(text, source, {})


if __name__ == "__main__":
    # Test the document processor
    print("Testing document processor...")
    
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)
    
    sample_text = """
    Renewable energy sources are becoming increasingly important in the global 
    effort to combat climate change. Solar power, wind energy, and hydroelectric 
    power are among the most common forms of renewable energy. These sources 
    produce little to no greenhouse gas emissions during operation, making them 
    attractive alternatives to fossil fuels.
    
    Solar energy harnesses the power of sunlight through photovoltaic cells or 
    solar thermal systems. Wind energy captures the kinetic energy of moving air 
    using turbines. Hydroelectric power generates electricity from flowing or 
    falling water.
    
    The adoption of renewable energy has accelerated in recent years due to 
    falling costs and improved technology. Many countries have set ambitious 
    targets to increase the share of renewable energy in their power mix.
    """
    
    chunks = chunk_document(sample_text, "test_document.txt")
    
    print(f"\nGenerated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\n--- Chunk {chunk.chunk_id} ({len(chunk.content)} chars) ---")
        print(chunk.content[:150] + "...")
