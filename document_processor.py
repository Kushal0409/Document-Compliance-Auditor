"""
Document processing module for production deployment.
Handles text extraction, chunking, and preprocessing.
"""
import os
from typing import List, Dict, Any, Optional, Tuple

# Try different import paths for langchain compatibility
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain_text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError(
                "langchain-text-splitters not installed. Run: pip install langchain-text-splitters"
            )

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        raise ImportError(
            "langchain-core not installed. Run: pip install langchain-core"
        )

from doc_utils import extract_text_from_path


class DocumentProcessor:
    """
    Processes documents for vector database storage.
    Handles chunking, metadata extraction, and preprocessing.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            separators: Text separators for splitting
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
    
    def process_file(
        self,
        file_path: str,
        source_label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process a file into chunks with metadata.
        
        Args:
            file_path: Path to the file
            source_label: Label for the source document
            metadata: Additional metadata to attach
        
        Returns:
            List of Document objects with chunks and metadata
        """
        # Extract text
        try:
            text = extract_text_from_path(file_path, show_warning=False)
        except Exception as e:
            print(f"⚠️  Warning: Failed to extract text from {file_path}: {e}")
            return []
        
        if not text or not text.strip():
            return []
        
        # Use filename as source if not provided
        if source_label is None:
            source_label = os.path.basename(file_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        for idx, chunk_text in enumerate(chunks):
            doc_metadata = {
                "source": source_label,
                "source_file": file_path,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }
            
            # Add custom metadata
            if metadata:
                doc_metadata.update(metadata)
            
            documents.append(Document(
                page_content=chunk_text,
                metadata=doc_metadata
            ))
        
        return documents
    
    def process_text(
        self,
        text: str,
        source_label: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process raw text into chunks with metadata.
        
        Args:
            text: Raw text to process
            source_label: Label for the source
            metadata: Additional metadata
        
        Returns:
            List of Document objects
        """
        if not text or not text.strip():
            return []
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects
        documents = []
        for idx, chunk_text in enumerate(chunks):
            doc_metadata = {
                "source": source_label,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            documents.append(Document(
                page_content=chunk_text,
                metadata=doc_metadata
            ))
        
        return documents
    
    def process_multiple_files(
        self,
        file_paths: List[str],
        source_labels: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Process multiple files.
        
        Args:
            file_paths: List of file paths
            source_labels: Optional list of labels (one per file)
        
        Returns:
            Combined list of all Document objects
        """
        all_documents = []
        
        for idx, file_path in enumerate(file_paths):
            label = source_labels[idx] if source_labels and idx < len(source_labels) else None
            documents = self.process_file(file_path, source_label=label)
            all_documents.extend(documents)
        
        return all_documents
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text (clean, normalize).
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters that might interfere
        # (Keep basic punctuation)
        
        return text.strip()


# Global instance
_document_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """Get or create global document processor instance."""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor
