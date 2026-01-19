"""
Vector database module for production deployment.
Uses ChromaDB for storing and retrieving regulation embeddings.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain.vectorstores import Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            Chroma = None

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        Document = None

from embeddings import get_embedding_generator


class VectorDatabase:
    """
    Manages vector database operations for regulation storage and retrieval.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "regulations"
    ):
        """
        Initialize vector database.
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
        """
        if chromadb is None:
            raise RuntimeError(
                "chromadb not installed. Run: pip install chromadb"
            )
        
        if Chroma is None:
            raise RuntimeError(
                "langchain.vectorstores.Chroma not available. Run: pip install langchain langchain-community"
            )
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embedding_generator = get_embedding_generator()
        
        # Initialize ChromaDB
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize ChromaDB with embeddings."""
        try:
            # Create vector store
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embedding_generator.embeddings
            )
        except Exception as e:
            print(f"⚠️  Warning: Failed to load existing database: {e}")
            # Create new database
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embedding_generator.embeddings
            )
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of Document objects
            batch_size: Number of documents to add at once
        
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        ids = []
        
        # Add in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                batch_ids = self.vector_store.add_documents(batch)
                ids.extend(batch_ids)
            except Exception as e:
                print(f"⚠️  Warning: Failed to add batch {i//batch_size + 1}: {e}")
        
        # Persist changes
        self.vector_store.persist()
        
        return ids
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents using semantic search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
        
        Returns:
            List of similar Document objects
        """
        if not query or not query.strip():
            return []
        
        try:
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            return results
        except Exception as e:
            print(f"⚠️  Warning: Search failed: {e}")
            return []
    
    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results
            filter_dict: Optional filters
        
        Returns:
            List of (Document, score) tuples
        """
        if not query or not query.strip():
            return []
        
        try:
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            return results
        except Exception as e:
            print(f"⚠️  Warning: Search with scores failed: {e}")
            return []
    
    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete documents from the database.
        
        Args:
            ids: List of document IDs to delete
            filter_dict: Optional metadata filters
        
        Returns:
            True if successful
        """
        try:
            if ids:
                self.vector_store.delete(ids=ids)
            elif filter_dict:
                # ChromaDB doesn't support filter-based delete directly
                # Need to find IDs first
                all_docs = self.vector_store.get()
                # This is a simplified version - full implementation would filter
                pass
            
            self.vector_store.persist()
            return True
        except Exception as e:
            print(f"⚠️  Warning: Delete failed: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {
                "error": str(e),
                "collection_name": self.collection_name
            }
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful
        """
        try:
            # Delete the collection and recreate
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._initialize_database()
            return True
        except Exception as e:
            print(f"⚠️  Warning: Clear collection failed: {e}")
            return False


# Global instance
_vector_db: Optional[VectorDatabase] = None


def get_vector_database() -> VectorDatabase:
    """Get or create global vector database instance."""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDatabase()
    return _vector_db
