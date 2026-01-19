"""
Retrieval module for production deployment.
Handles semantic search and hybrid search strategies.
"""
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
import re

from vector_db import get_vector_database


class RetrievalSystem:
    """
    Handles retrieval of relevant regulations using semantic search.
    """
    
    def __init__(self, vector_db=None):
        """
        Initialize retrieval system.
        
        Args:
            vector_db: VectorDatabase instance (optional)
        """
        self.vector_db = vector_db or get_vector_database()
    
    def semantic_search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform semantic search for relevant regulations.
        
        Args:
            query: Search query
            k: Number of results
            min_score: Minimum similarity score
            filter_dict: Optional metadata filters
        
        Returns:
            List of relevant Document objects
        """
        if not query or not query.strip():
            return []
        
        # Perform semantic search
        results_with_scores = self.vector_db.search_with_scores(
            query,
            k=k * 2,  # Get more results for filtering
            filter_dict=filter_dict
        )
        
        # Filter by minimum score and limit to k
        filtered_results = [
            doc for doc, score in results_with_scores
            if score >= min_score
        ][:k]
        
        return filtered_results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform hybrid search (semantic + keyword).
        
        Args:
            query: Search query
            k: Number of results
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            filter_dict: Optional filters
        
        Returns:
            List of relevant Document objects
        """
        # Semantic search
        semantic_results = self.semantic_search(query, k=k * 2, filter_dict=filter_dict)
        
        # Keyword search (simple implementation)
        keyword_results = self._keyword_search(query, filter_dict=filter_dict)
        
        # Combine and rank
        combined = self._merge_results(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        return combined[:k]
    
    def _keyword_search(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Simple keyword-based search.
        
        Args:
            query: Search query
            filter_dict: Optional filters
        
        Returns:
            List of Document objects
        """
        # Extract keywords
        keywords = re.findall(r'\b\w+\b', query.lower())
        
        # Get all documents (this is simplified - in production, use proper indexing)
        # For now, use semantic search as fallback
        return self.semantic_search(query, k=10, filter_dict=filter_dict)
    
    def _merge_results(
        self,
        semantic_results: List[Document],
        keyword_results: List[Document],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Document]:
        """
        Merge and rank results from different search methods.
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic results
            keyword_weight: Weight for keyword results
        
        Returns:
            Merged and ranked results
        """
        # Create score dictionary
        doc_scores = {}
        
        # Add semantic results
        for idx, doc in enumerate(semantic_results):
            doc_id = id(doc)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + semantic_weight * (1.0 - idx / len(semantic_results))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + semantic_weight
        
        # Add keyword results
        for idx, doc in enumerate(keyword_results):
            doc_id = id(doc)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + keyword_weight * (1.0 - idx / len(keyword_results))
        
        # Create document map
        doc_map = {}
        for doc in semantic_results + keyword_results:
            doc_map[id(doc)] = doc
        
        # Sort by score
        sorted_docs = sorted(
            doc_map.items(),
            key=lambda x: doc_scores.get(x[0], 0),
            reverse=True
        )
        
        return [doc for _, doc in sorted_docs]
    
    def retrieve_relevant_regulations(
        self,
        user_document: str,
        k: int = 5,
        use_hybrid: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Retrieve relevant regulations for a user document.
        
        Args:
            user_document: User's document text
            k: Number of regulations to retrieve
            use_hybrid: Whether to use hybrid search
        
        Returns:
            List of (label, text) tuples
        """
        if not user_document or not user_document.strip():
            return []
        
        # Extract key phrases from user document for search
        # Use first 500 chars as query (or full doc if shorter)
        query = user_document[:500] if len(user_document) > 500 else user_document
        
        # Perform search
        if use_hybrid:
            results = self.hybrid_search(query, k=k)
        else:
            results = self.semantic_search(query, k=k)
        
        # Convert to (label, text) format
        regulations = []
        seen_sources = set()
        
        for doc in results:
            source = doc.metadata.get("source", "Unknown")
            
            # Avoid duplicates from same source
            if source in seen_sources:
                continue
            
            seen_sources.add(source)
            regulations.append((source, doc.page_content))
        
        return regulations


# Global instance
_retrieval_system: Optional[RetrievalSystem] = None


def get_retrieval_system() -> RetrievalSystem:
    """Get or create global retrieval system instance."""
    global _retrieval_system
    if _retrieval_system is None:
        _retrieval_system = RetrievalSystem()
    return _retrieval_system
