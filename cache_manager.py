"""
Cache management module for production deployment.
Handles caching of embeddings, search results, and regulations.
"""
from typing import Dict, Any, Optional, List
import hashlib
import json
from functools import lru_cache


class CacheManager:
    """
    Manages caching for improved performance.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
        """
        self.max_size = max_size
        self._embedding_cache: Dict[str, List[float]] = {}
        self._search_cache: Dict[str, List[Any]] = {}
        self._regulation_cache: Dict[str, str] = {}
    
    def _generate_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding.
        
        Args:
            text: Text to look up
        
        Returns:
            Cached embedding or None
        """
        key = self._generate_key(text)
        return self._embedding_cache.get(key)
    
    def set_embedding(self, text: str, embedding: List[float]):
        """
        Cache an embedding.
        
        Args:
            text: Text
            embedding: Embedding vector
        """
        if len(self._embedding_cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self._embedding_cache))
            del self._embedding_cache[first_key]
        
        key = self._generate_key(text)
        self._embedding_cache[key] = embedding
    
    def get_search_result(self, query: str) -> Optional[List[Any]]:
        """
        Get cached search result.
        
        Args:
            query: Search query
        
        Returns:
            Cached results or None
        """
        key = self._generate_key(query)
        return self._search_cache.get(key)
    
    def set_search_result(self, query: str, results: List[Any]):
        """
        Cache search results.
        
        Args:
            query: Search query
            results: Search results
        """
        if len(self._search_cache) >= self.max_size:
            first_key = next(iter(self._search_cache))
            del self._search_cache[first_key]
        
        key = self._generate_key(query)
        self._search_cache[key] = results
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache.
        
        Args:
            cache_type: Type of cache to clear ('embedding', 'search', or None for all)
        """
        if cache_type == "embedding":
            self._embedding_cache.clear()
        elif cache_type == "search":
            self._search_cache.clear()
        elif cache_type == "regulation":
            self._regulation_cache.clear()
        else:
            self._embedding_cache.clear()
            self._search_cache.clear()
            self._regulation_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            "embedding_cache_size": len(self._embedding_cache),
            "search_cache_size": len(self._search_cache),
            "regulation_cache_size": len(self._regulation_cache),
            "max_size": self.max_size
        }


# Global instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
