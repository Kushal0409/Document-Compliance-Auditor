"""
Embedding generation module for production deployment.
Uses Google's embedding models to convert text to vectors.
"""
import os
from typing import List, Optional
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    GoogleGenerativeAIEmbeddings = None


class EmbeddingGenerator:
    """
    Handles embedding generation with caching for production efficiency.
    """
    
    def __init__(self, model_name: str = "models/embedding-001"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Google embedding model name
        """
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not found. Set it in .env file.")
        
        if GoogleGenerativeAIEmbeddings is None:
            raise RuntimeError(
                "langchain-google-genai not installed. Run: pip install langchain-google-genai"
            )
        
        # Initialize embeddings model
        model_id = model_name.replace("models/", "")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_id,
            google_api_key=self.api_key
        )
        
        # Cache for embeddings (in-memory)
        self._embedding_cache = {}
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings
        
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * 768  # Default embedding dimension
        
        # Check cache
        if use_cache and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            # Generate embedding
            embedding = self.embeddings.embed_query(text)
            
            # Cache it
            if use_cache:
                self._embedding_cache[text] = embedding
            
            return embedding
        except Exception as e:
            print(f"⚠️  Warning: Embedding generation failed: {e}")
            # Return zero vector on error
            return [0.0] * 768
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            batch_size: Number of texts to process at once
        
        Returns:
            List of embedding vectors
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Check cache for batch
            cached = []
            uncached = []
            uncached_indices = []
            
            for idx, text in enumerate(batch):
                if use_cache and text in self._embedding_cache:
                    cached.append(self._embedding_cache[text])
                else:
                    uncached.append(text)
                    uncached_indices.append(idx)
            
            # Generate embeddings for uncached texts
            if uncached:
                try:
                    batch_embeddings = self.embeddings.embed_documents(uncached)
                    
                    # Cache them
                    if use_cache:
                        for text, embedding in zip(uncached, batch_embeddings):
                            self._embedding_cache[text] = embedding
                except Exception as e:
                    print(f"⚠️  Warning: Batch embedding failed: {e}")
                    # Use zero vectors for failed embeddings
                    batch_embeddings = [[0.0] * 768] * len(uncached)
            else:
                batch_embeddings = []
            
            # Reconstruct batch results
            batch_results = [None] * len(batch)
            cache_idx = 0
            embed_idx = 0
            
            for idx in range(len(batch)):
                if idx in uncached_indices:
                    batch_results[idx] = batch_embeddings[embed_idx]
                    embed_idx += 1
                else:
                    batch_results[idx] = cached[cache_idx]
                    cache_idx += 1
            
            results.extend(batch_results)
        
        return results
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._embedding_cache)


# Global instance
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create global embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
