"""
Database management module for production deployment.
Handles initialization, indexing, and maintenance of the vector database.
"""
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from vector_db import get_vector_database
from document_processor import get_document_processor
from doc_utils import extract_text_from_path, is_supported_file_type


class DatabaseManager:
    """
    Manages the vector database: initialization, indexing, updates.
    """
    
    def __init__(self, reference_dir: str = "./reference_docs"):
        """
        Initialize database manager.
        
        Args:
            reference_dir: Directory containing reference documents
        """
        self.reference_dir = reference_dir
        self.vector_db = get_vector_database()
        self.doc_processor = get_document_processor()
    
    def index_reference_documents(
        self,
        force_reindex: bool = False,
        file_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Index all reference documents into the vector database.
        
        Args:
            force_reindex: If True, clear existing data and reindex
            file_paths: Optional list of specific files to index
        
        Returns:
            Dictionary with indexing results
        """
        results = {
            "indexed": 0,
            "failed": 0,
            "errors": []
        }
        
        # Clear if force reindex
        if force_reindex:
            print("Clearing existing database...")
            self.vector_db.clear_collection()
        
        # Get files to index
        if file_paths:
            files_to_index = file_paths
        else:
            files_to_index = self._get_reference_files()
        
        if not files_to_index:
            print("WARNING: No files found to index.")
            return results
        
        print(f"Indexing {len(files_to_index)} documents...")
        
        # Process each file
        all_documents = []
        
        for file_path in files_to_index:
            try:
                # Check file type
                is_valid, _ = is_supported_file_type(file_path)
                if not is_valid:
                    results["failed"] += 1
                    results["errors"].append(f"Unsupported file type: {file_path}")
                    continue
                
                # Get label
                label = os.path.basename(file_path)
                
                # Process file
                documents = self.doc_processor.process_file(file_path, source_label=label)
                
                if documents:
                    all_documents.extend(documents)
                    results["indexed"] += 1
                    print(f"  OK: Indexed: {label} ({len(documents)} chunks)")
                else:
                    results["failed"] += 1
                    results["errors"].append(f"No content extracted: {file_path}")
            
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Error processing {file_path}: {str(e)}")
                print(f"  FAILED: {os.path.basename(file_path)} - {e}")
        
        # Add all documents to vector database
        if all_documents:
            print(f"\nAdding {len(all_documents)} document chunks to vector database...")
            try:
                self.vector_db.add_documents(all_documents)
                print(f"SUCCESS: Successfully indexed {results['indexed']} documents with {len(all_documents)} chunks")
            except Exception as e:
                results["errors"].append(f"Failed to add documents to database: {str(e)}")
                print(f"ERROR: Error adding documents: {e}")
        
        return results
    
    def index_text(
        self,
        text: str,
        source_label: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Index a text document.
        
        Args:
            text: Text content
            source_label: Label for the source
            metadata: Optional metadata
        
        Returns:
            True if successful
        """
        try:
            documents = self.doc_processor.process_text(text, source_label, metadata)
            if documents:
                self.vector_db.add_documents(documents)
                return True
            return False
        except Exception as e:
            print(f"❌ Error indexing text: {e}")
            return False
    
    def _get_reference_files(self) -> List[str]:
        """
        Get all reference files from the reference directory.
        
        Returns:
            List of file paths
        """
        files = []
        
        if not os.path.isdir(self.reference_dir):
            return files
        
        for name in os.listdir(self.reference_dir):
            full_path = os.path.join(self.reference_dir, name)
            if os.path.isfile(full_path):
                is_valid, _ = is_supported_file_type(full_path)
                if is_valid:
                    files.append(full_path)
        
        return files
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with statistics
        """
        stats = self.vector_db.get_collection_info()
        stats["reference_directory"] = self.reference_dir
        stats["reference_files"] = len(self._get_reference_files())
        return stats
    
    def check_database_health(self) -> Dict[str, Any]:
        """
        Check database health and readiness.
        
        Returns:
            Dictionary with health status
        """
        health = {
            "status": "healthy",
            "database_exists": False,
            "document_count": 0,
            "reference_files": 0,
            "warnings": []
        }
        
        try:
            # Check database
            stats = self.get_database_stats()
            health["database_exists"] = True
            health["document_count"] = stats.get("document_count", 0)
            health["reference_files"] = stats.get("reference_files", 0)
            
            # Check if database is empty
            if health["document_count"] == 0:
                health["status"] = "empty"
                health["warnings"].append("Database is empty. Run indexing first.")
            
            # Check if reference files exist
            if health["reference_files"] == 0:
                health["warnings"].append("No reference files found in reference_docs folder.")
        
        except Exception as e:
            health["status"] = "error"
            health["warnings"].append(f"Database error: {str(e)}")
        
        return health


# Global instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
