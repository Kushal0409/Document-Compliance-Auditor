"""
Database initialization script for production deployment.
Run this script to index reference documents into the vector database.
"""
import os
import sys
from pathlib import Path

from database_manager import get_database_manager


def main():
    """Initialize the vector database with reference documents."""
    print("=" * 60)
    print("Vector Database Initialization")
    print("=" * 60)
    print()
    
    # Check if reference_docs directory exists
    reference_dir = "./reference_docs"
    if not os.path.isdir(reference_dir):
        print(f"ERROR: Reference directory '{reference_dir}' not found.")
        print(f"   Please create the directory and add regulation documents.")
        sys.exit(1)
    
    # Get database manager
    try:
        db_manager = get_database_manager()
    except Exception as e:
        print(f"ERROR: Error initializing database manager: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Check database health
    print("Checking database health...")
    health = db_manager.check_database_health()
    
    if health["status"] == "healthy" and health["document_count"] > 0:
        print(f"OK: Database is healthy with {health['document_count']} documents.")
        response = input("\nDo you want to reindex? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping reindexing.")
            sys.exit(0)
        force_reindex = True
    else:
        print("WARNING: Database is empty or has issues.")
        force_reindex = False
    
    # Index documents
    print("\nStarting indexing process...")
    results = db_manager.index_reference_documents(force_reindex=force_reindex)
    
    # Print results
    print("\n" + "=" * 60)
    print("Indexing Results")
    print("=" * 60)
    print(f"SUCCESS: Successfully indexed: {results['indexed']} documents")
    print(f"FAILED: {results['failed']} documents")
    
    if results['errors']:
        print(f"\nWARNINGS: Errors ({len(results['errors'])}):")
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"   - {error}")
        if len(results['errors']) > 10:
            print(f"   ... and {len(results['errors']) - 10} more errors")
    
    # Show database stats
    print("\n" + "=" * 60)
    print("Database Statistics")
    print("=" * 60)
    stats = db_manager.get_database_stats()
    print(f"Collection: {stats.get('collection_name', 'N/A')}")
    print(f"Document chunks: {stats.get('document_count', 0)}")
    print(f"Reference files: {stats.get('reference_files', 0)}")
    
    print("\nSUCCESS: Database initialization complete!")
    print("\nYou can now use the compliance auditor with semantic search.")


if __name__ == "__main__":
    main()
