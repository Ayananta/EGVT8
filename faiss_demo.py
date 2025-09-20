#!/usr/bin/env python3
"""
FAISS Vector Database Demo

This script demonstrates the complete FAISS vector database functionality
for the PDF chunks with various search and analysis capabilities.
"""

import json
import numpy as np
from pathlib import Path
from faiss_search import FAISSSearcher
import time


def demo_faiss_functionality():
    """Demonstrate comprehensive FAISS functionality."""
    print("=" * 70)
    print("FAISS VECTOR DATABASE DEMONSTRATION")
    print("=" * 70)
    
    # Initialize searcher
    searcher = FAISSSearcher('faiss_db')
    searcher.load_database()
    
    # Show database info
    stats = searcher.get_database_stats()
    print(f"\n📊 DATABASE OVERVIEW:")
    print(f"   Vectors stored: {stats['num_vectors']}")
    print(f"   Chunks indexed: {stats['num_chunks']}")
    print(f"   Average chunk size: {stats['chunk_size_stats']['mean']:.1f} characters")
    print(f"   Memory usage: ~3.66 MB")
    
    # Demo 1: Text-based semantic search
    print(f"\n🔍 DEMO 1: SEMANTIC TEXT SEARCH")
    print("-" * 40)
    
    search_queries = [
        "apartment ownership rights",
        "building construction permit", 
        "legal judgment appeal",
        "developer promoter responsibilities"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = searcher.search_by_text(query, top_k=2)
        
        for result in results:
            print(f"  📄 Chunk {result['chunk_id']} (similarity: {result['similarity']:.4f})")
            print(f"     {result['header'][:60]}...")
            print(f"     {result['content'][:80]}...")
    
    # Demo 2: Similarity-based chunk discovery
    print(f"\n🔗 DEMO 2: SIMILARITY-BASED DISCOVERY")
    print("-" * 40)
    
    test_chunk_ids = [49, 100, 150]
    
    for chunk_id in test_chunk_ids:
        print(f"\nFinding chunks similar to Chunk {chunk_id}:")
        try:
            # Get the original chunk first
            original_chunk = searcher.get_chunk_by_id(chunk_id)
            print(f"  Original: {original_chunk.get('header', 'No header')[:50]}...")
            
            # Find similar chunks
            similar_results = searcher.search_by_chunk_id(chunk_id, top_k=3)
            
            for result in similar_results:
                print(f"  📄 Chunk {result['chunk_id']} (similarity: {result['similarity']:.4f})")
                print(f"     {result['header'][:50]}...")
                
        except ValueError:
            print(f"  Chunk {chunk_id} not found")
    
    # Demo 3: Batch search
    print(f"\n📦 DEMO 3: BATCH SEARCH")
    print("-" * 40)
    
    batch_queries = [
        "court judgment",
        "apartment act",
        "developer liability"
    ]
    
    batch_results = searcher.batch_search(batch_queries, top_k=2)
    
    for query, results in batch_results.items():
        print(f"\nBatch query: '{query}'")
        for result in results[:2]:  # Show top 2 results
            print(f"  📄 Chunk {result['chunk_id']} ({result['similarity']:.4f})")
            print(f"     {result['header'][:40]}...")
    
    # Demo 4: Performance comparison
    print(f"\n⚡ DEMO 4: SEARCH PERFORMANCE")
    print("-" * 40)
    
    # Time different search operations
    test_queries = [
        "apartment ownership",
        "legal proceedings", 
        "construction permit",
        "developer obligations",
        "court decision"
    ]
    
    total_time = 0
    num_searches = 0
    
    for query in test_queries:
        start_time = time.time()
        results = searcher.search_by_text(query, top_k=5)
        search_time = time.time() - start_time
        total_time += search_time
        num_searches += 1
        
        print(f"  '{query}': {search_time*1000:.2f}ms, {len(results)} results")
    
    avg_time = total_time / num_searches
    print(f"\n📈 Performance Summary:")
    print(f"   Average search time: {avg_time*1000:.2f}ms")
    print(f"   Searches per second: {1/avg_time:.1f}")
    print(f"   Total searches: {num_searches}")
    
    # Demo 5: Show specific chunk details
    print(f"\n📋 DEMO 5: CHUNK DETAIL VIEW")
    print("-" * 40)
    
    # Show a specific chunk in detail
    chunk_id = 49
    try:
        chunk = searcher.get_chunk_by_id(chunk_id)
        print(f"\nFull content of Chunk {chunk_id}:")
        print(f"Type: {chunk.get('type', 'Unknown')}")
        print(f"Header: {chunk.get('header', 'No header')}")
        print(f"Size: {chunk.get('char_count', 0)} characters")
        print(f"\nContent preview:")
        print(chunk['content'][:400] + "..." if len(chunk['content']) > 400 else chunk['content'])
        
    except ValueError:
        print(f"Chunk {chunk_id} not found")
    
    print(f"\n✅ FAISS DEMONSTRATION COMPLETE!")
    print(f"   Your PDF chunks are now stored in a high-performance vector database")
    print(f"   enabling fast semantic search and similarity analysis.")
    
    print("=" * 70)


def show_faiss_files():
    """Show all FAISS-related files created."""
    print(f"\n📁 FAISS FILES CREATED:")
    
    faiss_files = [
        "faiss_db/faiss_index.bin",
        "faiss_db/metadata.json",
        "faiss_comparison/flat_index.bin",
        "faiss_comparison/ivf_index.bin", 
        "faiss_comparison/hnsw_index.bin",
        "faiss_comparison/comparison_summary.json"
    ]
    
    total_size = 0
    for file_path in faiss_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            total_size += size
            print(f"   ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"   ❌ {file_path} (not found)")
    
    print(f"\n📊 Total FAISS database size: {total_size:,} bytes ({total_size/(1024*1024):.2f} MB)")
    
    print(f"\n🛠️  AVAILABLE FAISS TOOLS:")
    print(f"   • faiss_vector_store.py - Build and manage FAISS databases")
    print(f"   • faiss_search.py - Search and similarity analysis")
    print(f"   • faiss_manager.py - Performance analysis and optimization")
    print(f"   • faiss_demo.py - This demonstration script")
    
    print(f"\n🚀 USAGE EXAMPLES:")
    print(f"   # Search by text:")
    print(f"   python faiss_search.py --query 'apartment ownership'")
    print(f"   ")
    print(f"   # Find similar chunks:")
    print(f"   python faiss_search.py --chunk-id 49")
    print(f"   ")
    print(f"   # Show database statistics:")
    print(f"   python faiss_search.py --stats")
    print(f"   ")
    print(f"   # Benchmark performance:")
    print(f"   python faiss_manager.py --benchmark faiss_db/faiss_index.bin")


if __name__ == "__main__":
    demo_faiss_functionality()
    show_faiss_files()
