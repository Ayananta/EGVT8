#!/usr/bin/env python3
"""
FAISS-based Search Utility for PDF Chunks

This script provides advanced search capabilities using FAISS vector database.
"""

import json
import numpy as np
import faiss
import argparse
from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import re


class FAISSSearcher:
    """Advanced search utility using FAISS vector database."""
    
    def __init__(self, faiss_db_dir: str):
        """Initialize FAISS searcher."""
        self.faiss_db_dir = Path(faiss_db_dir)
        self.index = None
        self.chunks = []
        self.metadata = {}
        self.vectorizer = None
        
    def load_database(self):
        """Load FAISS database and metadata."""
        print(f"Loading FAISS database from: {self.faiss_db_dir}")
        
        # Load FAISS index
        index_path = self.faiss_db_dir / 'faiss_index.bin'
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = self.faiss_db_dir / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        self.metadata = data['metadata']
        
        print(f"Loaded {self.index.ntotal} vectors from FAISS database")
        
        # Recreate vectorizer for text queries
        if 'feature_names' in self.metadata:
            feature_names = self.metadata['feature_names']
            self.vectorizer = TfidfVectorizer(vocabulary=feature_names)
            self.vectorizer.fit(['dummy text'])  # Fit with dummy data
        
    def encode_text_query(self, query: str) -> np.ndarray:
        """Encode text query to vector."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not available. Cannot encode text queries.")
        
        # Preprocess query
        query = query.lower().strip()
        query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
        query = ' '.join(query.split())
        
        # Encode using TF-IDF
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        
        return query_vector
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks similar to a text query."""
        print(f"Searching for: '{query}'")
        
        # Encode query
        query_vector = self.encode_text_query(query)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search using FAISS
        distances, indices = self.index.search(query_vector, top_k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'rank': i + 1,
                    'chunk_id': chunk['chunk_id'],
                    'similarity': float(distance),
                    'header': chunk.get('header', ''),
                    'content': chunk['content'][:300] + '...' if len(chunk['content']) > 300 else chunk['content'],
                    'type': chunk.get('type', ''),
                    'char_count': chunk.get('char_count', 0)
                })
        
        return results
    
    def search_by_chunk_id(self, chunk_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find chunks similar to a specific chunk."""
        # Find the chunk
        chunk_idx = None
        for i, chunk in enumerate(self.chunks):
            if chunk['chunk_id'] == chunk_id:
                chunk_idx = i
                break
        
        if chunk_idx is None:
            raise ValueError(f"Chunk ID {chunk_id} not found")
        
        # Get the vector for this chunk
        chunk_vector = self.index.reconstruct(chunk_idx).reshape(1, -1)
        
        # Search for similar chunks (excluding self)
        k = top_k + 1  # Get one extra to exclude self
        distances, indices = self.index.search(chunk_vector, k)
        
        # Filter out self and format results
        results = []
        rank = 1
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks) and idx != chunk_idx:
                chunk = self.chunks[idx]
                results.append({
                    'rank': rank,
                    'chunk_id': chunk['chunk_id'],
                    'similarity': float(distance),
                    'header': chunk.get('header', ''),
                    'content': chunk['content'][:300] + '...' if len(chunk['content']) > 300 else chunk['content'],
                    'type': chunk.get('type', ''),
                    'char_count': chunk.get('char_count', 0)
                })
                rank += 1
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_chunk_by_id(self, chunk_id: int) -> Dict[str, Any]:
        """Get full chunk by ID."""
        for chunk in self.chunks:
            if chunk['chunk_id'] == chunk_id:
                return chunk
        raise ValueError(f"Chunk ID {chunk_id} not found")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'num_vectors': self.index.ntotal,
            'dimension': self.metadata.get('dimension', 'Unknown'),
            'index_type': self.metadata.get('index_type', 'Unknown'),
            'num_chunks': len(self.chunks),
            'chunk_size_stats': {
                'mean': np.mean([c.get('char_count', 0) for c in self.chunks]),
                'min': np.min([c.get('char_count', 0) for c in self.chunks]),
                'max': np.max([c.get('char_count', 0) for c in self.chunks])
            }
        }
    
    def batch_search(self, queries: List[str], top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Perform batch search on multiple queries."""
        results = {}
        for query in queries:
            try:
                results[query] = self.search_by_text(query, top_k)
            except Exception as e:
                results[query] = [{'error': str(e)}]
        return results


def main():
    """Main function for FAISS search utility."""
    parser = argparse.ArgumentParser(description='FAISS-based search for PDF chunks')
    parser.add_argument('--db-dir', '-d', default='faiss_db',
                       help='FAISS database directory')
    parser.add_argument('--query', '-q', help='Text query to search for')
    parser.add_argument('--chunk-id', '-c', type=int, help='Find chunks similar to this chunk ID')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--show-chunk', type=int, help='Show full content of specific chunk ID')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--batch', help='Comma-separated list of queries for batch search')
    
    args = parser.parse_args()
    
    # Create searcher
    searcher = FAISSSearcher(args.db_dir)
    searcher.load_database()
    
    if args.stats:
        # Show statistics
        stats = searcher.get_database_stats()
        print("=== FAISS Database Statistics ===")
        print(f"Number of vectors: {stats['num_vectors']}")
        print(f"Dimension: {stats['dimension']}")
        print(f"Index type: {stats['index_type']}")
        print(f"Number of chunks: {stats['num_chunks']}")
        print(f"Average chunk size: {stats['chunk_size_stats']['mean']:.1f} characters")
        print(f"Chunk size range: {stats['chunk_size_stats']['min']}-{stats['chunk_size_stats']['max']} characters")
    
    elif args.show_chunk:
        # Show specific chunk
        try:
            chunk = searcher.get_chunk_by_id(args.show_chunk)
            print(f"=== Chunk {args.show_chunk} ===")
            print(f"Type: {chunk.get('type', 'Unknown')}")
            print(f"Header: {chunk.get('header', 'No header')}")
            print(f"Size: {chunk.get('char_count', 0)} characters")
            print(f"\nContent:")
            print(chunk['content'])
        except ValueError as e:
            print(f"Error: {e}")
    
    elif args.query:
        # Text search
        results = searcher.search_by_text(args.query, args.top_k)
        
        print(f"=== Search Results for '{args.query}' ===")
        for result in results:
            print(f"\n{result['rank']}. Chunk {result['chunk_id']} (similarity: {result['similarity']:.4f})")
            print(f"   Type: {result['type']}")
            if result['header']:
                print(f"   Header: {result['header']}")
            print(f"   Content: {result['content']}")
            print(f"   Size: {result['char_count']} characters")
    
    elif args.chunk_id:
        # Similarity search
        results = searcher.search_by_chunk_id(args.chunk_id, args.top_k)
        
        print(f"=== Chunks Similar to Chunk {args.chunk_id} ===")
        for result in results:
            print(f"\n{result['rank']}. Chunk {result['chunk_id']} (similarity: {result['similarity']:.4f})")
            print(f"   Type: {result['type']}")
            if result['header']:
                print(f"   Header: {result['header']}")
            print(f"   Content: {result['content']}")
            print(f"   Size: {result['char_count']} characters")
    
    elif args.batch:
        # Batch search
        queries = [q.strip() for q in args.batch.split(',')]
        results = searcher.batch_search(queries, args.top_k)
        
        print("=== Batch Search Results ===")
        for query, query_results in results.items():
            print(f"\n--- Query: '{query}' ---")
            if 'error' in query_results[0]:
                print(f"Error: {query_results[0]['error']}")
            else:
                for result in query_results:
                    print(f"  {result['rank']}. Chunk {result['chunk_id']} ({result['similarity']:.4f})")
                    print(f"     {result['header'][:50]}...")
    
    else:
        print("Please provide a search option:")
        print("  --query 'text'     : Search by text")
        print("  --chunk-id N       : Find similar chunks")
        print("  --show-chunk N     : Show full chunk content")
        print("  --stats            : Show database statistics")
        print("  --batch 'q1,q2,q3' : Batch search")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
