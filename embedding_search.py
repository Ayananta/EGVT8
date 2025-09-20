#!/usr/bin/env python3
"""
Embedding Search Utility

This script demonstrates how to search through the PDF chunks using their vector embeddings.
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse


class EmbeddingSearcher:
    """Search utility for PDF chunks using vector embeddings."""
    
    def __init__(self, embeddings_file: str):
        """Initialize the searcher with embeddings data."""
        self.embeddings_file = embeddings_file
        self.data = None
        self.embeddings = None
        self.vectorizer = None
        
    def load_embeddings(self):
        """Load embeddings and metadata."""
        with open(self.embeddings_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.embeddings = np.array(self.data['embeddings'])
        print(f"Loaded embeddings for {len(self.data['chunks'])} chunks")
        
    def search_by_text(self, query: str, top_k: int = 5):
        """Search for chunks similar to a text query."""
        if self.vectorizer is None:
            # Recreate vectorizer from saved vocabulary
            feature_names = self.data['metadata']['feature_names']
            self.vectorizer = TfidfVectorizer(vocabulary=feature_names)
            # Fit the vectorizer with dummy data to make it usable
            self.vectorizer.fit(['dummy text'])
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query.lower()]).toarray()
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.data['chunks'][idx]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'similarity': float(similarities[idx]),
                'header': chunk.get('header', ''),
                'content': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content'],
                'type': chunk.get('type', ''),
                'char_count': chunk.get('char_count', 0)
            })
        
        return results
    
    def find_similar_chunks(self, chunk_id: int, top_k: int = 5):
        """Find chunks similar to a specific chunk."""
        if chunk_id < 1 or chunk_id > len(self.embeddings):
            raise ValueError(f"Chunk ID {chunk_id} out of range (1-{len(self.embeddings)})")
        
        chunk_idx = chunk_id - 1  # Convert to 0-based index
        
        # Calculate similarities
        similarities = cosine_similarity([self.embeddings[chunk_idx]], self.embeddings)[0]
        
        # Get top results (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in top_indices:
            chunk = self.data['chunks'][idx]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'similarity': float(similarities[idx]),
                'header': chunk.get('header', ''),
                'content': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content'],
                'type': chunk.get('type', ''),
                'char_count': chunk.get('char_count', 0)
            })
        
        return results


def main():
    """Main function for the search utility."""
    parser = argparse.ArgumentParser(description='Search PDF chunks using vector embeddings')
    parser.add_argument('--embeddings', '-e', default='Data/elita_embeddings_fast.json',
                       help='Path to embeddings file')
    parser.add_argument('--query', '-q', help='Text query to search for')
    parser.add_argument('--chunk-id', '-c', type=int, help='Find chunks similar to this chunk ID')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    # Create searcher
    searcher = EmbeddingSearcher(args.embeddings)
    searcher.load_embeddings()
    
    if args.query:
        print(f"Searching for: '{args.query}'")
        results = searcher.search_by_text(args.query, args.top_k)
        
        print(f"\n=== Search Results ===")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Chunk {result['chunk_id']} (similarity: {result['similarity']:.4f})")
            print(f"   Type: {result['type']}")
            if result['header']:
                print(f"   Header: {result['header']}")
            print(f"   Content: {result['content']}")
            print(f"   Size: {result['char_count']} characters")
    
    elif args.chunk_id:
        print(f"Finding chunks similar to Chunk {args.chunk_id}")
        results = searcher.find_similar_chunks(args.chunk_id, args.top_k)
        
        print(f"\n=== Similar Chunks ===")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Chunk {result['chunk_id']} (similarity: {result['similarity']:.4f})")
            print(f"   Type: {result['type']}")
            if result['header']:
                print(f"   Header: {result['header']}")
            print(f"   Content: {result['content']}")
            print(f"   Size: {result['char_count']} characters")
    
    else:
        print("Please provide either --query or --chunk-id")
        print("\nExamples:")
        print("  python embedding_search.py --query 'apartment ownership'")
        print("  python embedding_search.py --chunk-id 10")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
