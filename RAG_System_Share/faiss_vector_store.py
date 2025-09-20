#!/usr/bin/env python3
"""
FAISS Vector Database Storage for PDF Chunks

This script stores vector embeddings in FAISS database for efficient similarity search.
"""

import json
import numpy as np
import faiss
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time


class FAISSVectorStore:
    """FAISS vector database for storing and searching PDF chunk embeddings."""
    
    def __init__(self, dimension: int = None, index_type: str = 'flat'):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension (will be auto-detected if None)
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.chunks = []
        self.chunk_ids = []
        self.metadata = {}
        
        # Index type configurations
        self.index_configs = {
            'flat': {'description': 'Exact search, slower but most accurate'},
            'ivf': {'description': 'Inverted file index, faster approximate search'},
            'hnsw': {'description': 'Hierarchical navigable small world, good balance'}
        }
        
    def create_index(self, dimension: int):
        """Create FAISS index based on the specified type."""
        self.dimension = dimension
        
        if self.index_type == 'flat':
            # Flat index for exact search
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            
        elif self.index_type == 'ivf':
            # IVF index for faster approximate search
            nlist = min(100, max(10, len(self.chunks) // 10))  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
        elif self.index_type == 'hnsw':
            # HNSW index for hierarchical navigable small world
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the connectivity parameter
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        print(f"Created FAISS {self.index_type} index with dimension {dimension}")
    
    def load_embeddings(self, embeddings_file: str):
        """Load embeddings from JSON file."""
        print(f"Loading embeddings from: {embeddings_file}")
        
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        embeddings = np.array(data['embeddings']).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create chunk IDs
        self.chunk_ids = [chunk['chunk_id'] for chunk in self.chunks]
        
        # Store metadata
        self.metadata = data['metadata']
        
        print(f"Loaded {len(self.chunks)} chunks with {embeddings.shape[1]} dimensions")
        
        return embeddings
    
    def build_index(self, embeddings_file: str, index_type: str = None):
        """Build FAISS index from embeddings file."""
        if index_type:
            self.index_type = index_type
        
        # Load embeddings
        embeddings = self.load_embeddings(embeddings_file)
        
        # Create index
        self.create_index(embeddings.shape[1])
        
        # Train index if needed (for IVF)
        if self.index_type == 'ivf':
            print("Training IVF index...")
            self.index.train(embeddings)
        
        # Add embeddings to index
        print("Adding embeddings to index...")
        self.index.add(embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
        
        return embeddings
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalize query vector
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector.reshape(1, -1))
        
        # Search
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx]
                results.append({
                    'rank': i + 1,
                    'chunk_id': chunk['chunk_id'],
                    'similarity': float(distance),  # For inner product, higher is more similar
                    'header': chunk.get('header', ''),
                    'content': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content'],
                    'type': chunk.get('type', ''),
                    'char_count': chunk.get('char_count', 0)
                })
        
        return results
    
    def search_by_text(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using text query (requires original vectorizer)."""
        # This is a placeholder - in practice, you'd need to encode the text
        # using the same vectorizer that created the embeddings
        print(f"Text search not fully implemented. Use search() with pre-encoded vectors.")
        return []
    
    def save_index(self, output_dir: str):
        """Save FAISS index and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save FAISS index
        index_path = output_dir / 'faiss_index.bin'
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata and chunks
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': self.chunks,
                'chunk_ids': self.chunk_ids,
                'metadata': self.metadata,
                'index_type': self.index_type,
                'dimension': self.dimension,
                'num_vectors': self.index.ntotal
            }, f, indent=2, ensure_ascii=False)
        
        print(f"FAISS index saved to: {index_path}")
        print(f"Metadata saved to: {metadata_path}")
    
    def load_index(self, index_dir: str):
        """Load FAISS index and metadata."""
        index_dir = Path(index_dir)
        
        # Load FAISS index
        index_path = index_dir / 'faiss_index.bin'
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = index_dir / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        self.chunk_ids = data['chunk_ids']
        self.metadata = data['metadata']
        self.index_type = data['index_type']
        self.dimension = data['dimension']
        
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"Index type: {self.index_type}")
        print(f"Dimension: {self.dimension}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'num_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'num_chunks': len(self.chunks),
            'index_description': self.index_configs.get(self.index_type, {}).get('description', 'Unknown')
        }


def main():
    """Main function for FAISS vector store operations."""
    parser = argparse.ArgumentParser(description='FAISS Vector Database for PDF Chunks')
    parser.add_argument('--embeddings', '-e', default='Data/elita_embeddings_fast.json',
                       help='Path to embeddings JSON file')
    parser.add_argument('--index-type', '-t', choices=['flat', 'ivf', 'hnsw'], 
                       default='flat', help='Type of FAISS index')
    parser.add_argument('--output-dir', '-o', default='faiss_db',
                       help='Output directory for FAISS database')
    parser.add_argument('--load', '-l', help='Load existing FAISS database from directory')
    parser.add_argument('--search', '-s', help='Search query (text)')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    # Create vector store
    vector_store = FAISSVectorStore(index_type=args.index_type)
    
    try:
        if args.load:
            # Load existing database
            vector_store.load_index(args.load)
            
            if args.search:
                print(f"Searching for: '{args.search}'")
                # Note: This would need proper text encoding in a full implementation
                results = vector_store.search_by_text(args.search, args.top_k)
                for result in results:
                    print(f"{result['rank']}. Chunk {result['chunk_id']} (similarity: {result['similarity']:.4f})")
                    print(f"   {result['header'][:60]}...")
                    print(f"   {result['content'][:100]}...")
                    print()
        
        else:
            # Build new database
            print("Building FAISS vector database...")
            embeddings = vector_store.build_index(args.embeddings, args.index_type)
            
            # Save database
            vector_store.save_index(args.output_dir)
            
            # Print statistics
            stats = vector_store.get_stats()
            print(f"\n=== Database Statistics ===")
            print(f"Index type: {stats['index_type']} ({stats['index_description']})")
            print(f"Number of vectors: {stats['num_vectors']}")
            print(f"Dimension: {stats['dimension']}")
            print(f"Number of chunks: {stats['num_chunks']}")
            
            # Test search with a random vector
            if stats['num_vectors'] > 0:
                print(f"\n=== Testing Search ===")
                test_vector = np.random.random((1, stats['dimension'])).astype('float32')
                results = vector_store.search(test_vector, 3)
                
                print("Random search results:")
                for result in results:
                    print(f"  {result['rank']}. Chunk {result['chunk_id']} (similarity: {result['similarity']:.4f})")
                    print(f"     {result['header'][:50]}...")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
