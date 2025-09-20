#!/usr/bin/env python3
"""
Demo script to show the embeddings functionality
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def demo_embeddings():
    """Demonstrate the embedding functionality."""
    try:
        # Load embeddings
        with open('Data/elita_embeddings_fast.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("=== PDF Chunks to Vector Embeddings Demo ===")
        print(f"✅ Successfully converted {data['metadata']['num_chunks']} PDF chunks to vector embeddings")
        print(f"✅ Embedding method: {data['method'].upper()}")
        print(f"✅ Embedding dimension: {data['metadata']['embedding_dimension']}")
        print(f"✅ Vocabulary size: {data['metadata']['vocabulary_size']}")
        
        # Show sample chunks
        print(f"\n=== Sample Chunks ===")
        for i in range(min(3, len(data['chunks']))):
            chunk = data['chunks'][i]
            print(f"\nChunk {chunk['chunk_id']}:")
            print(f"  Type: {chunk['type']}")
            if chunk.get('header'):
                print(f"  Header: {chunk['header'][:80]}...")
            print(f"  Content preview: {chunk['content'][:100]}...")
            print(f"  Characters: {chunk['char_count']}")
        
        # Show embedding statistics
        embeddings_array = np.array(data['embeddings'])
        print(f"\n=== Embedding Statistics ===")
        print(f"  Mean value: {np.mean(embeddings_array):.4f}")
        print(f"  Standard deviation: {np.std(embeddings_array):.4f}")
        print(f"  Value range: [{np.min(embeddings_array):.4f}, {np.max(embeddings_array):.4f}]")
        print(f"  Sparsity (zeros): {np.sum(embeddings_array == 0) / embeddings_array.size:.1%}")
        
        # Demonstrate similarity search
        print(f"\n=== Similarity Search Demo ===")
        if len(embeddings_array) > 1:
            # Find most similar chunks to the first chunk
            similarities = cosine_similarity([embeddings_array[0]], embeddings_array)[0]
            similar_indices = np.argsort(similarities)[::-1][1:4]  # Top 3 similar (excluding self)
            
            print(f"Most similar chunks to Chunk 1:")
            for idx in similar_indices:
                chunk = data['chunks'][idx]
                sim_score = similarities[idx]
                print(f"  Chunk {chunk['chunk_id']}: similarity = {sim_score:.4f}")
                if chunk.get('header'):
                    print(f"    Header: {chunk['header'][:60]}...")
        
        print(f"\n=== Files Generated ===")
        print(f"📄 Main embeddings: Data/elita_embeddings_fast.json")
        print(f"📊 Embedding format: JSON with metadata")
        print(f"🔍 Ready for similarity search and analysis")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    demo_embeddings()
