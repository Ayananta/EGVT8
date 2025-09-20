#!/usr/bin/env python3
"""
Check embedding results
"""

import json
import numpy as np

def check_embeddings():
    """Check the generated embeddings."""
    try:
        # Load embeddings
        with open('Data/elita_embeddings_fast.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("=== Embedding Results ===")
        print(f"Method: {data['method']}")
        print(f"Number of chunks: {data['metadata']['num_chunks']}")
        print(f"Embedding dimension: {data['metadata']['embedding_dimension']}")
        print(f"Vocabulary size: {data['metadata']['vocabulary_size']}")
        
        # Check embeddings
        embeddings = data['embeddings']
        print(f"Embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
        
        # Convert to numpy for stats
        embeddings_array = np.array(embeddings)
        print(f"Embedding statistics:")
        print(f"  Mean: {np.mean(embeddings_array):.4f}")
        print(f"  Std: {np.std(embeddings_array):.4f}")
        print(f"  Min: {np.min(embeddings_array):.4f}")
        print(f"  Max: {np.max(embeddings_array):.4f}")
        print(f"  Sparsity: {np.sum(embeddings_array == 0) / embeddings_array.size:.4f}")
        
        # Check similarity analysis
        try:
            with open('Data/elita_embeddings_fast_similarity_analysis.json', 'r', encoding='utf-8') as f:
                sim_data = json.load(f)
            
            print(f"\n=== Similarity Analysis ===")
            print(f"Total similar pairs: {sim_data['total_pairs_analyzed']}")
            print(f"Average similarity: {sim_data['average_similarity']:.4f}")
            print(f"Max similarity: {sim_data['max_similarity']:.4f}")
            print(f"Min similarity: {sim_data['min_similarity']:.4f}")
            
            # Show top similar pairs
            print(f"\nTop 5 most similar chunk pairs:")
            for i, pair in enumerate(sim_data['similarity_analysis'][:5]):
                print(f"  {i+1}. Chunks {pair['chunk_1_id']} & {pair['chunk_2_id']}: {pair['similarity']:.4f}")
                print(f"     Headers: '{pair['chunk_1_header'][:50]}...' & '{pair['chunk_2_header'][:50]}...'")
                
        except Exception as e:
            print(f"Error reading similarity analysis: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    check_embeddings()
