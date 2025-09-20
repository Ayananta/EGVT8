#!/usr/bin/env python3
"""
Simple Vector Embedding Generator for PDF Chunks

This script converts PDF chunks into vector embeddings using sentence transformers.
Simplified version without pandas dependency.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimpleEmbeddingGenerator:
    """Simple class to generate vector embeddings for PDF chunks."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.chunks = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        print(f"Loading model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_chunks(self, chunks_file: str):
        """Load chunks from JSON file."""
        print(f"Loading chunks from: {chunks_file}")
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"Loaded {len(self.chunks)} chunks")
        except Exception as e:
            print(f"Error loading chunks: {e}")
            raise
    
    def preprocess_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Preprocess chunks for embedding generation."""
        processed_chunks = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            
            # Clean and preprocess the content
            content = content.strip()
            
            # Add metadata to context if available
            context_info = []
            if chunk.get('header'):
                context_info.append(f"Section: {chunk['header']}")
            if chunk.get('type'):
                context_info.append(f"Type: {chunk['type']}")
            
            if context_info:
                content = f"{' '.join(context_info)} {content}"
            
            processed_chunks.append(content)
        
        return processed_chunks
    
    def generate_embeddings(self, batch_size: int = 16) -> np.ndarray:
        """Generate embeddings for all chunks."""
        if not self.model:
            self.load_model()
        
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_chunks() first.")
        
        print("Preprocessing chunks...")
        processed_chunks = self.preprocess_chunks(self.chunks)
        
        print(f"Generating embeddings for {len(processed_chunks)} chunks...")
        print(f"Using batch size: {batch_size}")
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(processed_chunks), batch_size):
            batch = processed_chunks[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(processed_chunks)-1)//batch_size + 1}")
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(all_embeddings)
        print(f"Generated embeddings shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def find_similar_chunks(self, chunk_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar chunks to a given chunk."""
        if self.embeddings is None:
            raise ValueError("No embeddings available.")
        
        if chunk_id >= len(self.embeddings):
            raise ValueError(f"Chunk ID {chunk_id} out of range.")
        
        # Calculate similarities
        similarities = cosine_similarity([self.embeddings[chunk_id]], self.embeddings)[0]
        
        # Get top-k similar chunks (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        similar_scores = similarities[similar_indices]
        
        return list(zip(similar_indices, similar_scores))
    
    def save_embeddings(self, output_path: str, format: str = 'json'):
        """Save embeddings to file."""
        if self.embeddings is None:
            raise ValueError("No embeddings available. Call generate_embeddings() first.")
        
        output_path = Path(output_path)
        
        if format.lower() == 'numpy':
            np.save(output_path, self.embeddings)
            print(f"Embeddings saved as numpy array: {output_path}.npy")
            
        elif format.lower() == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'model_name': self.model_name,
                    'chunks': self.chunks
                }, f)
            print(f"Embeddings saved as pickle: {output_path}")
            
        elif format.lower() == 'json':
            # Convert embeddings to list format for JSON
            embeddings_list = self.embeddings.tolist()
            data = {
                'model_name': self.model_name,
                'embeddings': embeddings_list,
                'chunks': self.chunks,
                'metadata': {
                    'num_chunks': len(self.chunks),
                    'embedding_dimension': self.embeddings.shape[1],
                    'model_info': 'Sentence transformer model for semantic similarity'
                }
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Embeddings saved as JSON: {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_similarity_analysis(self, output_path: str, top_k: int = 10):
        """Save similarity analysis results."""
        if self.embeddings is None:
            raise ValueError("No embeddings available.")
        
        print("Generating similarity analysis...")
        similarity_matrix = cosine_similarity(self.embeddings)
        
        # Find most similar pairs
        most_similar = []
        for i in range(len(self.chunks)):
            similarities = similarity_matrix[i]
            # Get top-k similar chunks (excluding self)
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            for j in top_indices:
                if similarities[j] > 0.7:  # Only include high similarity
                    most_similar.append({
                        'chunk_1_id': i + 1,
                        'chunk_2_id': j + 1,
                        'similarity': float(similarities[j]),
                        'chunk_1_header': self.chunks[i].get('header', ''),
                        'chunk_2_header': self.chunks[j].get('header', '')
                    })
        
        # Sort by similarity
        most_similar.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'similarity_analysis': most_similar,
                'total_pairs_analyzed': len(most_similar),
                'average_similarity': float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
                'max_similarity': float(np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
                'min_similarity': float(np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Similarity analysis saved to: {output_path}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the generated embeddings."""
        if self.embeddings is None:
            return {}
        
        stats = {
            'num_chunks': len(self.chunks),
            'embedding_dimension': self.embeddings.shape[1],
            'model_name': self.model_name,
            'embedding_mean': float(np.mean(self.embeddings)),
            'embedding_std': float(np.std(self.embeddings)),
            'embedding_min': float(np.min(self.embeddings)),
            'embedding_max': float(np.max(self.embeddings)),
            'chunk_size_stats': {
                'mean_chars': float(np.mean([c.get('char_count', 0) for c in self.chunks])),
                'std_chars': float(np.std([c.get('char_count', 0) for c in self.chunks])),
                'min_chars': int(np.min([c.get('char_count', 0) for c in self.chunks])),
                'max_chars': int(np.max([c.get('char_count', 0) for c in self.chunks]))
            }
        }
        
        return stats


def main():
    """Main function to run the embedding generator."""
    parser = argparse.ArgumentParser(description='Generate vector embeddings for PDF chunks')
    parser.add_argument('chunks_file', help='Path to the JSON file containing chunks')
    parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2', 
                       help='Sentence transformer model to use')
    parser.add_argument('--output', '-o', default='embeddings', 
                       help='Output file prefix (without extension)')
    parser.add_argument('--format', '-f', choices=['numpy', 'pickle', 'json'], 
                       default='json', help='Output format')
    parser.add_argument('--batch-size', '-b', type=int, default=16, 
                       help='Batch size for embedding generation')
    parser.add_argument('--similarity-analysis', action='store_true', 
                       help='Generate similarity analysis')
    
    args = parser.parse_args()
    
    # Create embedding generator
    generator = SimpleEmbeddingGenerator(model_name=args.model)
    
    try:
        # Load chunks
        generator.load_chunks(args.chunks_file)
        
        # Generate embeddings
        embeddings = generator.generate_embeddings(batch_size=args.batch_size)
        
        # Save embeddings
        output_path = f"{args.output}.{args.format}"
        generator.save_embeddings(output_path, args.format)
        
        # Generate similarity analysis if requested
        if args.similarity_analysis:
            similarity_path = f"{args.output}_similarity_analysis.json"
            generator.save_similarity_analysis(similarity_path)
        
        # Print statistics
        stats = generator.get_embedding_stats()
        print(f"\n=== Embedding Statistics ===")
        print(f"Model: {stats['model_name']}")
        print(f"Number of chunks: {stats['num_chunks']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Average chunk size: {stats['chunk_size_stats']['mean_chars']:.1f} characters")
        print(f"Embedding value range: [{stats['embedding_min']:.3f}, {stats['embedding_max']:.3f}]")
        print(f"Embedding mean: {stats['embedding_mean']:.3f}")
        print(f"Embedding std: {stats['embedding_std']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
