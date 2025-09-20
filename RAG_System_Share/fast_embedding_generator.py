#!/usr/bin/env python3
"""
Fast Vector Embedding Generator for PDF Chunks

This script converts PDF chunks into vector embeddings using TF-IDF and other fast methods.
Alternative to sentence transformers for faster processing.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
from collections import Counter


class FastEmbeddingGenerator:
    """Fast embedding generator using TF-IDF and other methods."""
    
    def __init__(self, method: str = 'tfidf', max_features: int = 5000):
        """
        Initialize the embedding generator.
        
        Args:
            method: Embedding method ('tfidf', 'count', 'binary')
            max_features: Maximum number of features for TF-IDF
        """
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        self.embeddings = None
        self.chunks = None
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding generation."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
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
    
    def prepare_texts(self) -> List[str]:
        """Prepare texts for embedding generation."""
        texts = []
        
        for chunk in self.chunks:
            content = chunk.get('content', '')
            
            # Preprocess the content
            processed_content = self.preprocess_text(content)
            
            # Add metadata to context if available
            context_info = []
            if chunk.get('header'):
                header_text = self.preprocess_text(chunk['header'])
                context_info.append(header_text)
            if chunk.get('type'):
                context_info.append(chunk['type'].lower())
            
            if context_info:
                combined_text = ' '.join(context_info) + ' ' + processed_content
            else:
                combined_text = processed_content
            
            texts.append(combined_text)
        
        return texts
    
    def generate_embeddings(self) -> np.ndarray:
        """Generate embeddings for all chunks."""
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_chunks() first.")
        
        print("Preparing texts for embedding generation...")
        texts = self.prepare_texts()
        
        print(f"Generating {self.method} embeddings for {len(texts)} chunks...")
        
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2),  # Use both unigrams and bigrams
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.95  # Ignore terms that appear in more than 95% of documents
            )
        elif self.method == 'count':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                use_idf=False,
                ngram_range=(1, 2)
            )
        elif self.method == 'binary':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                use_idf=False,
                binary=True,
                ngram_range=(1, 2)
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Generate embeddings
        self.embeddings = self.vectorizer.fit_transform(texts).toarray()
        
        print(f"Generated embeddings shape: {self.embeddings.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
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
    
    def get_top_terms(self, chunk_id: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top terms for a specific chunk."""
        if self.embeddings is None or self.vectorizer is None:
            raise ValueError("No embeddings available.")
        
        if chunk_id >= len(self.embeddings):
            raise ValueError(f"Chunk ID {chunk_id} out of range.")
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get embedding for this chunk
        chunk_embedding = self.embeddings[chunk_id]
        
        # Get top terms
        top_indices = np.argsort(chunk_embedding)[::-1][:top_k]
        top_terms = [(feature_names[i], chunk_embedding[i]) for i in top_indices]
        
        return top_terms
    
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
                    'method': self.method,
                    'chunks': self.chunks,
                    'vectorizer': self.vectorizer
                }, f)
            print(f"Embeddings saved as pickle: {output_path}")
            
        elif format.lower() == 'json':
            # Convert embeddings to list format for JSON
            embeddings_list = self.embeddings.tolist()
            data = {
                'method': self.method,
                'embeddings': embeddings_list,
                'chunks': self.chunks,
                'metadata': {
                    'num_chunks': len(self.chunks),
                    'embedding_dimension': self.embeddings.shape[1],
                    'vocabulary_size': len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
                    'feature_names': self.vectorizer.get_feature_names_out().tolist() if self.vectorizer else []
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
                if similarities[j] > 0.3:  # Lower threshold for TF-IDF
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
                'similarity_analysis': most_similar[:100],  # Limit to top 100
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
            'method': self.method,
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            'embedding_mean': float(np.mean(self.embeddings)),
            'embedding_std': float(np.std(self.embeddings)),
            'embedding_min': float(np.min(self.embeddings)),
            'embedding_max': float(np.max(self.embeddings)),
            'sparsity': float(np.sum(self.embeddings == 0) / self.embeddings.size),
            'chunk_size_stats': {
                'mean_chars': float(np.mean([c.get('char_count', 0) for c in self.chunks])),
                'std_chars': float(np.std([c.get('char_count', 0) for c in self.chunks])),
                'min_chars': int(np.min([c.get('char_count', 0) for c in self.chunks])),
                'max_chars': int(np.max([c.get('char_count', 0) for c in self.chunks]))
            }
        }
        
        return stats


def main():
    """Main function to run the fast embedding generator."""
    parser = argparse.ArgumentParser(description='Generate fast vector embeddings for PDF chunks')
    parser.add_argument('chunks_file', help='Path to the JSON file containing chunks')
    parser.add_argument('--method', '-m', choices=['tfidf', 'count', 'binary'], 
                       default='tfidf', help='Embedding method to use')
    parser.add_argument('--max-features', type=int, default=5000, 
                       help='Maximum number of features for TF-IDF')
    parser.add_argument('--output', '-o', default='embeddings', 
                       help='Output file prefix (without extension)')
    parser.add_argument('--format', '-f', choices=['numpy', 'pickle', 'json'], 
                       default='json', help='Output format')
    parser.add_argument('--similarity-analysis', action='store_true', 
                       help='Generate similarity analysis')
    
    args = parser.parse_args()
    
    # Create embedding generator
    generator = FastEmbeddingGenerator(method=args.method, max_features=args.max_features)
    
    try:
        # Load chunks
        generator.load_chunks(args.chunks_file)
        
        # Generate embeddings
        embeddings = generator.generate_embeddings()
        
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
        print(f"Method: {stats['method']}")
        print(f"Number of chunks: {stats['num_chunks']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Vocabulary size: {stats['vocabulary_size']}")
        print(f"Average chunk size: {stats['chunk_size_stats']['mean_chars']:.1f} characters")
        print(f"Embedding value range: [{stats['embedding_min']:.3f}, {stats['embedding_max']:.3f}]")
        print(f"Embedding mean: {stats['embedding_mean']:.3f}")
        print(f"Embedding std: {stats['embedding_std']:.3f}")
        print(f"Sparsity: {stats['sparsity']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
