#!/usr/bin/env python3
"""
FAISS Database Manager

This script provides comprehensive management of FAISS vector databases.
"""

import json
import numpy as np
import faiss
import argparse
from pathlib import Path
from typing import Dict, Any, List
import time


class FAISSManager:
    """Comprehensive FAISS database manager."""
    
    def __init__(self):
        self.index_types = {
            'flat': {
                'description': 'Exact search, most accurate but slower',
                'use_case': 'Small datasets, exact similarity needed'
            },
            'ivf': {
                'description': 'Inverted file index, faster approximate search',
                'use_case': 'Medium to large datasets, good speed/accuracy balance'
            },
            'hnsw': {
                'description': 'Hierarchical navigable small world',
                'use_case': 'Large datasets, fast approximate search'
            }
        }
    
    def create_index_comparison(self, embeddings_file: str, output_dir: str = 'faiss_comparison'):
        """Create multiple index types for comparison."""
        print("Creating FAISS index comparison...")
        
        # Load embeddings
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        embeddings = np.array(data['embeddings']).astype('float32')
        faiss.normalize_L2(embeddings)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        for index_type in ['flat', 'ivf', 'hnsw']:
            print(f"\n--- Creating {index_type.upper()} index ---")
            
            start_time = time.time()
            
            if index_type == 'flat':
                index = faiss.IndexFlatIP(embeddings.shape[1])
                
            elif index_type == 'ivf':
                nlist = min(100, max(10, len(embeddings) // 10))
                quantizer = faiss.IndexFlatIP(embeddings.shape[1])
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
                index.train(embeddings)
                
            elif index_type == 'hnsw':
                index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
            
            # Add vectors
            index.add(embeddings)
            
            build_time = time.time() - start_time
            
            # Test search performance
            test_queries = embeddings[:5]  # Use first 5 vectors as test queries
            search_start = time.time()
            
            for query in test_queries:
                _ = index.search(query.reshape(1, -1), 10)
            
            search_time = (time.time() - search_start) / len(test_queries)
            
            # Save index
            index_path = output_dir / f'{index_type}_index.bin'
            faiss.write_index(index, str(index_path))
            
            # Save metadata
            metadata = {
                'index_type': index_type,
                'num_vectors': index.ntotal,
                'dimension': embeddings.shape[1],
                'build_time': build_time,
                'avg_search_time': search_time,
                'description': self.index_types[index_type]['description'],
                'use_case': self.index_types[index_type]['use_case']
            }
            
            metadata_path = output_dir / f'{index_type}_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            results[index_type] = metadata
            
            print(f"✅ {index_type.upper()} index created")
            print(f"   Build time: {build_time:.3f}s")
            print(f"   Avg search time: {search_time*1000:.2f}ms")
            print(f"   Vectors: {index.ntotal}")
        
        # Save comparison summary
        summary_path = output_dir / 'comparison_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Index Comparison Summary ===")
        print(f"Results saved to: {output_dir}")
        print(f"\nIndex Performance:")
        for index_type, metadata in results.items():
            print(f"  {index_type.upper():4}: {metadata['build_time']:.3f}s build, {metadata['avg_search_time']*1000:.2f}ms search")
        
        return results
    
    def benchmark_search(self, index_path: str, embeddings_file: str, num_queries: int = 100):
        """Benchmark search performance."""
        print(f"Benchmarking search performance...")
        
        # Load index
        index = faiss.read_index(index_path)
        
        # Load test queries
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        embeddings = np.array(data['embeddings']).astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Use random queries for benchmarking
        np.random.seed(42)
        query_indices = np.random.choice(len(embeddings), min(num_queries, len(embeddings)), replace=False)
        test_queries = embeddings[query_indices]
        
        # Benchmark different k values
        k_values = [1, 5, 10, 20]
        results = {}
        
        for k in k_values:
            times = []
            similarities = []
            
            start_time = time.time()
            
            for query in test_queries:
                query_start = time.time()
                distances, indices = index.search(query.reshape(1, -1), k)
                query_time = time.time() - query_start
                
                times.append(query_time)
                similarities.extend(distances[0])
            
            total_time = time.time() - start_time
            
            results[k] = {
                'avg_time_ms': np.mean(times) * 1000,
                'total_time': total_time,
                'queries_per_second': len(test_queries) / total_time,
                'avg_similarity': np.mean(similarities),
                'max_similarity': np.max(similarities),
                'min_similarity': np.min(similarities)
            }
        
        print(f"\n=== Search Benchmark Results ===")
        print(f"Index: {Path(index_path).name}")
        print(f"Test queries: {len(test_queries)}")
        print(f"\nPerformance by k value:")
        
        for k, stats in results.items():
            print(f"  k={k:2d}: {stats['avg_time_ms']:.2f}ms/query, {stats['queries_per_second']:.1f} qps")
            print(f"         Similarity: {stats['avg_similarity']:.4f} avg, {stats['max_similarity']:.4f} max")
        
        return results
    
    def optimize_index(self, input_index_path: str, output_index_path: str, optimization_type: str = 'quantization'):
        """Optimize existing index for better performance."""
        print(f"Optimizing index...")
        
        # Load original index
        index = faiss.read_index(input_index_path)
        
        if optimization_type == 'quantization':
            # Apply product quantization
            dimension = index.d
            if dimension % 4 == 0:
                pq_index = faiss.IndexPQ(dimension, 8, 8)  # 8 subquantizers, 8 bits each
                pq_index.train(index.reconstruct_n(0, index.ntotal))
                pq_index.add(index.reconstruct_n(0, index.ntotal))
                
                faiss.write_index(pq_index, output_index_path)
                print(f"✅ Quantized index saved to: {output_index_path}")
                print(f"   Compression ratio: {pq_index.code_size / (dimension * 4):.2f}")
            else:
                print(f"❌ Dimension {dimension} not suitable for PQ quantization")
        
        elif optimization_type == 'ivf_optimization':
            # Convert flat index to IVF for better search speed
            if isinstance(index, faiss.IndexFlat):
                dimension = index.d
                nlist = min(100, max(10, index.ntotal // 10))
                
                quantizer = faiss.IndexFlatIP(dimension)
                ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                
                # Train and add vectors
                vectors = index.reconstruct_n(0, index.ntotal)
                ivf_index.train(vectors)
                ivf_index.add(vectors)
                
                faiss.write_index(ivf_index, output_index_path)
                print(f"✅ IVF optimized index saved to: {output_index_path}")
            else:
                print(f"❌ Index is not a flat index, cannot optimize with IVF")
    
    def analyze_index(self, index_path: str):
        """Analyze FAISS index properties."""
        index = faiss.read_index(index_path)
        
        print(f"=== FAISS Index Analysis ===")
        print(f"Index type: {type(index).__name__}")
        print(f"Dimension: {index.d}")
        print(f"Number of vectors: {index.ntotal}")
        
        if hasattr(index, 'metric_type'):
            metric_types = {
                faiss.METRIC_INNER_PRODUCT: "Inner Product (Cosine Similarity)",
                faiss.METRIC_L2: "L2 Distance",
                faiss.METRIC_L1: "L1 Distance"
            }
            print(f"Metric type: {metric_types.get(index.metric_type, 'Unknown')}")
        
        if hasattr(index, 'nlist'):
            print(f"Number of clusters (nlist): {index.nlist}")
        
        if hasattr(index, 'nprobe'):
            print(f"Number of probes: {index.nprobe}")
        
        # Memory usage estimation
        if hasattr(index, 'code_size'):
            total_size = index.ntotal * index.code_size
            print(f"Estimated memory usage: {total_size / (1024*1024):.2f} MB")
        else:
            estimated_size = index.ntotal * index.d * 4  # Assuming float32
            print(f"Estimated memory usage: {estimated_size / (1024*1024):.2f} MB")


def main():
    """Main function for FAISS manager."""
    parser = argparse.ArgumentParser(description='FAISS Database Manager')
    parser.add_argument('--embeddings', '-e', default='Data/elita_embeddings_fast.json',
                       help='Path to embeddings JSON file')
    parser.add_argument('--compare', action='store_true',
                       help='Create comparison of different index types')
    parser.add_argument('--benchmark', help='Benchmark search performance of index file')
    parser.add_argument('--optimize', help='Optimize index file')
    parser.add_argument('--analyze', help='Analyze index file')
    parser.add_argument('--output-dir', '-o', default='faiss_comparison',
                       help='Output directory for comparison results')
    parser.add_argument('--num-queries', type=int, default=100,
                       help='Number of queries for benchmarking')
    
    args = parser.parse_args()
    
    manager = FAISSManager()
    
    try:
        if args.compare:
            manager.create_index_comparison(args.embeddings, args.output_dir)
        
        elif args.benchmark:
            manager.benchmark_search(args.benchmark, args.embeddings, args.num_queries)
        
        elif args.optimize:
            output_path = args.optimize.replace('.bin', '_optimized.bin')
            manager.optimize_index(args.optimize, output_path)
        
        elif args.analyze:
            manager.analyze_index(args.analyze)
        
        else:
            print("Please specify an operation:")
            print("  --compare     : Create comparison of different index types")
            print("  --benchmark FILE : Benchmark search performance")
            print("  --optimize FILE  : Optimize index file")
            print("  --analyze FILE   : Analyze index properties")
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
