#!/usr/bin/env python3
"""
Multi-document search using existing FAISS infrastructure.
This script searches across all three documents using the original FAISS setup.
"""

import json
import argparse
from pathlib import Path

def load_chunks(file_path):
    """Load chunks from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def search_across_documents(query, top_k=5):
    """Search across all documents using simple text matching."""
    
    # Load all chunk files
    documents = {
        'Elita Order': 'Data/elita_chunks_improved.json',
        'Sale Agreement': 'Data/sale_agreement_chunks',
        'DEED of Conveyance': 'Data/conveyance_deed_chunks'
    }
    
    all_results = []
    
    for doc_name, file_path in documents.items():
        try:
            chunks_data = load_chunks(file_path)
            # Handle different file formats
            if isinstance(chunks_data, list):
                chunks = chunks_data
            else:
                chunks = chunks_data.get('chunks', [])
            
            print(f"Searching {doc_name} ({len(chunks)} chunks)...")
            
            # Simple text search (case-insensitive)
            query_lower = query.lower()
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                if query_lower in content:
                    # Calculate simple relevance score based on query word frequency
                    query_words = query_lower.split()
                    score = sum(content.count(word) for word in query_words)
                    
                    if score > 0:
                        all_results.append({
                            'document': doc_name,
                            'source_file': Path(file_path).name,
                            'chunk_id': chunk.get('chunk_id', 'unknown'),
                            'content': chunk.get('content', ''),
                            'score': score,
                            'section': chunk.get('section', ''),
                            'page': chunk.get('page', '')
                        })
                        
        except Exception as e:
            print(f"Error loading {doc_name}: {e}")
    
    # Sort by score and return top results
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:top_k]

def main():
    parser = argparse.ArgumentParser(description='Search across all documents')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    print(f"Searching for: '{args.query}'")
    print("="*50)
    
    results = search_across_documents(args.query, args.top_k)
    
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Document: {result['document']}")
        print(f"   Source: {result['source_file']}")
        print(f"   Score: {result['score']}")
        print(f"   Section: {result['section']}")
        print(f"   Page: {result['page']}")
        print(f"   Chunk ID: {result['chunk_id']}")
        print(f"   Content: {result['content'][:200]}...")
        print()

if __name__ == "__main__":
    main()
