#!/usr/bin/env python3
"""
Combined search script for all three documents.
"""

import json
import numpy as np
import faiss
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

class CombinedSearcher:
    def __init__(self):
        self.indexes = {}
        self.metadata = {}
        self.vectorizers = {}
        
        # Load all document indexes
        docs = ['elita', 'sale', 'conveyance']
        for doc in docs:
            try:
                # Load index
                index_path = f"faiss_db_combined/{doc}_index.bin"
                self.indexes[doc] = faiss.read_index(index_path)
                
                # Load metadata
                metadata_path = f"faiss_db_combined/{doc}_metadata.json"
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata[doc] = json.load(f)
                
                # Recreate vectorizer from metadata
                chunks = self.metadata[doc]['chunks']
                if chunks:
                    # Use the vocabulary from the original embeddings
                    texts = [chunk['content'] for chunk in chunks]
                    self.vectorizers[doc] = TfidfVectorizer()
                    self.vectorizers[doc].fit(texts)
                
                print(f"Loaded {doc}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error loading {doc}: {e}")
    
    def search_query(self, query, top_k=5, doc_filter=None):
        """Search across all documents."""
        results = []
        
        docs_to_search = [doc_filter] if doc_filter else self.indexes.keys()
        
        for doc in docs_to_search:
            if doc not in self.indexes:
                continue
                
            try:
                # Transform query
                query_vector = self.vectorizers[doc].transform([query.lower()]).toarray()
                
                # Normalize for cosine similarity (manual normalization)
                query_vector = query_vector.astype(np.float32)
                query_norm = np.linalg.norm(query_vector)
                if query_norm > 0:
                    query_vector = query_vector / query_norm
                
                # Search
                scores, indices = self.indexes[doc].search(query_vector, min(top_k, len(self.metadata[doc]['chunks'])))
                
                # Add results
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0:  # Valid index
                        chunk = self.metadata[doc]['chunks'][idx]
                        results.append({
                            'document': doc,
                            'source_file': chunk.get('source_document', 'unknown'),
                            'chunk_id': chunk.get('chunk_id', idx),
                            'content': chunk['content'],
                            'score': float(score),
                            'section': chunk.get('section', ''),
                            'page': chunk.get('page', '')
                        })
                        
            except Exception as e:
                print(f"Error searching {doc}: {e}")
                import traceback
                traceback.print_exc()
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def search_by_document(self, query, doc_name, top_k=5):
        """Search within a specific document."""
        return self.search_query(query, top_k, doc_name)

def main():
    parser = argparse.ArgumentParser(description='Search across all documents')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--document', choices=['elita', 'sale', 'conveyance'], help='Search specific document only')
    
    args = parser.parse_args()
    
    searcher = CombinedSearcher()
    results = searcher.search_query(args.query, args.top_k, args.document)
    
    print(f"\nSearch results for: '{args.query}'")
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Document: {result['document']} ({result['source_file']})")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Section: {result['section']}")
        print(f"   Page: {result['page']}")
        print(f"   Content: {result['content'][:200]}...")
        print()

if __name__ == "__main__":
    main()
