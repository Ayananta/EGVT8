#!/usr/bin/env python3
"""
Script to update FAISS database with all three documents:
1. Elita Order (existing)
2. Sale Agreement (new)
3. DEED of Conveyance (new)
"""

import json
import numpy as np
import faiss
import os
from pathlib import Path

def load_embeddings(file_path):
    """Load embeddings from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def combine_all_embeddings():
    """Combine embeddings from all three documents."""
    
    # Load existing Elita embeddings
    print("Loading Elita Order embeddings...")
    elita_data = load_embeddings("Data/elita_embeddings_fast2.json")
    elita_embeddings = np.array(elita_data['embeddings'], dtype=np.float32)
    elita_chunks = elita_data['chunks']
    
    # Add document source to each chunk
    for chunk in elita_chunks:
        chunk['source_document'] = 'Elita Order_29.08.2025.pdf'
    
    print(f"Elita Order: {len(elita_chunks)} chunks, {elita_embeddings.shape[1]} dimensions")
    
    # Load Sale Agreement embeddings
    print("Loading Sale Agreement embeddings...")
    sale_data = load_embeddings("Data/sale_agreement_embeddings.json")
    sale_embeddings = np.array(sale_data['embeddings'], dtype=np.float32)
    sale_chunks = sale_data['chunks']
    
    # Add document source to each chunk
    for chunk in sale_chunks:
        chunk['source_document'] = 'Sale Agreement_redacted.pdf'
    
    print(f"Sale Agreement: {len(sale_chunks)} chunks, {sale_embeddings.shape[1]} dimensions")
    
    # Load Conveyance Deed embeddings
    print("Loading DEED of Conveyance embeddings...")
    conveyance_data = load_embeddings("Data/conveyance_deed_embeddings.json")
    conveyance_embeddings = np.array(conveyance_data['embeddings'], dtype=np.float32)
    conveyance_chunks = conveyance_data['chunks']
    
    # Add document source to each chunk
    for chunk in conveyance_chunks:
        chunk['source_document'] = 'DEED of Conveyance_redacted.pdf'
    
    print(f"DEED of Conveyance: {len(conveyance_chunks)} chunks, {conveyance_embeddings.shape[1]} dimensions")
    
    # Check if dimensions match - they might not due to different vocabularies
    print(f"\nDimension check:")
    print(f"Elita: {elita_embeddings.shape[1]}")
    print(f"Sale Agreement: {sale_embeddings.shape[1]}")
    print(f"Conveyance Deed: {conveyance_embeddings.shape[1]}")
    
    # For now, we'll use separate FAISS indexes for each document
    # This is because TF-IDF vocabularies are different for each document
    # In a production system, you'd want to use a shared vocabulary or sentence transformers
    
    return {
        'elita': {'embeddings': elita_embeddings, 'chunks': elita_chunks},
        'sale': {'embeddings': sale_embeddings, 'chunks': sale_chunks},
        'conveyance': {'embeddings': conveyance_embeddings, 'chunks': conveyance_chunks}
    }

def create_combined_faiss_index(all_data):
    """Create FAISS indexes for each document."""
    
    # Create directory for combined indexes
    os.makedirs("faiss_db_combined", exist_ok=True)
    
    results = {}
    
    for doc_name, data in all_data.items():
        embeddings = data['embeddings']
        chunks = data['chunks']
        
        print(f"\nCreating FAISS index for {doc_name}...")
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Create FAISS index (Flat index for simplicity)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Save index
        index_path = f"faiss_db_combined/{doc_name}_index.bin"
        faiss.write_index(index, index_path)
        
        # Save metadata
        metadata_path = f"faiss_db_combined/{doc_name}_metadata.json"
        metadata = {
            "index_type": "Flat",
            "dimension": dimension,
            "num_vectors": len(chunks),
            "document_source": chunks[0]['source_document'] if chunks else "unknown",
            "chunks": chunks
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        results[doc_name] = {
            'index_path': index_path,
            'metadata_path': metadata_path,
            'num_chunks': len(chunks)
        }
        
        print(f"Saved {doc_name} index: {index_path}")
        print(f"Saved {doc_name} metadata: {metadata_path}")
    
    return results

def create_combined_search_script():
    """Create a search script that can search across all documents."""
    
    search_script = '''#!/usr/bin/env python3
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
                
                # Normalize for cosine similarity
                faiss.normalize_L2(query_vector)
                
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
    
    print(f"\\nSearch results for: '{args.query}'")
    print(f"Found {len(results)} results\\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Document: {result['document']} ({result['source_file']})")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Section: {result['section']}")
        print(f"   Page: {result['page']}")
        print(f"   Content: {result['content'][:200]}...")
        print()

if __name__ == "__main__":
    main()
'''
    
    with open("combined_search.py", 'w', encoding='utf-8') as f:
        f.write(search_script)
    
    print("Created combined_search.py")

def main():
    print("=== Updating FAISS Database with All Documents ===")
    
    # Combine all embeddings
    all_data = combine_all_embeddings()
    
    # Create FAISS indexes
    results = create_combined_faiss_index(all_data)
    
    # Create combined search script
    create_combined_search_script()
    
    print("\n=== Summary ===")
    total_chunks = sum(result['num_chunks'] for result in results.values())
    print(f"Total chunks across all documents: {total_chunks}")
    
    for doc_name, result in results.items():
        print(f"{doc_name}: {result['num_chunks']} chunks")
    
    print("\nFiles created:")
    print("- faiss_db_combined/ directory with indexes and metadata")
    print("- combined_search.py for searching across all documents")
    
    print("\nTo search across all documents, use:")
    print("python combined_search.py 'your query'")
    print("python combined_search.py 'your query' --document elita")
    print("python combined_search.py 'your query' --top-k 10")

if __name__ == "__main__":
    main()
