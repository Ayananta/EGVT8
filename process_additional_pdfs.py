#!/usr/bin/env python3
"""
Script to process additional PDF documents and integrate them with the existing RAG system.
Processes: Sale Agreement_redacted.pdf and DEED of Conveyance_redacted.pdf
"""

import json
import os
from improved_pdf_splitter import PDFProcessor
from fast_embedding_generator import FastEmbeddingGenerator
import faiss
import numpy as np

def process_additional_pdfs():
    """Process the two additional PDF files and integrate with existing system."""
    
    # Initialize PDF processor
    processor = PDFProcessor()
    
    # List of additional PDFs to process
    additional_pdfs = [
        "Data/Sale Agreement_redacted.pdf",
        "Data/DEED of Conveyance_redacted.pdf"
    ]
    
    all_chunks = []
    chunk_offset = 0
    
    # Process each PDF
    for pdf_path in additional_pdfs:
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            continue
            
        print(f"📄 Processing: {pdf_path}")
        
        # Extract filename for output
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        output_json = f"Data/{filename}_chunks.json"
        output_txt = f"Data/{filename}_chunks.txt"
        
        try:
            # Process PDF
            chunks = processor.process_pdf(pdf_path, output_json, output_txt)
            
            # Adjust chunk IDs to avoid conflicts with existing chunks
            for chunk in chunks:
                chunk['chunk_id'] += chunk_offset
                chunk['source_document'] = pdf_path
                chunk['document_type'] = 'legal_document'
            
            all_chunks.extend(chunks)
            chunk_offset = max([c['chunk_id'] for c in all_chunks]) + 1
            
            print(f"✅ Processed {len(chunks)} chunks from {pdf_path}")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {str(e)}")
            continue
    
    # Save combined chunks
    if all_chunks:
        combined_output = "Data/additional_documents_chunks.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Total chunks from additional documents: {len(all_chunks)}")
        print(f"💾 Saved combined chunks to: {combined_output}")
        
        return all_chunks
    
    return []

def generate_embeddings_for_additional_docs():
    """Generate embeddings for the additional document chunks."""
    
    # Load additional chunks
    chunks_file = "Data/additional_documents_chunks.json"
    if not os.path.exists(chunks_file):
        print("❌ Additional chunks file not found. Run process_additional_pdfs() first.")
        return None
    
    print("🔍 Generating embeddings for additional documents...")
    
    # Initialize embedding generator
    embedding_gen = FastEmbeddingGenerator()
    
    # Generate embeddings
    embeddings_file = "Data/additional_documents_embeddings.json"
    embedding_gen.generate_embeddings(chunks_file, embeddings_file)
    
    print(f"✅ Embeddings generated and saved to: {embeddings_file}")
    return embeddings_file

def update_faiss_database():
    """Update the existing FAISS database with new embeddings."""
    
    # Load existing FAISS database
    existing_index_file = "faiss_db/faiss_index.bin"
    existing_metadata_file = "faiss_db/metadata.json"
    
    if not os.path.exists(existing_index_file) or not os.path.exists(existing_metadata_file):
        print("❌ Existing FAISS database not found. Please run faiss_vector_store.py first.")
        return
    
    # Load existing data
    with open(existing_metadata_file, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    
    existing_index = faiss.read_index(existing_index_file)
    
    # Load new embeddings
    new_embeddings_file = "Data/additional_documents_embeddings.json"
    if not os.path.exists(new_embeddings_file):
        print("❌ New embeddings file not found. Run generate_embeddings_for_additional_docs() first.")
        return
    
    with open(new_embeddings_file, 'r', encoding='utf-8') as f:
        new_data = json.load(f)
    
    # Combine embeddings
    existing_embeddings = np.array(existing_data['embeddings'])
    new_embeddings = np.array(new_data['embeddings'])
    
    combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
    
    # Combine chunks
    combined_chunks = existing_data['chunks'] + new_data['chunks']
    
    # Update metadata
    combined_metadata = {
        'chunks': combined_chunks,
        'embeddings': combined_embeddings.tolist(),
        'metadata': {
            'total_chunks': len(combined_chunks),
            'embedding_dimension': combined_embeddings.shape[1],
            'feature_names': new_data['metadata']['feature_names'],
            'documents_processed': [
                'Elita Order_29.08.2025.pdf',
                'Sale Agreement_redacted.pdf',
                'DEED of Conveyance_redacted.pdf'
            ]
        }
    }
    
    # Create new FAISS index
    dimension = combined_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(combined_embeddings)
    index.add(combined_embeddings.astype('float32'))
    
    # Save updated index and metadata
    faiss.write_index(index, existing_index_file)
    
    with open(existing_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(combined_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ FAISS database updated with {len(new_data['chunks'])} additional chunks")
    print(f"📊 Total chunks in database: {len(combined_chunks)}")
    print(f"📄 Documents included: {combined_metadata['metadata']['documents_processed']}")

def main():
    """Main function to process additional PDFs and update the RAG system."""
    
    print("🚀 Processing Additional PDF Documents for RAG System")
    print("=" * 60)
    
    # Step 1: Process PDFs into chunks
    print("\n📄 Step 1: Processing PDFs into chunks...")
    chunks = process_additional_pdfs()
    
    if not chunks:
        print("❌ No chunks processed. Exiting.")
        return
    
    # Step 2: Generate embeddings
    print("\n🔍 Step 2: Generating embeddings...")
    embeddings_file = generate_embeddings_for_additional_docs()
    
    if not embeddings_file:
        print("❌ Embeddings generation failed. Exiting.")
        return
    
    # Step 3: Update FAISS database
    print("\n🗄️ Step 3: Updating FAISS database...")
    update_faiss_database()
    
    print("\n✅ Additional PDF processing completed successfully!")
    print("\n📋 Summary:")
    print(f"   • Processed {len(chunks)} new chunks")
    print(f"   • Generated embeddings: {embeddings_file}")
    print(f"   • Updated FAISS database with all documents")
    print(f"   • Ready for enhanced RAG queries!")

if __name__ == "__main__":
    main()
