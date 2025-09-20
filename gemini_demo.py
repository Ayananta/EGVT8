#!/usr/bin/env python3
"""
Gemini RAG Demo

This script demonstrates the Gemini RAG system capabilities.
"""

import json
import os
from pathlib import Path
from faiss_search import FAISSSearcher


def demo_rag_pipeline():
    """Demonstrate RAG pipeline components."""
    print("🤖 GEMINI RAG SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Check if FAISS database exists
    if not Path('faiss_db/faiss_index.bin').exists():
        print("❌ FAISS database not found. Please run the embedding generation first.")
        return
    
    # Initialize FAISS searcher
    try:
        searcher = FAISSSearcher('faiss_db')
        searcher.load_database()
        print("✅ FAISS database loaded successfully")
    except Exception as e:
        print(f"❌ Error loading FAISS database: {e}")
        return
    
    # Demo questions
    demo_questions = [
        "What is this legal case about?",
        "What are the main arguments of the appellants?",
        "What does the West Bengal Apartment Ownership Act say?",
        "What are the duties of a promoter?",
        "What was the court's final decision?"
    ]
    
    print(f"\n📋 Demo Questions:")
    for i, question in enumerate(demo_questions, 1):
        print(f"{i}. {question}")
    
    print(f"\n🔍 RAG Pipeline Demonstration:")
    print("=" * 40)
    
    for i, question in enumerate(demo_questions[:2], 1):  # Demo first 2 questions
        print(f"\n[{i}] Question: {question}")
        print("-" * 50)
        
        # Step 1: Retrieve relevant chunks
        print("Step 1: Retrieving relevant chunks...")
        try:
            relevant_chunks = searcher.search_by_text(question, top_k=3)
            
            if relevant_chunks:
                print(f"✅ Found {len(relevant_chunks)} relevant chunks:")
                for j, chunk in enumerate(relevant_chunks, 1):
                    print(f"   {j}. Chunk {chunk['chunk_id']} (similarity: {chunk['similarity']:.4f})")
                    print(f"      Header: {chunk['header'][:60]}...")
                    print(f"      Content: {chunk['content'][:80]}...")
            else:
                print("❌ No relevant chunks found")
                continue
                
        except Exception as e:
            print(f"❌ Error retrieving chunks: {e}")
            continue
        
        # Step 2: Format context (what would be sent to Gemini)
        print(f"\nStep 2: Context formatting (for Gemini):")
        context = format_context_for_demo(relevant_chunks)
        print(f"Context length: {len(context)} characters")
        print(f"Context preview: {context[:200]}...")
        
        # Step 3: Show what Gemini would receive
        print(f"\nStep 3: Gemini prompt structure:")
        prompt = create_demo_prompt(question, context)
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Prompt preview:")
        print(prompt[:300] + "...")
        
        print("\n" + "="*60)
    
    # Show system capabilities
    print(f"\n🎯 SYSTEM CAPABILITIES:")
    print("✅ Semantic search using FAISS vector database")
    print("✅ Context-aware question answering")
    print("✅ Legal document analysis")
    print("✅ Source citation and referencing")
    print("✅ Interactive chat interface")
    print("✅ Batch question processing")
    
    # Show file structure
    print(f"\n📁 SYSTEM FILES:")
    files = [
        "gemini_rag.py - Main RAG system",
        "faiss_search.py - Vector search functionality", 
        "faiss_db/ - Vector database",
        "Data/elita_embeddings_fast.json - Embeddings data",
        "setup_gemini.py - API key setup",
        "gemini_demo.py - This demonstration"
    ]
    
    for file in files:
        print(f"   • {file}")
    
    # Show usage examples
    print(f"\n🚀 USAGE EXAMPLES:")
    print("1. Setup API key:")
    print("   python setup_gemini.py")
    print("\n2. Ask a question:")
    print("   python gemini_rag.py --question 'What is this case about?'")
    print("\n3. Interactive chat:")
    print("   python gemini_rag.py --chat")
    print("\n4. Batch processing:")
    print("   python gemini_rag.py --batch example_questions.txt")


def format_context_for_demo(chunks):
    """Format chunks for demo context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"--- Source {i} (Chunk {chunk['chunk_id']}) ---")
        if chunk.get('header'):
            context_parts.append(f"Section: {chunk['header']}")
        context_parts.append(f"Content: {chunk['content']}")
        context_parts.append("")
    return "\n".join(context_parts)


def create_demo_prompt(question, context):
    """Create demo prompt structure."""
    return f"""You are a legal document assistant helping users understand a legal judgment about apartment ownership and construction disputes.

Based on the following context from the legal document, please answer the user's question accurately and comprehensively.

Context from the legal document:
{context}

User Question: {question}

Please provide a clear, accurate answer based on the provided context."""


def check_system_status():
    """Check system status and requirements."""
    print("🔍 SYSTEM STATUS CHECK")
    print("=" * 30)
    
    # Check FAISS database
    faiss_files = [
        'faiss_db/faiss_index.bin',
        'faiss_db/metadata.json'
    ]
    
    print("FAISS Database:")
    for file in faiss_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} (missing)")
    
    # Check embeddings
    embeddings_file = 'Data/elita_embeddings_fast.json'
    if Path(embeddings_file).exists():
        size = Path(embeddings_file).stat().st_size
        print(f"  ✅ {embeddings_file} ({size:,} bytes)")
    else:
        print(f"  ❌ {embeddings_file} (missing)")
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"  ✅ GEMINI_API_KEY (configured)")
    else:
        print(f"  ⚠️  GEMINI_API_KEY (not set)")
        print(f"     Run: python setup_gemini.py")
    
    print()


if __name__ == "__main__":
    check_system_status()
    demo_rag_pipeline()
