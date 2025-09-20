#!/usr/bin/env python3
"""
Test Gemini Integration

This script tests the Gemini RAG integration without requiring API key.
"""

import os
import sys
from pathlib import Path


def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import google.generativeai as genai
        print("✅ google.generativeai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import google.generativeai: {e}")
        return False
    
    try:
        from faiss_search import FAISSSearcher
        print("✅ FAISSSearcher imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import FAISSSearcher: {e}")
        return False
    
    return True


def test_faiss_database():
    """Test FAISS database availability."""
    print("\n🗄️ Testing FAISS database...")
    
    faiss_files = [
        'faiss_db/faiss_index.bin',
        'faiss_db/metadata.json'
    ]
    
    all_exist = True
    for file in faiss_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file} exists ({size:,} bytes)")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist


def test_embeddings_data():
    """Test embeddings data availability."""
    print("\n📊 Testing embeddings data...")
    
    embeddings_file = 'Data/elita_embeddings_fast.json'
    if Path(embeddings_file).exists():
        size = Path(embeddings_file).stat().st_size
        print(f"✅ {embeddings_file} exists ({size:,} bytes)")
        
        # Try to load and check structure
        try:
            import json
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"   - Method: {data.get('method', 'Unknown')}")
            print(f"   - Chunks: {len(data.get('chunks', []))}")
            print(f"   - Embeddings: {len(data.get('embeddings', []))}")
            print(f"   - Dimension: {data.get('metadata', {}).get('embedding_dimension', 'Unknown')}")
            
            return True
        except Exception as e:
            print(f"❌ Error reading embeddings: {e}")
            return False
    else:
        print(f"❌ {embeddings_file} missing")
        return False


def test_search_functionality():
    """Test FAISS search functionality."""
    print("\n🔍 Testing search functionality...")
    
    try:
        from faiss_search import FAISSSearcher
        
        searcher = FAISSSearcher('faiss_db')
        searcher.load_database()
        
        # Test a simple search
        results = searcher.search_by_text("apartment ownership", top_k=3)
        
        if results:
            print(f"✅ Search successful - found {len(results)} results")
            print(f"   Top result: Chunk {results[0]['chunk_id']} (similarity: {results[0]['similarity']:.4f})")
            return True
        else:
            print("❌ Search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False


def test_gemini_configuration():
    """Test Gemini configuration (without API key)."""
    print("\n🤖 Testing Gemini configuration...")
    
    try:
        import google.generativeai as genai
        
        # Check if API key is set
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            print("✅ GEMINI_API_KEY is set")
            
            # Test basic configuration
            genai.configure(api_key=api_key)
            print("✅ Gemini configuration successful")
            
            # Try to create a model (this will fail if API key is invalid)
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                print("✅ Gemini model creation successful")
                return True
            except Exception as e:
                print(f"⚠️  Model creation failed (API key might be invalid): {e}")
                return False
        else:
            print("⚠️  GEMINI_API_KEY not set")
            print("   Run: python setup_gemini.py")
            return False
            
    except Exception as e:
        print(f"❌ Gemini configuration failed: {e}")
        return False


def show_system_summary():
    """Show system summary."""
    print("\n" + "=" * 60)
    print("📋 GEMINI RAG SYSTEM SUMMARY")
    print("=" * 60)
    
    # Check all components
    imports_ok = test_imports()
    faiss_ok = test_faiss_database()
    embeddings_ok = test_embeddings_data()
    search_ok = test_search_functionality()
    gemini_ok = test_gemini_configuration()
    
    print(f"\n🎯 SYSTEM STATUS:")
    print(f"   Imports: {'✅' if imports_ok else '❌'}")
    print(f"   FAISS DB: {'✅' if faiss_ok else '❌'}")
    print(f"   Embeddings: {'✅' if embeddings_ok else '❌'}")
    print(f"   Search: {'✅' if search_ok else '❌'}")
    print(f"   Gemini: {'✅' if gemini_ok else '⚠️'}")
    
    if all([imports_ok, faiss_ok, embeddings_ok, search_ok]):
        print(f"\n🎉 CORE SYSTEM READY!")
        if gemini_ok:
            print("   Full RAG functionality available")
        else:
            print("   RAG ready - just need Gemini API key")
            print("   Run: python setup_gemini.py")
    else:
        print(f"\n❌ System not ready - please fix the issues above")
    
    print(f"\n🚀 NEXT STEPS:")
    if not gemini_ok:
        print("1. Set up Gemini API key: python setup_gemini.py")
    print("2. Try the demo: python gemini_demo.py")
    print("3. Ask a question: python gemini_rag.py --question 'What is this case about?'")
    print("4. Start chat: python gemini_rag.py --chat")


if __name__ == "__main__":
    show_system_summary()
