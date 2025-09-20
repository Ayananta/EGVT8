#!/usr/bin/env python3
"""
Test script to debug embedding generation
"""

import json
import sys

def test_load_chunks():
    """Test loading chunks from JSON file."""
    try:
        print("Testing chunk loading...")
        with open("Data/elita_chunks_improved.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Successfully loaded {len(chunks)} chunks")
        
        # Print first chunk info
        if chunks:
            first_chunk = chunks[0]
            print(f"First chunk keys: {list(first_chunk.keys())}")
            print(f"First chunk content preview: {first_chunk.get('content', '')[:100]}...")
        
        return True
    except Exception as e:
        print(f"Error loading chunks: {e}")
        return False

def test_sentence_transformers():
    """Test sentence transformers import and basic functionality."""
    try:
        print("Testing sentence transformers...")
        from sentence_transformers import SentenceTransformer
        print("SentenceTransformers imported successfully")
        
        # Test loading a model
        print("Loading model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Model loaded successfully. Dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test encoding
        test_text = ["This is a test sentence."]
        embeddings = model.encode(test_text)
        print(f"Generated embedding shape: {embeddings.shape}")
        
        return True
    except Exception as e:
        print(f"Error with sentence transformers: {e}")
        return False

def test_numpy():
    """Test numpy import."""
    try:
        import numpy as np
        print("NumPy imported successfully")
        return True
    except Exception as e:
        print(f"Error with NumPy: {e}")
        return False

def test_sklearn():
    """Test sklearn import."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("Scikit-learn imported successfully")
        return True
    except Exception as e:
        print(f"Error with scikit-learn: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Dependencies ===")
    
    tests = [
        ("NumPy", test_numpy),
        ("Scikit-learn", test_sklearn),
        ("Sentence Transformers", test_sentence_transformers),
        ("Chunk Loading", test_load_chunks)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        results[test_name] = test_func()
    
    print(f"\n=== Test Results ===")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\nAll tests passed! Ready to generate embeddings.")
    else:
        print("\nSome tests failed. Please fix the issues before proceeding.")
