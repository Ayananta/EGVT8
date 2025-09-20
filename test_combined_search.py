#!/usr/bin/env python3
"""
Test script for combined search functionality.
"""

from combined_search import CombinedSearcher

def test_search():
    print("Testing combined search...")
    
    try:
        searcher = CombinedSearcher()
        
        # Test query
        query = "property sale agreement"
        results = searcher.search_query(query, top_k=5)
        
        print(f"\nSearch results for: '{query}'")
        print(f"Found {len(results)} results\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Document: {result['document']} ({result['source_file']})")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Section: {result['section']}")
            print(f"   Page: {result['page']}")
            print(f"   Content: {result['content'][:150]}...")
            print()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    # Test individual document search
    print("\n" + "="*50)
    print("Testing individual document searches...")
    
    try:
        searcher = CombinedSearcher()
        
        # Test each document individually
        for doc in ['elita', 'sale', 'conveyance']:
            print(f"\nTesting {doc} document:")
            try:
                results = searcher.search_by_document("property", doc, top_k=3)
                print(f"Found {len(results)} results for {doc}")
                if results:
                    print(f"Top result score: {results[0]['score']:.4f}")
            except Exception as e:
                print(f"Error searching {doc}: {e}")
                
    except Exception as e:
        print(f"Error in individual testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search()
