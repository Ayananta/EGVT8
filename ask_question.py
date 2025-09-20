#!/usr/bin/env python3
"""
Simple Question Interface for Gemini RAG
"""

import os
import sys
from dotenv import load_dotenv

# Set API key
os.environ['GEMINI_API_KEY'] = 'AIzaSyDk0JhMnS5S41RAsV6f8v3qCuS3PFGnn-Y'

# Import and run the RAG system
if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"🤖 Asking: {question}")
        
        # Import the RAG system
        from gemini_rag import GeminiRAG
        
        # Create RAG instance
        rag = GeminiRAG()
        
        # Ask the question
        if rag.model and rag.searcher:
            response = rag.ask_question(question, top_k=5)
            print(f"\n✅ Answer: {response['answer']}")
        else:
            print("❌ RAG system not properly initialized")
    else:
        print("Usage: python ask_question.py 'Your question here'")
        print("Example: python ask_question.py 'What is this legal case about?'")
