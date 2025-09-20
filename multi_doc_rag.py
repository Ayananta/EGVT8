#!/usr/bin/env python3
"""
Elita Garden Vista Tower 8 Case RAG System using Gemini AI.
This system can search across all three case documents and provide intelligent answers.
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from multi_doc_search import search_across_documents

class MultiDocumentRAG:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        print("Elita Garden Vista Tower 8 Case RAG System initialized successfully!")
        print("Documents available:")
        print("- Elita Order (273 chunks)")
        print("- Sale Agreement (34 chunks)")  
        print("- DEED of Conveyance (14 chunks)")
        print("- Total: 321 chunks across all case documents")
    
    def search_documents(self, query, top_k=5):
        """Search across all documents."""
        return search_across_documents(query, top_k)
    
    def generate_answer(self, query, context_chunks, max_context_length=4000):
        """Generate answer using Gemini with context from multiple documents."""
        
        # Prepare context from all relevant chunks
        context_parts = []
        total_length = 0
        
        for chunk in context_chunks:
            chunk_text = f"""
Document: {chunk['document']}
Source: {chunk['source_file']}
Section: {chunk.get('section', 'N/A')}
Page: {chunk.get('page', 'N/A')}
Content: {chunk['content']}
---
"""
            
            if total_length + len(chunk_text) > max_context_length:
                break
                
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a legal document analysis assistant. Based on the provided context from multiple legal documents, please answer the following question comprehensively and accurately.

Context from legal documents:
{context}

Question: {query}

Please provide a detailed answer based on the context above. If the information is not available in the provided context, please state that clearly. Include relevant quotes or references to specific sections when applicable.

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def ask_question(self, question, top_k=5, max_context_length=4000):
        """Ask a question and get an intelligent answer using RAG."""
        
        print(f"\nQuestion: {question}")
        print("="*60)
        
        # Search for relevant chunks
        print("Searching across all documents...")
        relevant_chunks = self.search_documents(question, top_k)
        
        if not relevant_chunks:
            return "No relevant information found in the documents for your question."
        
        print(f"Found {len(relevant_chunks)} relevant chunks:")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"{i}. {chunk['document']} - Score: {chunk['score']}")
        
        # Generate answer
        print("\nGenerating intelligent answer...")
        answer = self.generate_answer(question, relevant_chunks, max_context_length)
        
        return answer

def main():
    """Interactive RAG system."""
    
    try:
        rag = MultiDocumentRAG()
        
        print("\n" + "="*60)
        print("ELITA GARDEN VISTA TOWER 8 CASE RAG SYSTEM")
        print("="*60)
        print("Ask questions about any of the three case documents:")
        print("1. Elita Order")
        print("2. Sale Agreement") 
        print("3. DEED of Conveyance")
        print("\nType 'quit' to exit.")
        print("="*60)
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                answer = rag.ask_question(question)
                print(f"\nAnswer:\n{answer}")
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        print("Make sure GEMINI_API_KEY is set in your .env file")

if __name__ == "__main__":
    main()
