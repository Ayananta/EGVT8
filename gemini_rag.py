#!/usr/bin/env python3
"""
Gemini RAG System for PDF Chunks

This script implements Retrieval-Augmented Generation using Gemini AI
with FAISS vector database for intelligent question answering about your PDF content.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from faiss_search import FAISSSearcher
import time

# Load environment variables
load_dotenv()


class GeminiRAG:
    """Retrieval-Augmented Generation system using Gemini AI and FAISS."""
    
    def __init__(self, faiss_db_dir: str = 'faiss_db', model_name: str = 'gemini-1.5-flash'):
        """
        Initialize Gemini RAG system.
        
        Args:
            faiss_db_dir: Path to FAISS database directory
            model_name: Gemini model to use
        """
        self.faiss_db_dir = faiss_db_dir
        self.model_name = model_name
        self.searcher = None
        self.model = None
        self.chat_session = None
        
        # Initialize components
        self._setup_gemini()
        self._setup_faiss()
        
    def _setup_gemini(self):
        """Setup Gemini AI model."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("⚠️  GEMINI_API_KEY not found in environment variables.")
            print("Please set your Gemini API key:")
            print("1. Get API key from: https://makersuite.google.com/app/apikey")
            print("2. Create .env file with: GEMINI_API_KEY=your_api_key_here")
            print("3. Or set environment variable: set GEMINI_API_KEY=your_api_key_here")
            return
        
        try:
            genai.configure(api_key=api_key)
            
            # Configure generation settings
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            print(f"✅ Gemini {self.model_name} model initialized")
            
        except Exception as e:
            print(f"❌ Error initializing Gemini: {e}")
            self.model = None
    
    def _setup_faiss(self):
        """Setup FAISS vector search."""
        try:
            self.searcher = FAISSSearcher(self.faiss_db_dir)
            self.searcher.load_database()
            print(f"✅ FAISS database loaded with {self.searcher.index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error loading FAISS database: {e}")
            self.searcher = None
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from FAISS database."""
        if not self.searcher:
            return []
        
        try:
            results = self.searcher.search_by_text(query, top_k)
            return results
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context for Gemini."""
        if not chunks:
            return "No relevant information found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"--- Source {i} (Chunk {chunk['chunk_id']}) ---")
            if chunk.get('header'):
                context_parts.append(f"Section: {chunk['header']}")
            context_parts.append(f"Content: {chunk['content']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Gemini with provided context."""
        if not self.model:
            return "❌ Gemini model not available. Please check your API key."
        
        prompt = f"""You are a legal document assistant helping users understand a legal judgment about apartment ownership and construction disputes. 

Based on the following context from the legal document, please answer the user's question accurately and comprehensively. If the context doesn't contain enough information to answer the question, please say so.

Context from the legal document:
{context}

User Question: {question}

Please provide a clear, accurate answer based on the provided context. Include relevant legal references and citations when possible."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Error generating answer: {e}"
    
    def ask_question(self, question: str, top_k: int = 5, show_sources: bool = True) -> Dict[str, Any]:
        """Ask a question using RAG pipeline."""
        print(f"\n🤔 Question: {question}")
        print("=" * 60)
        
        # Step 1: Retrieve relevant chunks
        print("🔍 Retrieving relevant information...")
        start_time = time.time()
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k)
        retrieval_time = time.time() - start_time
        
        if not relevant_chunks:
            return {
                'question': question,
                'answer': "❌ No relevant information found in the document.",
                'sources': [],
                'retrieval_time': retrieval_time,
                'generation_time': 0
            }
        
        print(f"✅ Found {len(relevant_chunks)} relevant chunks ({retrieval_time:.2f}s)")
        
        # Step 2: Format context
        context = self.format_context(relevant_chunks)
        
        # Step 3: Generate answer
        print("🧠 Generating answer with Gemini...")
        start_time = time.time()
        answer = self.generate_answer(question, context)
        generation_time = time.time() - start_time
        
        print(f"✅ Answer generated ({generation_time:.2f}s)")
        
        # Step 4: Prepare response
        response = {
            'question': question,
            'answer': answer,
            'sources': relevant_chunks,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + generation_time
        }
        
        # Display results
        print(f"\n💡 Answer:")
        print("-" * 40)
        print(answer)
        
        if show_sources:
            print(f"\n📚 Sources ({len(relevant_chunks)} chunks):")
            print("-" * 40)
            for i, chunk in enumerate(relevant_chunks, 1):
                print(f"{i}. Chunk {chunk['chunk_id']} (similarity: {chunk['similarity']:.4f})")
                if chunk.get('header'):
                    print(f"   Section: {chunk['header']}")
                print(f"   Content: {chunk['content'][:100]}...")
                print()
        
        print(f"\n⏱️  Performance:")
        print(f"   Retrieval: {retrieval_time:.2f}s")
        print(f"   Generation: {generation_time:.2f}s")
        print(f"   Total: {response['total_time']:.2f}s")
        
        return response
    
    def start_chat_session(self):
        """Start an interactive chat session."""
        if not self.model:
            print("❌ Gemini model not available. Please check your API key.")
            return
        
        print("\n" + "=" * 70)
        print("🤖 GEMINI RAG CHAT - Ask questions about your legal document")
        print("=" * 70)
        print("Commands:")
        print("  - Type your question and press Enter")
        print("  - Type 'quit' or 'exit' to end the session")
        print("  - Type 'sources' to see last answer sources")
        print("  - Type 'help' for more commands")
        print("=" * 70)
        
        last_response = None
        
        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n📖 Available commands:")
                    print("  - Ask any question about the legal document")
                    print("  - 'sources' - Show sources from last answer")
                    print("  - 'quit' - End the session")
                    print("  - 'help' - Show this help")
                    continue
                
                elif user_input.lower() == 'sources':
                    if last_response and last_response.get('sources'):
                        print(f"\n📚 Sources from last answer:")
                        for i, chunk in enumerate(last_response['sources'], 1):
                            print(f"{i}. Chunk {chunk['chunk_id']} - {chunk['header'][:50]}...")
                    else:
                        print("No previous answer sources available.")
                    continue
                
                elif not user_input:
                    continue
                
                # Process the question
                response = self.ask_question(user_input, top_k=5)
                last_response = response
                
            except KeyboardInterrupt:
                print("\n👋 Chat session ended.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def batch_questions(self, questions: List[str], output_file: Optional[str] = None):
        """Process multiple questions in batch."""
        print(f"\n📋 Processing {len(questions)} questions...")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing question...")
            response = self.ask_question(question, show_sources=False)
            results.append(response)
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        
        return results


def main():
    """Main function for Gemini RAG system."""
    parser = argparse.ArgumentParser(description='Gemini RAG System for PDF Chunks')
    parser.add_argument('--faiss-db', '-d', default='faiss_db',
                       help='FAISS database directory')
    parser.add_argument('--model', '-m', default='gemini-1.5-flash',
                       choices=['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro'],
                       help='Gemini model to use')
    parser.add_argument('--question', '-q', help='Single question to ask')
    parser.add_argument('--chat', action='store_true', help='Start interactive chat session')
    parser.add_argument('--batch', help='File containing questions (one per line)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of relevant chunks to retrieve')
    parser.add_argument('--output', '-o', help='Output file for batch results')
    
    args = parser.parse_args()
    
    # Create RAG system
    rag = GeminiRAG(faiss_db_dir=args.faiss_db, model_name=args.model)
    
    if not rag.model:
        print("❌ Cannot proceed without Gemini model. Please check your API key.")
        return 1
    
    try:
        if args.question:
            # Single question
            rag.ask_question(args.question, top_k=args.top_k)
        
        elif args.chat:
            # Interactive chat
            rag.start_chat_session()
        
        elif args.batch:
            # Batch processing
            with open(args.batch, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
            
            rag.batch_questions(questions, args.output)
        
        else:
            print("Please specify an operation:")
            print("  --question 'text'    : Ask a single question")
            print("  --chat               : Start interactive chat session")
            print("  --batch FILE         : Process questions from file")
            print("  --help               : Show this help")
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
