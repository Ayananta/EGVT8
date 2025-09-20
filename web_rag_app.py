#!/usr/bin/env python3
"""
Flask Web Application for Elita Garden Vista Tower 8 Case RAG System
"""

from flask import Flask, render_template, request, jsonify, session
import json
import os
from datetime import datetime
from multi_doc_rag import MultiDocumentRAG
import traceback

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Global RAG instance
rag_system = None

def initialize_rag():
    """Initialize the RAG system."""
    global rag_system
    try:
        rag_system = MultiDocumentRAG()
        return True
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query_rag():
    """API endpoint to handle RAG queries."""
    try:
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized'
            }), 500

        data = request.get_json()
        question = data.get('question', '').strip()
        top_k = data.get('top_k', 5)
        max_context = data.get('max_context', 4000)

        if not question:
            return jsonify({
                'success': False,
                'error': 'No question provided'
            }), 400

        # Search for relevant chunks
        relevant_chunks = rag_system.search_documents(question, top_k)
        
        if not relevant_chunks:
            return jsonify({
                'success': True,
                'answer': 'No relevant information found in the documents for your question.',
                'chunks': [],
                'query': question,
                'timestamp': datetime.now().isoformat()
            })

        # Generate answer
        answer = rag_system.generate_answer(question, relevant_chunks, max_context)

        # Store query in session for history
        if 'query_history' not in session:
            session['query_history'] = []
        
        session['query_history'].append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'chunks_found': len(relevant_chunks)
        })

        # Keep only last 10 queries
        if len(session['query_history']) > 10:
            session['query_history'] = session['query_history'][-10:]

        return jsonify({
            'success': True,
            'answer': answer,
            'chunks': relevant_chunks,
            'query': question,
            'timestamp': datetime.now().isoformat(),
            'chunks_count': len(relevant_chunks)
        })

    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing query: {str(e)}'
        }), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """API endpoint to search documents without AI generation."""
    try:
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized'
            }), 500

        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 10)

        if not query:
            return jsonify({
                'success': False,
                'error': 'No search query provided'
            }), 400

        results = rag_system.search_documents(query, top_k)

        return jsonify({
            'success': True,
            'results': results,
            'query': query,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in search: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error searching documents: {str(e)}'
        }), 500

@app.route('/api/history')
def get_history():
    """Get query history."""
    history = session.get('query_history', [])
    return jsonify({
        'success': True,
        'history': history
    })

@app.route('/api/status')
def get_status():
    """Get system status."""
    try:
        if rag_system:
            return jsonify({
                'success': True,
                'status': 'ready',
                'documents': {
                    'Elita Order': 273,
                    'Sale Agreement': 34,
                    'DEED of Conveyance': 14
                },
                'total_chunks': 321
            })
        else:
            return jsonify({
                'success': False,
                'status': 'not_ready',
                'error': 'RAG system not initialized'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/sample_questions')
def get_sample_questions():
    """Get sample questions for users."""
    return jsonify({
        'success': True,
        'questions': [
            "What are the key terms of the Elita Garden Vista Tower 8 sale agreement?",
            "What are the builder's obligations regarding possession of Tower 8?",
            "What happens if the builder fails to deliver Tower 8 on time?",
            "What are the payment terms for Elita Garden Vista Tower 8?",
            "What legal protections exist for Tower 8 buyers?",
            "What are the consequences of builder delays for Tower 8?",
            "What are the buyer's rights under the Tower 8 agreement?",
            "What are the penalties for late payments in Tower 8?",
            "What are the maintenance charges for Elita Garden Vista Tower 8?",
            "What are the dispute resolution mechanisms for Tower 8?"
        ]
    })

if __name__ == '__main__':
    print("Initializing RAG system...")
    if initialize_rag():
        print("RAG system initialized successfully!")
        print("Starting web server...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize RAG system. Please check your configuration.")
