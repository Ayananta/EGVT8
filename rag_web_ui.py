#!/usr/bin/env python3
"""
RAG Web UI - A modern web interface for asking questions to the RAG system
"""

import os
import json
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import tempfile
from gemini_rag import GeminiRAGSystem

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Global RAG system instance
rag_system = None

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_system
    try:
        rag_system = GeminiRAGSystem()
        print("✅ RAG system initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint to ask questions"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Please provide a question'
            })
        
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized. Please check your setup.'
            })
        
        # Get answer from RAG system
        answer = rag_system.ask_question(question)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer['answer'],
            'sources': answer.get('sources', []),
            'confidence': answer.get('confidence', 'N/A')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing question: {str(e)}'
        })

@app.route('/api/status')
def get_status():
    """Get system status"""
    try:
        if rag_system:
            # Try to get some basic info about the system
            status = {
                'initialized': True,
                'faiss_loaded': True,
                'gemini_connected': True,
                'message': 'RAG system is ready'
            }
        else:
            status = {
                'initialized': False,
                'faiss_loaded': False,
                'gemini_connected': False,
                'message': 'RAG system not initialized'
            }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'initialized': False,
            'error': str(e),
            'message': 'Error checking system status'
        })

@app.route('/api/sample-questions')
def get_sample_questions():
    """Get sample questions for the user"""
    sample_questions = [
        "What are the main arguments in this legal case?",
        "Why did the builder's arguments fail to save the 16th tower?",
        "What are the key legal principles discussed in the judgment?",
        "What compensation was awarded to the affected parties?",
        "What are the implications of this judgment for future cases?",
        "Summarize the facts of the case",
        "What were the main legal issues in dispute?",
        "How did the court interpret the relevant statutes?",
        "What precedent cases were cited in the judgment?",
        "What was the final order of the court?"
    ]
    
    return jsonify(sample_questions)

if __name__ == '__main__':
    print("🚀 Starting RAG Web UI...")
    
    # Initialize RAG system
    if initialize_rag_system():
        print("🌐 Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize RAG system. Please check your configuration.")
