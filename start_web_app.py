#!/usr/bin/env python3
"""
Simple startup script for the Elita Garden Vista Tower 8 Case RAG Web Application.
"""

import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if all required files exist."""
    required_files = [
        'web_rag_app.py',
        'multi_doc_rag.py',
        'multi_doc_search.py',
        'templates/index.html',
        '.env'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def check_env_file():
    """Check if .env file has the required API key."""
    try:
        with open('.env', 'r') as f:
            content = f.read()
            if 'GEMINI_API_KEY=' in content:
                return True
    except:
        pass
    
    print("❌ GEMINI_API_KEY not found in .env file")
    print("   Please make sure your .env file contains:")
    print("   GEMINI_API_KEY=your_api_key_here")
    return False

def start_server():
    """Start the Flask server."""
    print("🚀 Starting Elita Garden Vista Tower 8 Case RAG Web Application...")
    print("=" * 50)
    
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the missing files.")
        return False
    
    if not check_env_file():
        print("\n❌ Environment check failed. Please configure your API key.")
        return False
    
    print("✅ All requirements met!")
    print("🌐 Starting web server on http://localhost:5000")
    print("📚 Available documents:")
    print("   - Elita Order (273 chunks)")
    print("   - Sale Agreement (34 chunks)")
    print("   - DEED of Conveyance (14 chunks)")
    print("   - Total: 321 chunks")
    print("\n💡 The web interface will open automatically in your browser.")
    print("💡 Press Ctrl+C to stop the server.")
    print("=" * 50)
    
    # Wait a moment before opening browser
    time.sleep(2)
    
    # Open browser
    try:
        webbrowser.open('http://localhost:5000')
        print("🌐 Browser opened automatically")
    except:
        print("🌐 Please open your browser and go to: http://localhost:5000")
    
    # Start the server
    try:
        subprocess.run([sys.executable, 'web_rag_app.py'])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_server()
