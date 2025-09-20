#!/usr/bin/env python3
"""
Setup Script for Gemini RAG System
Run this script to set up the system on a new machine.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and show progress."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ {description} completed")
        else:
            print(f"  ❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ {description} failed: {e}")
        return False
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Gemini RAG System")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return
    
    # Create virtual environment (optional but recommended)
    print("\n💡 Optional: Create virtual environment for isolation")
    print("   python -m venv venv")
    print("   venv\\Scripts\\activate  (Windows)")
    print("   source venv/bin/activate  (Linux/Mac)")
    
    print("\n🎉 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Get Gemini API key: https://makersuite.google.com/app/apikey")
    print("2. Run: python setup_gemini.py")
    print("3. Test: python ask_question.py 'What is this legal case about?'")
    print("4. Chat: python gemini_rag.py --chat")

if __name__ == "__main__":
    main()
