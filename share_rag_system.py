#!/usr/bin/env python3
"""
RAG System Sharing Package

This script creates a complete package that others can use to run your RAG system.
"""

import os
import shutil
import json
from pathlib import Path
import zipfile


def create_sharing_package():
    """Create a complete sharing package for the RAG system."""
    print("📦 Creating RAG System Sharing Package...")
    
    # Create sharing directory
    share_dir = Path("RAG_System_Package")
    share_dir.mkdir(exist_ok=True)
    
    # Files to include
    essential_files = [
        # Core RAG system
        "gemini_rag.py",
        "faiss_search.py", 
        "faiss_vector_store.py",
        "faiss_manager.py",
        
        # Setup and utilities
        "setup_gemini.py",
        "fast_embedding_generator.py",
        "improved_pdf_splitter.py",
        
        # Data files
        "Data/elita_embeddings_fast.json",
        "faiss_db/faiss_index.bin",
        "faiss_db/metadata.json",
        
        # Configuration
        "requirements.txt",
        
        # Documentation
        "gemini_summary.py",
        "test_gemini_integration.py"
    ]
    
    print("📋 Copying essential files...")
    for file_path in essential_files:
        src = Path(file_path)
        if src.exists():
            dst = share_dir / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            if src.is_file():
                shutil.copy2(src, dst)
                print(f"   ✅ {file_path}")
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"   ✅ {file_path}/ (directory)")
        else:
            print(f"   ⚠️  {file_path} (not found)")
    
    # Create setup instructions
    create_setup_instructions(share_dir)
    
    # Create example questions file
    create_example_questions(share_dir)
    
    # Create batch setup script
    create_batch_setup_script(share_dir)
    
    # Create README
    create_readme(share_dir)
    
    print(f"\n✅ Sharing package created in: {share_dir}")
    
    # Create ZIP file
    create_zip_package(share_dir)
    
    return share_dir


def create_setup_instructions(share_dir):
    """Create setup instructions file."""
    instructions = """# RAG System Setup Instructions

## Quick Start (3 steps):

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key
- Go to: https://makersuite.google.com/app/apikey
- Sign in with Google account
- Click "Create API Key"
- Copy the generated key

### 3. Configure API Key
```bash
python setup_gemini.py
```
Choose option 1 and paste your API key when prompted.

## Usage Examples:

### Ask Questions:
```bash
python gemini_rag.py --question "What is this legal case about?"
```

### Interactive Chat:
```bash
python gemini_rag.py --chat
```

### Vector Search (no AI needed):
```bash
python faiss_search.py --query "apartment ownership"
```

### View Statistics:
```bash
python faiss_search.py --stats
```

## What's Included:
- ✅ Pre-processed legal document (273 chunks)
- ✅ Vector embeddings (3,515 dimensions)
- ✅ FAISS database (optimized for fast search)
- ✅ Gemini AI integration
- ✅ Complete RAG system

## System Requirements:
- Python 3.8+
- Internet connection (for Gemini AI)
- ~20MB disk space

## Troubleshooting:
- If API key issues: Run `python setup_gemini.py`
- If import errors: Run `pip install -r requirements.txt`
- For help: Check the README.md file
"""
    
    with open(share_dir / "SETUP_INSTRUCTIONS.txt", 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("   ✅ SETUP_INSTRUCTIONS.txt")


def create_example_questions(share_dir):
    """Create example questions file."""
    questions = [
        "What is this legal case about?",
        "Who are the main parties in this dispute?",
        "What are the key legal arguments?",
        "What does the West Bengal Apartment Ownership Act say?",
        "What are the duties of a promoter under the law?",
        "What was the court's final decision?",
        "What legal precedents were cited?",
        "What are the rights of apartment owners?",
        "What is the role of NKDA in this case?",
        "What are the construction permit requirements?",
        "What is the legal status of the 16th tower?",
        "What are the consequences of illegal construction?",
        "How does the West Bengal Building Act apply?",
        "What are the promoter's liabilities?",
        "What is the relationship between NKDA and WB Apartment Ownership Act?"
    ]
    
    with open(share_dir / "example_questions.txt", 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(question + '\n')
    
    print("   ✅ example_questions.txt")


def create_batch_setup_script(share_dir):
    """Create automated setup script."""
    setup_script = """@echo off
echo Installing RAG System Dependencies...
pip install -r requirements.txt

echo.
echo Setting up Gemini API...
echo Please get your API key from: https://makersuite.google.com/app/apikey
echo.
python setup_gemini.py

echo.
echo Testing system...
python test_gemini_integration.py

echo.
echo Setup complete! Try asking a question:
echo python gemini_rag.py --question "What is this legal case about?"
pause
"""
    
    with open(share_dir / "setup.bat", 'w', encoding='utf-8') as f:
        f.write(setup_script)
    
    print("   ✅ setup.bat (Windows)")


def create_readme(share_dir):
    """Create comprehensive README."""
    readme = """# Legal Document RAG System

A powerful Retrieval-Augmented Generation (RAG) system for analyzing legal documents using AI.

## 🎯 What This System Does

This system processes a 94-page legal judgment about apartment ownership disputes and provides:
- **Intelligent Question Answering** using Gemini AI
- **Semantic Search** across legal content
- **Source Citations** for all answers
- **Interactive Chat Interface**
- **Batch Processing** capabilities

## 📊 System Statistics

- **Source Document**: Elita Order_29.08.2025.pdf (94 pages)
- **Processed Chunks**: 273 semantic chunks
- **Vector Dimensions**: 3,515 (TF-IDF embeddings)
- **Database Size**: ~16MB (FAISS optimized)
- **Search Speed**: Sub-millisecond vector search
- **AI Response Time**: 2-3 seconds per question

## 🚀 Quick Start

### Option 1: Automated Setup (Windows)
```bash
setup.bat
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get Gemini API key from: https://makersuite.google.com/app/apikey

# 3. Configure API key
python setup_gemini.py

# 4. Test system
python test_gemini_integration.py
```

## 💬 Usage Examples

### Ask Questions:
```bash
python gemini_rag.py --question "What is this legal case about?"
```

### Interactive Chat:
```bash
python gemini_rag.py --chat
```

### Process Multiple Questions:
```bash
python gemini_rag.py --batch example_questions.txt
```

### Vector Search (No AI):
```bash
python faiss_search.py --query "apartment ownership"
```

## 🔧 System Components

- **gemini_rag.py** - Main RAG system with AI
- **faiss_search.py** - Vector search interface
- **faiss_vector_store.py** - Database management
- **setup_gemini.py** - API configuration
- **Data/** - Processed embeddings and chunks
- **faiss_db/** - Optimized vector database

## 📋 Sample Questions

Try these questions to explore the legal document:

1. "What is this legal case about?"
2. "What are the main parties in this dispute?"
3. "What does the West Bengal Apartment Ownership Act say?"
4. "What are the duties of a promoter?"
5. "What was the court's final decision?"
6. "What are the rights of apartment owners?"
7. "What legal precedents were cited?"

## 🛠️ Technical Details

- **AI Model**: Google Gemini 1.5 Flash
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embedding Method**: TF-IDF with 1-2 grams
- **Search Algorithm**: Cosine similarity
- **Programming Language**: Python 3.8+
- **Dependencies**: See requirements.txt

## 📞 Support

For issues or questions:
1. Check SETUP_INSTRUCTIONS.txt
2. Run: `python test_gemini_integration.py`
3. Verify API key: `python setup_gemini.py`

## 🎉 Features

✅ **Semantic Search** - Find content by meaning, not just keywords
✅ **AI-Powered Analysis** - Intelligent legal document understanding
✅ **Source Verification** - Always cites which chunks were used
✅ **Interactive Interface** - Chat-like question answering
✅ **Batch Processing** - Handle multiple questions efficiently
✅ **Performance Optimized** - Sub-second search with FAISS
✅ **Legal Specialized** - Designed for legal document analysis

---

*This RAG system transforms your legal documents into an intelligent, searchable knowledge base powered by AI.*
"""
    
    with open(share_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print("   ✅ README.md")


def create_zip_package(share_dir):
    """Create ZIP package for easy sharing."""
    zip_path = "RAG_System_Package.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in share_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(share_dir.parent)
                zipf.write(file_path, arcname)
    
    zip_size = Path(zip_path).stat().st_size
    print(f"   ✅ {zip_path} ({zip_size:,} bytes)")


def main():
    """Create sharing package."""
    print("🎯 RAG SYSTEM SHARING PACKAGE CREATOR")
    print("=" * 50)
    
    package_dir = create_sharing_package()
    
    print(f"\n🎉 SHARING PACKAGE READY!")
    print("=" * 30)
    print(f"📁 Directory: {package_dir}")
    print(f"📦 ZIP File: RAG_System_Package.zip")
    print(f"📋 Instructions: {package_dir}/SETUP_INSTRUCTIONS.txt")
    print(f"📖 Documentation: {package_dir}/README.md")
    
    print(f"\n🚀 SHARING OPTIONS:")
    print("-" * 20)
    print("1. Share the ZIP file: RAG_System_Package.zip")
    print("2. Share the directory: RAG_System_Package/")
    print("3. Upload to cloud storage (Google Drive, Dropbox, etc.)")
    print("4. Email the ZIP file to colleagues")
    
    print(f"\n💡 WHAT OTHERS NEED:")
    print("-" * 20)
    print("• Python 3.8+ installed")
    print("• Internet connection")
    print("• Free Gemini API key (5 minutes to get)")
    print("• ~20MB disk space")
    
    print(f"\n🎯 RECIPIENTS CAN:")
    print("-" * 20)
    print("• Run: setup.bat (Windows) or follow SETUP_INSTRUCTIONS.txt")
    print("• Ask questions about your legal document")
    print("• Use interactive chat interface")
    print("• Perform semantic search")
    print("• Get AI-powered legal analysis")


if __name__ == "__main__":
    main()
