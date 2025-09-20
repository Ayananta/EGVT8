#!/usr/bin/env python3
"""
Create Sharing Package for Gemini RAG System

This script creates a complete package that others can use to run your RAG system.
"""

import os
import shutil
import zipfile
from pathlib import Path
import json


def create_sharing_package():
    """Create a complete sharing package."""
    print("📦 Creating Gemini RAG Sharing Package...")
    
    # Create sharing directory
    share_dir = Path("RAG_System_Share")
    if share_dir.exists():
        shutil.rmtree(share_dir)
    share_dir.mkdir()
    
    # Files to include
    essential_files = [
        "gemini_rag.py",
        "faiss_search.py", 
        "faiss_vector_store.py",
        "faiss_manager.py",
        "improved_pdf_splitter.py",
        "fast_embedding_generator.py",
        "setup_gemini.py",
        "ask_question.py",
        "requirements.txt",
        "README_SHARE.md"
    ]
    
    # Directories to include
    essential_dirs = [
        "faiss_db",
        "Data"
    ]
    
    print("📁 Copying essential files...")
    for file in essential_files:
        if Path(file).exists():
            shutil.copy2(file, share_dir)
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")
    
    print("📁 Copying essential directories...")
    for dir_name in essential_dirs:
        if Path(dir_name).exists():
            shutil.copytree(dir_name, share_dir / dir_name)
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
    
    # Create setup script for others
    setup_script = share_dir / "setup_for_others.py"
    with open(setup_script, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
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
    print("\\n💡 Optional: Create virtual environment for isolation")
    print("   python -m venv venv")
    print("   venv\\\\Scripts\\\\activate  (Windows)")
    print("   source venv/bin/activate  (Linux/Mac)")
    
    print("\\n🎉 Setup Complete!")
    print("\\n📋 Next Steps:")
    print("1. Get Gemini API key: https://makersuite.google.com/app/apikey")
    print("2. Run: python setup_gemini.py")
    print("3. Test: python ask_question.py 'What is this legal case about?'")
    print("4. Chat: python gemini_rag.py --chat")

if __name__ == "__main__":
    main()
''')
    
    print(f"  ✅ Created setup_for_others.py")
    
    # Create comprehensive README
    readme_content = '''# 🤖 Gemini RAG System for Legal Documents

A complete Retrieval-Augmented Generation (RAG) system that combines PDF processing, vector embeddings, and Gemini AI for intelligent question answering about legal documents.

## 🎯 What This System Does

- **Processes PDF documents** into intelligent chunks
- **Creates vector embeddings** for semantic search
- **Uses FAISS database** for lightning-fast similarity search
- **Integrates Gemini AI** for intelligent question answering
- **Provides source citations** for all answers

## 📊 System Statistics

- **Source Document**: 94-page legal judgment
- **Chunks Created**: 273 semantic chunks
- **Vector Dimensions**: 3,515 TF-IDF features
- **Database Size**: ~15.5 MB
- **Search Speed**: Sub-millisecond vector search
- **AI Response**: 2-3 seconds per question

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install Python packages
pip install -r requirements.txt

# Optional: Create virtual environment
python -m venv venv
venv\\Scripts\\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Get Gemini API Key
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the generated key

### 3. Configure API Key
```bash
python setup_gemini.py
```

### 4. Test the System
```bash
# Ask a question
python ask_question.py "What is this legal case about?"

# Start interactive chat
python gemini_rag.py --chat

# Vector search (no AI needed)
python faiss_search.py --query "apartment ownership"
```

## 🛠️ Available Commands

### AI-Powered Question Answering
```bash
# Single question
python gemini_rag.py --question "Your question here"

# Interactive chat session
python gemini_rag.py --chat

# Batch processing
python gemini_rag.py --batch questions.txt
```

### Vector Search (No AI Required)
```bash
# Search by text
python faiss_search.py --query "search terms"

# Find similar chunks
python faiss_search.py --chunk-id 49

# Database statistics
python faiss_search.py --stats
```

### System Management
```bash
# Performance benchmarking
python faiss_manager.py --benchmark faiss_db/faiss_index.bin

# Create different index types
python faiss_manager.py --compare
```

## 📁 File Structure

```
RAG_System_Share/
├── gemini_rag.py              # Main RAG system with Gemini AI
├── faiss_search.py            # Vector search interface
├── faiss_vector_store.py      # FAISS database management
├── improved_pdf_splitter.py   # PDF chunking for legal documents
├── fast_embedding_generator.py # TF-IDF embedding generation
├── setup_gemini.py            # API key configuration
├── ask_question.py            # Simple question interface
├── requirements.txt           # Python dependencies
├── faiss_db/                  # Vector database
│   ├── faiss_index.bin        # FAISS index file
│   └── metadata.json          # Database metadata
└── Data/                      # Processed documents
    ├── elita_embeddings_fast.json # Vector embeddings
    └── elita_chunks_improved.json # Document chunks
```

## 💬 Sample Questions

- "What is this legal case about?"
- "Who are the main parties in this dispute?"
- "What are the key legal arguments?"
- "What does the West Bengal Apartment Ownership Act say?"
- "What are the duties of a promoter under the law?"
- "What was the court's final decision?"
- "What legal precedents were cited?"
- "What are the rights of apartment owners?"

## 🔧 Technical Details

### Technologies Used
- **Python 3.8+**
- **Google Generative AI (Gemini)**
- **FAISS (Facebook AI Similarity Search)**
- **scikit-learn (TF-IDF embeddings)**
- **pdfplumber (PDF processing)**

### Architecture
1. **PDF → Intelligent Chunking** (273 semantic chunks)
2. **Chunks → TF-IDF Embeddings** (3,515 dimensions)
3. **Embeddings → FAISS Database** (high-performance search)
4. **Query → Vector Search** (retrieve relevant chunks)
5. **Context + Question → Gemini AI** (generate answer)

## 🎯 Performance Metrics

- **Vector Search**: ~1ms per query
- **AI Generation**: 2-3 seconds per question
- **Memory Usage**: ~4MB for full database
- **Throughput**: 1000+ searches per second
- **Accuracy**: Context-aware with source citations

## 🆘 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **"API key not found"**
   ```bash
   python setup_gemini.py
   ```

3. **"FAISS database not found"**
   - Ensure `faiss_db/` directory exists
   - Check that `faiss_index.bin` is present

4. **"Gemini API disabled"**
   - Enable Generative Language API in Google Cloud Console
   - Visit: https://console.developers.google.com/apis/api/generativelanguage.googleapis.com

### Getting Help

- Check system status: `python test_gemini_integration.py`
- View database stats: `python faiss_search.py --stats`
- Test connection: `python test_gemini_connection.py`

## 🎉 Success!

Once everything is set up, you'll have a powerful AI system that can:
- Answer complex questions about legal documents
- Provide source citations for all answers
- Search through documents semantically
- Handle conversational interactions

Your legal document is now powered by AI! 🚀
'''
    
    readme_path = share_dir / "README_SHARE.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"  ✅ Created README_SHARE.md")
    
    # Create zip file
    zip_path = "Gemini_RAG_System_Complete.zip"
    print(f"\n📦 Creating zip file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in share_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(share_dir)
                zipf.write(file_path, arcname)
    
    # Get zip file size
    zip_size = Path(zip_path).stat().st_size
    
    print(f"✅ Package created successfully!")
    print(f"📦 Zip file: {zip_path} ({zip_size:,} bytes / {zip_size/(1024*1024):.1f} MB)")
    print(f"📁 Share directory: {share_dir}")
    
    # Create sharing instructions
    instructions = f"""
🎉 SHARING PACKAGE CREATED!

📦 Files created:
  • {zip_path} - Complete system package
  • {share_dir}/ - Unzipped directory for testing

📋 How to share:
1. Send the zip file: {zip_path}
2. Recipients extract and run: python setup_for_others.py
3. They get their own Gemini API key and configure it
4. Ready to use!

🚀 What others will get:
  • Complete RAG system with your processed legal document
  • All 273 chunks and vector embeddings
  • FAISS database ready to use
  • Full AI question answering capabilities
  • Comprehensive documentation

💡 Sharing options:
  • Email the zip file
  • Upload to cloud storage (Google Drive, Dropbox, etc.)
  • Share via file transfer services
  • Host on GitHub (remove API key from .env first)
"""
    
    print(instructions)
    
    # Save instructions to file
    with open("SHARING_INSTRUCTIONS.txt", 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    return zip_path, share_dir


if __name__ == "__main__":
    create_sharing_package()
