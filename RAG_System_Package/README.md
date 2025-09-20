# Legal Document RAG System

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
