# 🤖 Gemini RAG System for Legal Documents

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
venv\Scripts\activate  # Windows
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
