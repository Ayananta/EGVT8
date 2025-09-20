# 🌐 Elita Garden Vista Tower 8 Case RAG Web Interface

A beautiful, user-friendly web interface for querying the Elita Garden Vista Tower 8 case documents using AI-powered Retrieval-Augmented Generation (RAG).

## 🚀 Quick Start

### Option 1: Easy Startup (Recommended)
```bash
.\venv\Scripts\python.exe start_web_app.py
```

### Option 2: Manual Startup
```bash
.\venv\Scripts\python.exe web_rag_app.py
```

Then open your browser and go to: **http://localhost:5000**

## ✨ Features

### 🤖 **Ask AI Tab**
- **Intelligent Q&A**: Ask natural language questions about the Elita Garden Vista Tower 8 case documents
- **AI-Powered Answers**: Get comprehensive answers using Gemini AI
- **Source Citations**: See exactly which documents and sections were used
- **Sample Questions**: Click on pre-built questions to get started quickly

### 🔍 **Search Documents Tab**
- **Keyword Search**: Find specific terms across all documents
- **Document Filtering**: See which document contains your search terms
- **Relevance Scoring**: Results ranked by relevance to your query

### 📜 **Query History Tab**
- **Session History**: View all your previous questions and answers
- **Quick Reference**: Easily revisit past queries
- **Export Ready**: Copy answers for external use

## 📚 Available Documents

| Document | Chunks | Description |
|----------|--------|-------------|
| **Elita Order** | 273 | Court order and legal proceedings for Elita Garden Vista Tower 8 |
| **Sale Agreement** | 34 | Property sale terms and conditions for Tower 8 |
| **DEED of Conveyance** | 14 | Property transfer documentation for Tower 8 |
| **Total** | **321** | **Complete Elita Garden Vista Tower 8 case document corpus** |

## 🎯 Example Questions

Try asking these questions to get started:

### 📋 **General Questions**
- "What are the key terms of the sale agreement?"
- "What are the builder's obligations regarding possession?"
- "What happens if the builder fails to deliver on time?"

### 💰 **Financial Questions**
- "What are the payment terms in the agreement?"
- "What are the penalties for late payments?"
- "What are the maintenance charges mentioned?"

### ⚖️ **Legal Questions**
- "What legal protections exist for buyers?"
- "What are the consequences of builder delays?"
- "What are the dispute resolution mechanisms?"

### 🔍 **Specific Searches**
- Search for "possession" to find all possession-related clauses
- Search for "penalty" to find all penalty provisions
- Search for "agreement" to find agreement terms

## 🛠️ Technical Details

### **Backend Architecture**
- **Flask**: Web framework for the API
- **Multi-Document RAG**: Custom RAG system for multiple documents
- **Gemini AI**: Google's generative AI for intelligent answers
- **TF-IDF Search**: Fast text-based document retrieval

### **Frontend Features**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern UI**: Clean, professional interface
- **Real-time Status**: Live system status indicators
- **Interactive Elements**: Smooth animations and transitions

### **API Endpoints**
- `GET /api/status` - System status and document counts
- `POST /api/query` - Ask AI questions
- `POST /api/search` - Search documents directly
- `GET /api/history` - Query history
- `GET /api/sample_questions` - Sample questions

## 🔧 Configuration

### **Environment Setup**
Make sure your `.env` file contains:
```
GEMINI_API_KEY=your_api_key_here
```

### **System Requirements**
- Python 3.8+
- Virtual environment activated
- All dependencies installed (`pip install -r requirements.txt`)
- Valid Gemini API key

## 🌐 Accessing the Interface

### **Local Access**
- **URL**: http://localhost:5000
- **Network**: Only accessible from your computer

### **Network Access** (for sharing)
To allow others on your network to access:
1. Find your computer's IP address
2. Access via: `http://YOUR_IP:5000`
3. Make sure firewall allows port 5000

### **Example Network URLs**
- `http://192.168.1.100:5000`
- `http://10.0.0.50:5000`

## 📱 Mobile-Friendly

The interface is fully responsive and works great on:
- 📱 **Smartphones** (iOS/Android)
- 📱 **Tablets** (iPad/Android tablets)
- 💻 **Laptops** and **Desktops**

## 🔒 Security Notes

- The interface is designed for local/trusted network use
- No authentication is implemented (add if needed for production)
- API keys are stored in environment variables
- Session data is stored locally

## 🚨 Troubleshooting

### **Server Won't Start**
```bash
# Check if port 5000 is in use
netstat -an | findstr :5000

# Kill existing process if needed
taskkill /f /im python.exe
```

### **"RAG system not initialized" Error**
- Check your `.env` file has `GEMINI_API_KEY`
- Verify your API key is valid
- Make sure all document files exist

### **No Results Found**
- Try broader search terms
- Check if documents were processed correctly
- Verify chunk files exist in `Data/` directory

### **Slow Responses**
- Reduce `top_k` parameter (fewer results)
- Reduce `max_context` parameter (shorter answers)
- Check your internet connection for Gemini API

## 📊 Performance

- **Typical Response Time**: 2-5 seconds
- **Search Speed**: <1 second for document search
- **Memory Usage**: ~500MB for full system
- **Concurrent Users**: Tested with 1-5 users

## 🔄 Updates and Maintenance

### **Adding New Documents**
1. Process new PDF with `improved_pdf_splitter.py`
2. Generate embeddings with `fast_embedding_generator.py`
3. Update the RAG system configuration
4. Restart the web server

### **Updating Existing Documents**
1. Re-process the PDF files
2. Regenerate embeddings
3. Restart the web server

## 🎉 Success!

Your Elita Garden Vista Tower 8 case RAG system now has a beautiful, professional web interface that anyone can use to ask intelligent questions about the case documents!

---

**Need Help?** Check the main project README or create an issue in the repository.
