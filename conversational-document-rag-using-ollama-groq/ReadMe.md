# Conversational Document RAG Chatbot using Groq

A sophisticated conversational chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about uploaded documents. Built with Streamlit, LangChain, and powered by Groq's LPU inference engine.

<img width="1618" height="987" alt="image" src="https://github.com/user-attachments/assets/eab9270c-6240-4a0c-8dd3-e96ae484b449" />


## 🚀 Features

- **Document Upload & Processing**: Upload PDF documents and automatically process them for RAG
- **Conversational Memory**: Maintains chat history across sessions using `ChatMessageHistory` and `RunnableHistory`
- **Context-Aware Retrieval**: Uses `create_history_aware_retrieval_chain` to make context retrieval aware of chat history
- **Vector Storage**: Chroma vector store for efficient document retrieval
- **Multiple Model Support**: Compatible with various Groq models (llama-3.1-8b-instant, gemma2-9b-it, gpt-oss-20b)
- **Real-time Chat Interface**: Interactive Streamlit-based chat interface

## 🎯 Architecture

The project implements a sophisticated RAG pipeline:

1. **Document Processing**: PDFs are processed and chunked into searchable segments
2. **Vector Embedding**: Documents are embedded and stored in Chroma vector store
3. **Context-Aware Retrieval**: `create_history_aware_retrieval_chain` retrieves relevant context considering chat history
4. **Document Stuffing**: `create_stuff_documents_chain` combines retrieved context with prompts
5. **Conversational Memory**: `RunnableWithMessageHistory` maintains conversation state across interactions
6. **LLM Inference**: Groq LPU engine generates responses using retrieved context and chat history

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/conversational-document-rag-using-ollama-groq.git
   cd conversational-document-rag-using-ollama-groq
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

## 🔧 Configuration

1. **Get your Groq API key** from [Groq Console](https://console.groq.com/)
2. **Create a `.env` file** in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload a PDF document** using the file uploader
3. **Enter a session ID** to maintain conversation context
4. **Ask questions** about your uploaded document
5. **Enjoy context-aware responses** that remember your conversation history

## 📚 How It Works

1. **Document Upload**: User uploads a PDF document
2. **Processing**: Document is chunked, embedded, and stored in vector database
3. **Question Input**: User asks a question with a session ID
4. **Context Retrieval**: System retrieves relevant document chunks considering chat history
5. **Response Generation**: LLM generates answer using retrieved context and conversation memory
6. **Memory Update**: Chat history is updated for future context-aware interactions

## 🎯 Use Cases

- **Research Assistant**: Ask questions about research papers and documents
- **Document Q&A**: Get instant answers from long documents, manuals, or reports
- **Educational Tool**: Interactive learning from textbooks and educational materials
- **Business Intelligence**: Extract insights from business documents and reports

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
---

**Note**: Make sure you have sufficient Groq API credits for testing and production use.
    
