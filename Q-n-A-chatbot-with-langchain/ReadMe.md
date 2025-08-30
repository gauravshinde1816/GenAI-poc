# Q&A Chatbot with LangChain

A powerful question-and-answer chatbot built with modern AI technologies, featuring multimodal support, vector search, and  Streamlit interface.


<img width="1708" height="940" alt="image" src="https://github.com/user-attachments/assets/3ae70007-10dd-4169-9a26-8b348768e1eb" />


## 🚀 Technologies Used

- **Streamlit** - Modern web application framework for data science
- **Groq LPU Engine** - High-performance inference engine for multimodal open-source models
- **Ollama Embeddings** - Local embedding generation using Ollama
- **Chroma Vector Store** - Fast and scalable vector database for similarity search
- **LangChain** - Framework for building LLM-powered applications
- **LangChain Text Splitters** - Intelligent document chunking and processing

## ✨ Features

- **Multi-Model Support**: Choose from multiple LLM models including:
  - Llama 3.1 8B Instant
  - Gemma2 9B IT
  - OpenAI GPT-OSS 20B
- **Vector Search**: Intelligent document retrieval using Chroma vector store
- **Local Embeddings**: Generate embeddings locally with Ollama
- **Chat History**: Maintains conversation context across sessions
- **Responsive UI**: Clean, modern interface built with Streamlit
- **Configurable Parameters**: Adjust max tokens and other settings

## 🏗️ Architecture

The chatbot follows a sophisticated architecture:

1. **Document Processing**: Loads and splits documents using LangChain text splitters
2. **Vector Embeddings**: Generates embeddings using Ollama's nomic-embed-text model
3. **Vector Storage**: Stores embeddings in Chroma vector database
4. **Retrieval**: Finds relevant context using similarity search
5. **LLM Generation**: Generates responses using Groq's high-performance models
6. **Web Interface**: Beautiful Streamlit UI for user interaction

## 📋 Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Groq API key
- Virtual environment (recommended)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Q-n-A-chatbot-with-langchain
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Install and start Ollama**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the embedding model
   ollama pull nomic-embed-text
   ```

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Configure settings** in the sidebar:
   - Select your preferred LLM model
   - Adjust max tokens
   - Clear chat history if needed

4. **Start chatting** by typing your questions in the input field

## 📁 Project Structure

```
Q-n-A-chatbot-with-langchain/
├── app.py                 # Main Streamlit application
├── chatbot_engine.py      # Core chatbot logic and LangChain integration
├── data/
│   └── data.txt          # Sample data for the chatbot
├── requirement.txt        # Python dependencies
├── ReadMe.md             # This file
└── venv/                 # Virtual environment (created during setup)
```

## 🔧 Configuration

### Model Selection
The chatbot supports multiple models through Groq's LPU engine:
- **llama-3.1-8b-instant**: Fast and efficient Llama model
- **gemma2-9b-it**: Google's Gemma model for instruction tuning
- **openai/gpt-oss-20b**: OpenAI's open-source GPT model

### Embedding Model
Currently uses `nomic-embed-text` through Ollama for local embedding generation.

### Vector Store
Chroma vector database with:
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Retrieval of top relevant documents for context

## 🎯 How It Works

1. **Document Loading**: The system loads documents from the `data/` directory
2. **Text Splitting**: Documents are intelligently split into chunks using LangChain's RecursiveCharacterTextSplitter
3. **Embedding Generation**: Each chunk is converted to a vector using Ollama embeddings
4. **Vector Storage**: Embeddings are stored in Chroma vector database
5. **Query Processing**: When a user asks a question:
   - The question is embedded
   - Similar documents are retrieved from the vector store
   - Context is combined with the question and chat history
   - The LLM generates a response using the provided context
6. **Response Generation**: The chatbot responds with relevant information and maintains conversation context

## 🚧 Troubleshooting

### Common Issues

1. **Ollama not running**
   ```bash
   ollama serve
   ```

2. **Missing embedding model**
   ```bash
   ollama pull nomic-embed-text
   ```

3. **Groq API key issues**
   - Ensure your `.env` file contains the correct `GROQ_API_KEY`
   - Verify your Groq account has sufficient credits

4. **Dependencies issues**
   ```bash
   pip install --upgrade -r requirement.txt
   ```
   
## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the powerful framework
- [Groq](https://groq.com/) for high-performance inference
- [Ollama](https://ollama.ai/) for local model hosting
- [Chroma](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface

---

**Happy Chatting! 🤖✨**

