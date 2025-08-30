# 🗄️ Database Q&A Agent with LangChain

A powerful Streamlit application that allows you to chat with your SQL databases using natural language queries powered by AI. This application supports both SQLite and MySQL databases and provides an intuitive chat interface for database exploration.


https://github.com/user-attachments/assets/a4c21b96-c3dc-4f04-815d-058a47f620a6




## ✨ Features

- **🤖 AI-Powered Queries**: Use natural language to query your database
- **🗃️ Multi-Database Support**: Connect to SQLite or MySQL databases
- **💬 Interactive Chat Interface**: Chat-like experience with conversation history

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key (for AI model access)
- MySQL server (optional, for MySQL database support)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Database-Q-n-A-agent-using-LangChain-SQL-toolkit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🗃️ Database Setup

### SQLite Database
The application automatically creates a local SQLite database (`tutorial.db`) with sample product data including:
- Product names, prices, categories, and descriptions
- 12 sample products across different categories
- Automatic table creation and data population

### MySQL Database
To connect to a MySQL database:
1. Select "MySQL" in the database selection
2. Enter your MySQL connection details:
   - Host (default: localhost)
   - Port (default: 3306)
   - Username
   - Password
   - Database name
3. Click "Connect to MySQL"

## 💬 Usage

### Getting Started
1. **Select Database Type**: Choose between SQLite or MySQL
2. **Configure Connection**: Enter database connection details if using MySQL
3. **Add API Key**: Provide your Groq API key
4. **Start Chatting**: Ask questions about your database in natural language

### Sample Questions
- "What products are available in the database?"
- "Show me the most expensive product"
- "What is the average price of all products?"
- "List all products with prices above $200"
- "How many products are there in total?"
- "What are the different product categories?"

### Features
- **Chat History**: All conversations are saved in session state
- **Error Handling**: Comprehensive error messages and validation
- **Status Monitoring**: Real-time connection and agent status
- **Session Reset**: Clear chat history and reset application state

## 🏗️ Architecture

### Components
- **`app.py`**: Main Streamlit application with UI and session management
- **`chatbot_engine.py`**: Database toolkit creation and management
- **`tutorial.db`**: Sample SQLite database with product data

### Key Functions
- **Session State Management**: Persistent storage of chat history and application state
- **Database Connection**: Secure connection handling with validation
- **Agent Creation**: LangChain SQL agent with error handling
- **Query Processing**: Natural language to SQL conversion and execution

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key for AI model access

### Database Configuration
- **SQLite**: Automatic local database creation
- **MySQL**: Configurable connection parameters

### AI Model Settings
- **Model**: gemma2-9b-it (Groq)
- **Temperature**: 0.1 (for consistent responses)
- **Streaming**: Enabled for real-time responses


## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your Groq API key is correct
   - Check if the API key has sufficient credits

2. **Database Connection Issues**
   - Verify database credentials
   - Check if the database server is running
   - Ensure network connectivity for remote databases

3. **Agent Creation Failure**
   - Check database connection status
   - Verify API key configuration
   - Review error messages in the status section

### Error Messages
The application provides detailed error messages for:
- Database connection failures
- API authentication errors
- Query processing issues
- Agent creation problems

## 🔄 Session Management

### Persistent State
- **Chat History**: All messages are preserved during the session
- **Database Connection**: Connection status is maintained
- **Agent State**: Agent creation status is tracked
- **Error State**: Error messages are displayed until resolved

### Reset Functionality
- **Session Reset**: Clear all chat history and reset application state
- **Fresh Start**: Begin with a clean slate
- **State Cleanup**: Remove all stored session data

## 📈 Performance Features

- **Connection Caching**: Database connections are optimized
- **Streaming Responses**: Real-time AI response streaming
- **Efficient Queries**: Optimized database query execution
- **Memory Management**: Efficient session state handling


## 🙏 Acknowledgments

- **LangChain**: For the powerful SQL agent framework
- **Streamlit**: For the excellent web application framework
- **Groq**: For the fast AI model inference
- **SQLAlchemy**: For robust database abstraction

---

**Happy Database Chatting! 🗄️💬** 
