import os
from dotenv import load_dotenv
import streamlit as st
from chatbot_engine import create_database_toolkit, create_local_db
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, AgentType
from langchain.agents import create_sql_agent
from langchain.callbacks import StreamlitCallbackHandler

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Database Q&A Agent",
    page_icon="🗄️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    
    }
    .user-message {

        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        right: 0;
        border-left: 4px solid #9c27b0;
    }
    .sidebar-section {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# create local database
create_local_db()

# initialize llm
llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))

# Display header
st.markdown("""
<div class="main-header">
    <h1>🗄️ Database Q&A Agent</h1>
    <p>Ask questions about your database and get intelligent answers powered by AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
database = st.sidebar.radio("Select Database", ["SQLite", "MySQL"])
st.sidebar.markdown('</div>', unsafe_allow_html=True)

if database == "MySQL":
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 🔐 MySQL Connection")
    host = st.sidebar.text_input("Enter the MySQL host", value="localhost")
    user = st.sidebar.text_input("Enter the MySQL username") 
    password = st.sidebar.text_input("Enter the MySQL password", type="password")
    db = st.sidebar.text_input("Enter the MySQL database name")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

database_uri = ""
if database == "MySQL":
    if not (host and user and password and db): 
        st.sidebar.error("Please fill in all fields")
    else:
        database_uri = f"mysql+pymysql://{user}:{password}@{host}:3307/{db}"
else:
    database_uri = f"sqlite:///tutorial.db"

# Status section
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 📊 Status")
if database_uri != "":
    if database == "MySQL" and (host and user and password and db):
        toolkit = create_database_toolkit(llm, database_uri)
        st.sidebar.success("✅ Connected to MySQL")
    elif database == "SQLite":  
        toolkit = create_database_toolkit(llm, database_uri)
        st.sidebar.success("✅ Connected to SQLite")
    else:
        st.sidebar.error("Please select a database")
else:
    st.sidebar.warning("⚠️ Database not configured")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Clear chat button and message count
message_count = len(st.session_state.messages)
st.sidebar.info(f"💬 Messages: {message_count}")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Main chat interface
st.markdown("### 💬 Chat Interface")

# Display existing messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Chat input
if database_uri != "" and 'toolkit' in locals():
    query = st.chat_input("Ask me anything about the database...")
    
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message immediately
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {query}
        </div>
        """, unsafe_allow_html=True)
        
        # Get agent response
        with st.spinner("🤔 Thinking..."):
            try:
                agent_executor = create_sql_agent(llm, toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
                response = agent_executor.run(query, callbacks=[StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)])
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display assistant response
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {response}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)
else:
    st.info("Please configure your database connection to start chatting!")
    
    # Show sample questions
    st.markdown("### 💡 Sample Questions You Can Ask")
    sample_questions = [
        "What products are available in the database?",
        "Show me the most expensive product",
        "What is the average price of all products?",
        "List all products with prices above $200",
        "How many products are there in total?"
    ]
    
    for question in sample_questions:
        st.markdown(f"• {question}")






    







