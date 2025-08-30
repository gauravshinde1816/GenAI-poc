import os
from dotenv import load_dotenv
import streamlit as st
import chatbot_engine as chatbot

# Load environment variables
load_dotenv()
models = ['llama-3.1-8b-instant', 'gemma2-9b-it', 'openai/gpt-oss-20b']

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Page config
st.set_page_config(
    page_title="Q&A Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 LangChain Q&A Chatbot")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    model_name = st.selectbox("Select a model", models)
    st.write("---")
    max_tokens = st.slider("Max Tokens", min_value=10, max_value=100, value=40, step=10)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Main chat area
col1, col2, col3 = st.columns([1, 6, 1])


with col2:
    # Chat history in a container
    chat_container = st.container(height=400, border=True)
    
    with chat_container:
        if not st.session_state.messages:
            st.write("👋 Hi! I'm your AI assistant. Ask me anything!")
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("ai"):
                    st.write(message["content"])
                    st.caption(f"Model: {message.get('model', 'Unknown')}")

    # Input at the bottom
    st.write("---")
    
    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question:", 
            placeholder="Type your message here...",
            key="chat_input"
        )
        
        col_a, col_b, col_c = st.columns([3, 3, 3])
        with col_b:
            submit_button = st.form_submit_button("Send 📤", use_container_width=True)

    # Handle form submission
    if submit_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.spinner("🤔 Thinking..."):
            try:
                if type(st.session_state.messages) == list:
                    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                else:
                    chat_history = str(st.session_state.messages)
                response = chatbot.call_llm(user_input, model_name, chat_history, max_tokens)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "model": model_name
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"❌ Error: {str(e)}",
                    "model": model_name
                })
        
        st.rerun()