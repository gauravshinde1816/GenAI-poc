import os
from dotenv import load_dotenv
import streamlit as st
import chatbot_engine as chatbot
from utils import process_pdf

# upload directory
UPLOAD_DIR = "uploads"
retriever = None


if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load environment variables
load_dotenv()
models = ['llama-3.1-8b-instant', 'gemma2-9b-it', 'openai/gpt-oss-20b']

st.title("Coversational RAG with Groq")

file = st.file_uploader("Upload a file", type=["pdf"], key="file")
if file is not None:
    with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
        f.write(file.read())
    retriever = process_pdf(os.path.join(UPLOAD_DIR, file.name))
else:
    st.write("No file uploaded")


# chatbot
st.subheader("Chatbot")
question = st.text_input("Enter your question", key="question")
session_id = st.text_input("Enter your session id", key="session_id")
if st.button("Ask"):
    response = chatbot.call_llm(question, "gemma2-9b-it", 1000, retriever, session_id)
    st.write(response)





