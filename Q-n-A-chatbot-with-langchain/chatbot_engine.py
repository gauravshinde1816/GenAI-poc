import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
# load documents
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate



load_dotenv()


# load documents
docs = TextLoader("data/data.txt").load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)


# embed chunks
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings)

# retriever
retriever = vectorstore.as_retriever()

# llm model
llm = ChatGroq(model="llama3.1:8b" , api_key=os.getenv("GROQ_API_KEY"))



# propmts 
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that can answer questions about the following text:
    {text}

    using context {context} and chat history {chat_history}
    if there are not relevant context, answer as best as you can.

    also append the model name to the answer.
    model: {model}

    question: {query}

    answer:

    """
)

def call_llm(query: str , model: str , chat_history: str , max_tokens: int)->str:
    llm = ChatGroq(model=model , max_tokens=max_tokens)
    chain = prompt | llm
    contextDocs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in contextDocs])
    response = chain.invoke({"text": context , "context": context , "query": query , "model": model , "chat_history": chat_history})
    return response.content












