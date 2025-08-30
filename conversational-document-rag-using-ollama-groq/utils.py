from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def split_pdf(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)
    return docs



def getRetriever(docs , model="nomic-embed-text"):
    embeddings = OllamaEmbeddings(model=model)
    chroma_db = Chroma.from_documents(docs, embeddings)
    retriever = chroma_db.as_retriever()
    return retriever


def process_pdf(file_path):
    print(file_path)
    docs = load_pdf(file_path)
    docs = split_pdf(docs)
    retriever = getRetriever(docs)
    return retriever