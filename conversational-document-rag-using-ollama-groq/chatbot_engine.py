import os
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain , create_history_aware_retriever
from dotenv import load_dotenv
# load documents

from langchain.prompts import ChatPromptTemplate


load_dotenv()


store = {}

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that can answer questions about the following text:\n"
     "{input}\n\nUse context {context} when relevant. "
     "If context isn't relevant, answer as best as you can.\n"
     "Also append the model name to the answer.\nmodel: {model}"),
    MessagesPlaceholder("chat_history"),           
    ("human", "question: {input}\nanswer:")
])


def get_session_history(session_id: str)->str:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def call_llm(query: str , model: str , max_tokens: int , retriever , session_id: str)->str:
    config = {"configurable": {"session_id": session_id}}
    llm = ChatGroq(model=model , max_tokens=max_tokens)

    # By this even context is aware about the chat history
    query_rebuild_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that you can rebuild the query based on the chat history and the new query\n"),
        MessagesPlaceholder("chat_history"),           
        ("human", "query: {input}\nrebuilt query:")
    ])
    # create history aware retriever
    history_aware_retriever = create_history_aware_retriever(llm , retriever , query_rebuild_prompt)
    # create stuff chain
    stuff_chain = create_stuff_documents_chain(llm , prompt)
    # create retrieval chain
    retrieval_chain = create_retrieval_chain(history_aware_retriever , stuff_chain)
    # create runnable with message history
    

    runnable_with_chat_history = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # I have used create_retrieval_chain to inject the context into prompt rathetr than explicitly passing it
    # contextDocs = retriever.get_relevant_documents(query)
    # context = "\n\n".join([doc.page_content for doc in contextDocs])
    response = runnable_with_chat_history.invoke({ "input": query , "model": model} , config=config)
    print(response['context'])
    return response["answer"]












