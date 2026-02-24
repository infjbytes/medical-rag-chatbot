import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Medical RAG Chatbot")

st.title("Medical RAG Chatbot")

@st.cache_resource
def load_rag_system():
    # Load PDF
    loader = PyPDFLoader("medical.pdf")
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OllamaEmbeddings(model="llama3")

    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create LLM
    llm = Ollama(model="llama3")

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa_chain

qa = load_rag_system()

# Search bar
query = st.text_input("Ask a medical question:")

if query:
    with st.spinner("Generating answer..."):
        response = qa.run(query)
    st.write(response)