import os
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
load_dotenv()

LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT = st.secrets["LANGCHAIN_PROJECT"]
LANGCHAIN_ENDPOINT = st.secrets["LANGCHAIN_ENDPOINT"]
LANGCHAIN_TRACING_V2 = st.secrets["LANGCHAIN_TRACING_V2"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

load_dotenv()

#clean text data
@traceable(name = 'clean_text')
def clean_text(text):
    text = re.sub(r"\s+", " ", text) 
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    return text.strip()

# Process PDF and clean the data
@traceable(name = 'load_pdf')
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# Create FAISS vector store
@traceable(name = 'creating_vector_store')
def create_vector_store(texts):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embedding)
    return vector_store

# Build QA Chain
@traceable(name = 'retrieving_&_generating')
def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_type = 'similarity', search_k = 5)
    llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash')
    qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever = retriever, return_source_documents = True)
    return qa_chain

# Streamlit app
def main():
    st.write('<h1 style="text-align: center; color: blue;">PDF QnA Chatbot</h1>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Upload your PDF file', type=['pdf'])

    if uploaded_file:
        if "qa_chain" not in st.session_state:  # ✅ Only first time banega
            with open("temp.pdf", 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success("File uploaded successfully.....")
            with st.spinner("Processing PDF..."):
                texts = process_pdf("temp.pdf")
                vectorstore = create_vector_store(texts)
                st.session_state.qa_chain = build_qa_chain(vectorstore)  # save in session
            st.success("Chatbot is ready....")

        # Query chatbot
        st.write("Ask a question....")
        user_query = st.text_input("Your Question...")
        if user_query:
            with st.spinner("Generating Answer...."):
                response = st.session_state.qa_chain({"query": user_query})
                st.write("### Answer:")
                st.write(response['result'])
        else:
            st.info("Please ask a question to get started.....")
    else:
        st.info("Please upload a PDF to get started......")

if __name__ == '__main__':
    main()
