import os
import re
import streamlit as st
from dotenv import load_dotenv

# loaders / text splitters / vectorstores
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# embeddings + LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# LCEL (runnable-based) pieces
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# tracing
from langsmith import traceable

load_dotenv()

# --- Secrets / env (Streamlit secrets or .env) ---
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = st.secrets.get("LANGCHAIN_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT = st.secrets.get("LANGCHAIN_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_TRACING_V2 = st.secrets.get("LANGCHAIN_TRACING_V2") or os.getenv("LANGCHAIN_TRACING_V2")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

# --------------- utilities ---------------
# @traceable(name='clean_text')
# def clean_text(text: str) -> str:
#     text = re.sub(r"\s+", " ", text)
# #    text = re.sub(r"[^\w\s.,!?-]", "", text)
#     return text.strip()

# --------------- PDF processing ---------------
@traceable(name='load_pdf')
def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # for doc in documents:
    #     doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    texts = text_splitter.split_documents(documents)
    return texts

# --------------- vector store creation ---------------
@traceable(name='creating_vector_store')
def create_vector_store(texts):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embedding)
    return vector_store

# --------------- build runnable RAG chain ---------------
@traceable(name='retrieving_&_generating')
def build_qa_chain(vector_store):
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # instantiate LLM (Gemini)
    llm = ChatGroq(model="openai/gpt-oss-20b")

    # simple prompt that accepts {context} and {question}
    prompt = ChatPromptTemplate.from_template(
        """
        Use the provided context to answer the question. If the answer is not contained
        in the context, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """
    )

    rag_chain = (
        {
            "context": retriever,             
            "question": RunnablePassthrough() 
        }
        | prompt
        | llm
    )

    return rag_chain

# --------------- Streamlit UI ---------------
def main():
    st.set_page_config(page_title="PDF QnA Chatbot", layout="wide")
    st.markdown('<h1 style="text-align: center; color: blue;">PDF QnA Chatbot</h1>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader('Upload your PDF file', type=['pdf'])
    if uploaded_file:
        # save temp pdf
        temp_path = "temp.pdf"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        st.success("File uploaded successfully.")
        texts = process_pdf(temp_path)

        if 'faiss' not in st.session_state:
            with st.spinner("Processing PDF and building vector store..."):
                vectorstore = create_vector_store(texts)
                st.session_state['faiss'] = vectorstore
                st.success("Chatbot is ready.")
        else:
            st.info("Using existing vector DB.")

        # build chain each run (cheap), or cache if you prefer
        vectorstore = st.session_state.get('faiss')
        if vectorstore is None:
            st.error("Vector store missing. Re-upload the PDF.")
            return

        qa_chain = build_qa_chain(vectorstore)

        st.write("Ask a question:")
        user_query = st.text_input("Your Question...")

        if user_query:
            with st.spinner("Generating answer..."):
                # invoke the runnable. we pass a dict with 'question' to be explicit.
                try:
                    raw_resp = qa_chain.invoke(user_query)
                except TypeError:
                    # fallback: pass string directly (some versions accept direct input)
                    raw_resp = qa_chain.invoke(user_query)

                # robust extraction of text from possible return types
                answer_text = None
                try:
                    # try common attributes
                    if hasattr(raw_resp, "content"):
                        answer_text = raw_resp.content
                    elif hasattr(raw_resp, "text"):
                        answer_text = raw_resp.text
                    elif isinstance(raw_resp, dict):
                        # try common keys
                        for k in ("answer", "result", "text", "output", "content"):
                            if k in raw_resp:
                                answer_text = raw_resp[k]
                                break
                        if answer_text is None:
                            # stringify if no key matched
                            answer_text = str(raw_resp)
                    else:
                        answer_text = str(raw_resp)
                except Exception as e:
                    answer_text = f"Error extracting response: {e}"

                st.markdown("### Answer:")
                st.write(answer_text)

        else:
            st.info("Please ask a question to get started.")
    else:
        st.info("Please upload a PDF to get started.")
        st.session_state.pop('faiss', None)


if __name__ == '__main__':
    main()







