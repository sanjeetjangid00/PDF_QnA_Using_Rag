# 📄 PDF QnA Chatbot 🤖
A Streamlit-powered chatbot that lets you ask questions from any PDF document using LangChain, Google Gemini (via Generative AI), FAISS, and Hugging Face Embeddings. Just upload your PDF, and start chatting with your documents in real-time!

🚀 Features
✅ Upload any PDF and instantly make it searchable and queryable
✅ Intelligent text cleaning and chunking using LangChain
✅ Semantic search using FAISS vector store
✅ Embedding generation with Hugging Face (all-MiniLM-L6-v2)
✅ Question-answering via Google Gemini-1.5 Flash LLM
✅ Simple, interactive Streamlit UI

🧠 Tech Stack
| Tool/Library                                 | Purpose                                    |
| -------------------------------------------- | ------------------------------------------ |
| Streamlit                                    | Web-based UI                               |
| LangChain                                    | Text splitting, retrieval, and chain setup |
| FAISS                                        | Fast vector similarity search              |
| Hugging Face Embeddings                      | Text embedding generation                  |
| Google Gemini (via `langchain_google_genai`) | Powerful LLM for QA                        |
| PyPDFLoader                                  | PDF file loader                            |
| Python (dotenv, os, re)                      | Utility and environment configuration      |


📂 How It Works
1. Upload a PDF using the Streamlit UI.

2. The app:
- Cleans and splits the text using LangChain tools.
- Creates semantic embeddings with Hugging Face.
- Stores them in a FAISS vector database.
- Builds a QA chain using Google Gemini as the LLM.
  
3. Ask any question — get precise answers from your PDF content!

📷 Screenshot
(Add a screenshot or screen recording of your app in action here)

streamlit run app.py
⚠️ Requirements
- Python 3.8+
- Google Generative AI API key (Gemini)
- Internet connection (for model access)

🙌 Future Improvements
Support only PDF file formats (e.g., DOCX, TXT)
Chat history and download responses
Local model support (offline use)
UI enhancements and theme customization

📄 License
MIT License — feel free to use, modify, and contribute!
