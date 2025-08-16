import os
import shutil
import tempfile
import streamlit as st
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# Google Gemini (AI Studio) SDK
import google.generativeai as genai

# dotenv for local development
from dotenv import load_dotenv
load_dotenv()

# ----------------- Streamlit Config / UI -----------------
st.set_page_config(page_title="PDF/Doc Q&A (Gemini • Streamlit)", page_icon="🟣", layout="wide")
st.title("📄 Document Q&A — Gemini (Google AI Studio)")

# ✅ Load API Key (Secrets.toml for cloud, .env for local)
GOOGLE_API_KEY = None
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Add it in `.streamlit/secrets.toml` or `.env`.")
    st.stop()

# Configure Gemini client
genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL_NAME = "gemini-1.5-flash"  # free + fast tier
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"]) 

if uploaded_file:
    persist_dir = "rag_chroma_store"

    # ----- Remove old vector store safely -----
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir, ignore_errors=True)
        except Exception as e:
            st.warning(f"Could not clear old store: {e}")

    # ----- Save uploaded file temporarily -----
    temp_dir = tempfile.TemporaryDirectory()
    file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ----- Extract text -----
    st.info("📑 Extracting text…")
    text = ""
    if uploaded_file.name.lower().endswith(".pdf"):
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {e}")
            st.stop()
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            st.error(f"Failed to read TXT: {e}")
            st.stop()

    if not text.strip():
        st.error("❌ No extractable text found.")
        st.stop()

    # ----- Chunk text -----
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # ----- Embeddings & Vectorstore -----
    with st.spinner("🔎 Building embeddings (first time may take a minute)…"):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=persist_dir)

    # ----- Retriever -----
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.3})

    st.success("✅ File processed successfully — ask your question 👇")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("🤔 Thinking…"):
            # Retrieve relevant docs
            retrieved_docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Prompt
            prompt = (
                "You are a helpful AI assistant for document Q&A. "
                "Answer the user's question using ONLY the provided context. "
                "If the answer is not present in the context, reply exactly with: I don't know.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n"
                "Answer:"
            )

            try:
                response = model.generate_content(prompt)
                answer = (response.text or "").strip()
            except Exception as e:
                st.error(f"Generation failed: {e}")
                answer = ""

        st.markdown("### 💬 Answer")
        if answer:
            st.write(answer)
        else:
            st.write("I don't know")

        # Show unique source chunks
        seen = set()
        unique_sources = []
        for doc in retrieved_docs:
            content = doc.page_content.strip()
            if content and content not in seen:
                seen.add(content)
                unique_sources.append(content)

        if unique_sources:
            with st.expander("📄 Source chunks"):
                for i, src in enumerate(unique_sources, start=1):
                    st.markdown(f"**Source {i}:** {src[:500]}{'...' if len(src) > 500 else ''}")
