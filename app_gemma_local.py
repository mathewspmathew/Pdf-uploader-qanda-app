# app_gemma_rag_local.py
import os
import shutil
import tempfile
import streamlit as st
import pdfplumber
import torch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ----------------- Config / UI -----------------
st.set_page_config(page_title="PDF/Doc Q&A (Gemma 1B Local)", page_icon="🟣", layout="wide")
st.title("📄 Document Q&A — Google Gemma 1B (Local)")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    persist_dir = "rag_chroma_store"

    # ----- Remove old vector store safely -----
    if os.path.exists(persist_dir):
        try:
            old_vs = Chroma(persist_directory=persist_dir, embedding_function=None)
            old_vs.delete_collection()
            del old_vs
        except Exception as e:
            st.warning(f"Could not clear old store: {e}")
        shutil.rmtree(persist_dir, ignore_errors=True)

    # ----- Save uploaded file temporarily -----
    temp_dir = tempfile.TemporaryDirectory()
    file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ----- Extract text -----
    st.info("Extracting text...")
    text = ""
    if uploaded_file.name.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text.strip():
        st.error("No extractable text found.")
        st.stop()

    # ----- Chunk text -----
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # ----- Embeddings & Vectorstore -----
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=persist_dir)

    # ----- Load Local Gemma -----
    model_path = "google/gemma-3-1b-it"
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0,
        device=device
    )

    # ----- Retriever -----
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.3})

    st.success("File processed successfully — ask your question 👇")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Thinking..."):
            # Retrieve relevant docs
            retrieved_docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Chat template prompt
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Answer based ONLY on the given context. If the answer is not in the context, say 'I don't know'."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Generate answer
            raw_output = generation_pipeline(prompt_text)[0]['generated_text']
            # Remove the chat template from answer
            answer = raw_output.replace(prompt_text, "").strip()

        st.markdown("### 💬 Answer")
        st.write(answer)

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
