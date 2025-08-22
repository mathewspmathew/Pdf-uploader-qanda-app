# Ask questions to a PDF

This project is a local **RAG (Retrieval-Augmented Generation)** pipeline built with:
- **Google Gemma 1B (IT)** → for answer generation
- **all-MiniLM-L6-v2** → for text embeddings
- **ChromaDB** → for vector storage and retrieval
- **Streamlit** → for the UI

As for vector database: I used Chromadb for app_gemma_local.py and for Streamlit hosting, it doesn't allow Chromadb, so use FAISS by facebook.

   
Here I have concentrated on app_gemma_local.py
Other apps are just for understanding purpose.

Inside app.py, I am using google/flan-t5-base model as the LLM.

## 🤔 Why I used all-MiniLM-L6-v2?

-  **Small & fast** → Runs smoothly on both CPU and GPU  
-  **General-purpose semantic embeddings** → Works well across many domains  
-  **384-dimensional vectors** → Lower memory and storage cost  
-  **Balanced** → Great trade-off between retrieval quality and efficiency  

##  Reference Article

For a detailed walkthrough of building a local RAG app with Google Gemma 1B, check out the following article:

[Building a Local RAG App with Google Gemma 1B](https://medium.com/@mathewsparan/building-a-local-rag-app-with-google-gemma-1b-d6e1064b852c) – an excellent guide covering the architecture, tool choices, and workflow behind a similar implementation.

<img width="1799" height="721" alt="Screenshot 2025-08-16 184119" src="https://github.com/user-attachments/assets/ee6e9226-fe9d-4ace-b347-7672af8834ef" />
