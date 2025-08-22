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
