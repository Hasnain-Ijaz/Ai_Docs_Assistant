# 📚 AI Multi-Document Assistant

A Retrieval-Augmented Generation (RAG) Streamlit application that allows users to query PDF, DOCX, TXT, and Markdown files loaded either locally or from public Google Drive links. The app uses Sentence Transformers and FAISS for semantic search, keyword matching for exact hits, Groq (`llama-3.3-70b-versatile`) for context-bounded generation, and automatically suggests relevant questions upon document upload.

---

## ✨ Features

- **Multi-Source Ingestion**: Process multiple local files simultaneously or paste multiple Google Drive links (files or folders).
- **Document Extractors**: Extracts text from `.pdf`, `.docx`, `.txt`, and `.md` formats while keeping track of `filename` and `page` metadata.
- **Hybrid RAG Search**: Blends dense semantic vector search (70% weight using Sentence Transformers + FAISS) and sparse keyword matching (30% weight) to return the most relevant document chunks.
- **AI-Suggested Quick Questions**: Generates 3 contextual questions automatically from document excerpts upon processing.
- **Grounded LLM Responses**: Utilizes Groq (`llama-3.3-70b-versatile`) to answer queries strictly based on provided context.
- **Source Attribution**: Displays transparent expanders showing retrieved source text, file names, page numbers, and relevance scores.
- **Session Caching**: Stores FAISS indices and vector embeddings in Streamlit `session_state` so you can ask multiple questions without re-embedding.
- **Reset Feature**: Clear the knowledge base anytime to start fresh.

---

## 📁 Repository Structure

```text
streamlit-doc-assistant/
├── .streamlit/
│   └── secrets.toml          # Local secrets configuration for API keys
├── app.py                    # Main Streamlit application
├── requirements.txt          # Project dependencies
└── README.md                 # Documentation
