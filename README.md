# Simple Streamlit AI Document Assistant

An intuitive Retrieval-Augmented Generation (RAG) assistant that allows users to query PDF, DOCX, TXT, and Markdown files uploaded locally or fetched from Google Drive using hybrid search (Semantic + Keyword) and Groq LLMs.

## Features
- **Multi-Source Ingestion**: Local uploads or public Google Drive files/folders.
- **Document Extractors**: Preserves metadata (`filename`, `page number`).
- **Hybrid RAG Search**: Combines Sentence Transformers + FAISS vector search with keyword matching.
- **Strict Context Prompting**: Groq LLM answers strictly based on the provided context.
- **Source Attribution**: Transparent display of source documents and pages below each answer.
- **Cached Memory**: Embeddings and vector indices are stored in Streamlit `session_state` so questions can be asked repeatedly without re-processing.

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A Groq API Key (obtain from [Groq Console](https://console.groq.com/))

### 2. Installation
Clone or copy the files into a single directory, then install the required packages:

```bash
pip install -r requirements.txt