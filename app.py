import os
import re
import tempfile
from typing import List, Dict, Any, Tuple
import streamlit as st
import numpy as np
import pypdf
import docx
import gdown
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")

@st.cache_resource
def load_embedding_model():
    """Cache the embedding model so it's loaded only once."""
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Initialize Session State
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# -----------------------------------------------------------------------------
# 1. Document Extraction Functions
# -----------------------------------------------------------------------------
def extract_txt(file_path: str, filename: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text, "filename": filename, "page": "N/A"}]

def extract_pdf(file_path: str, filename: str) -> List[Dict[str, Any]]:
    documents = []
    reader = pypdf.PdfReader(file_path)
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append({"text": text, "filename": filename, "page": idx + 1})
    return documents

def extract_docx(file_path: str, filename: str) -> List[Dict[str, Any]]:
    doc = docx.Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [{"text": text, "filename": filename, "page": "N/A"}]

def process_single_file(file_path: str, filename: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".txt", ".md"]:
        return extract_txt(file_path, filename)
    elif ext == ".pdf":
        return extract_pdf(file_path, filename)
    elif ext == ".docx":
        return extract_docx(file_path, filename)
    return []

# -----------------------------------------------------------------------------
# 2. Text Chunking
# -----------------------------------------------------------------------------
def chunk_documents(docs: List[Dict[str, Any]], chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    chunks = []
    for doc in docs:
        text = doc["text"]
        filename = doc["filename"]
        page = doc["page"]
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "filename": filename,
                    "page": page
                })
            start += chunk_size - overlap
    return chunks

# -----------------------------------------------------------------------------
# 3. Embedding & FAISS Index Building
# -----------------------------------------------------------------------------
def build_vector_store(chunks: List[Dict[str, Any]]):
    if not chunks:
        return
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.faiss_index = index

# -----------------------------------------------------------------------------
# 4. Search Mechanisms
# -----------------------------------------------------------------------------
def semantic_search(query: str, k: int = 10) -> List[Tuple[int, float]]:
    if st.session_state.faiss_index is None:
        return []
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    scores, indices = st.session_state.faiss_index.search(query_vector, k)
    return list(zip(indices[0], scores[0]))

def keyword_search(query: str) -> Dict[int, float]:
    keywords = set(re.findall(r'\w+', query.lower()))
    if not keywords:
        return {}
    
    scores = {}
    for idx, chunk in enumerate(st.session_state.chunks):
        text_words = re.findall(r'\w+', chunk["text"].lower())
        if not text_words:
            continue
        matches = sum(1 for word in text_words if word in keywords)
        score = matches / len(text_words)
        if score > 0:
            scores[idx] = score
    return scores

def hybrid_search(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    if not st.session_state.chunks:
        return []
    
    sem_results = dict(semantic_search(query, k=min(20, len(st.session_state.chunks))))
    kw_results = keyword_search(query)
    
    combined_scores = {}
    max_kw = max(kw_results.values()) if kw_results else 1.0
    
    all_indices = set(sem_results.keys()).union(set(kw_results.keys()))
    for idx in all_indices:
        sem_score = sem_results.get(idx, 0.0)
        kw_score = (kw_results.get(idx, 0.0) / max_kw) if max_kw > 0 else 0.0
        combined_scores[idx] = (0.7 * sem_score) + (0.3 * kw_score)
    
    sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    retrieved_chunks = []
    for idx, score in sorted_indices:
        chunk = st.session_state.chunks[idx].copy()
        chunk["score"] = score
        retrieved_chunks.append(chunk)
        
    return retrieved_chunks

# -----------------------------------------------------------------------------
# 5. Groq Integration
# -----------------------------------------------------------------------------
def answer_question(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY is missing. Please configure `.streamlit/secrets.toml` or set it in Streamlit Cloud secrets."
    
    client = Groq(api_key=api_key)
    context_str = "\n\n".join(
        [f"[Source {i+1} | {c['filename']} (Page {c['page']})]:\n{c['text']}" for i, c in enumerate(retrieved_chunks)]
    )
    
    prompt = f"""You are a helpful AI assistant. Answer the user's question ONLY using the provided document context below. 
If the information required to answer the question is not present in the context, clearly state that the information is not available in the uploaded documents.

Context:
{context_str}

Question: {query}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Groq API: {str(e)}"

# -----------------------------------------------------------------------------
# 6. Streamlit User Interface
# -----------------------------------------------------------------------------
st.title("📚 AI Document Assistant")
st.markdown("Upload documents locally or import from Google Drive to chat with your knowledge base.")

# -----------------------------------------------------------------------------
# Updated Streamlit UI & Document Ingestion
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Add Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files", 
        type=["pdf", "docx", "txt", "md"], 
        accept_multiple_files=True
    )
    
    st.subheader("Or Google Drive Link")
    st.caption("⚠️ File/Folder MUST be set to 'Anyone with the link can view'")
    drive_url = st.text_input("Folder or File Link:")
    process_btn = st.button("Process Documents", type="primary")

# Robust Document Processing Workflow
if process_btn:
    all_raw_docs = []
    
    with st.spinner("Extracting text from documents..."):
        # 1. Process Local Uploads
        if uploaded_files:
            for file in uploaded_files:
                ext = os.path.splitext(file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                extracted = process_single_file(tmp_path, file.name)
                if extracted:
                    all_raw_docs.extend(extracted)
                else:
                    st.sidebar.warning(f"Could not extract text from: {file.name}")
                os.remove(tmp_path)
                
        # 2. Process Google Drive Link
        if drive_url.strip():
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Case A: Google Drive Folder
                    if "folders" in drive_url or "drive/folders" in drive_url:
                        downloaded_files = gdown.download_folder(
                            url=drive_url, 
                            output=temp_dir, 
                            quiet=True, 
                            remaining_ok=True
                        )
                    # Case B: Single File (Shared Link, Google Doc, PDF, etc.)
                    else:
                        downloaded_path = gdown.download(
                            url=drive_url, 
                            output=os.path.join(temp_dir, ""), 
                            quiet=True, 
                            fuzzy=True
                        )
                    
                    # Inspect temp directory for downloaded files
                    found_any = False
                    for root, _, files in os.walk(temp_dir):
                        for fname in files:
                            found_any = True
                            fpath = os.path.join(root, fname)
                            extracted = process_single_file(fpath, fname)
                            if extracted:
                                all_raw_docs.extend(extracted)
                            else:
                                st.sidebar.warning(f"Skipped unsupported or empty file: {fname}")
                    
                    if not found_any:
                        st.sidebar.error("Google Drive link was downloaded, but no files were found inside.")

            except Exception as e:
                st.sidebar.error(f"Error accessing Google Drive link: {e}")

    # 3. Vectorization Check
    if all_raw_docs:
        with st.spinner("Chunking & generating embeddings..."):
            chunks = chunk_documents(all_raw_docs)
            if chunks:
                build_vector_store(chunks)
                st.sidebar.success(f"Success! Embedded {len(all_raw_docs)} document section(s) into {len(chunks)} chunks.")
            else:
                st.sidebar.warning("Text extracted, but resulted in 0 chunks. Try adjusting chunk size.")
    else:
        st.sidebar.error("No valid text extracted. Please check file formats or Google Drive permissions.")

# Question Answering Interface
st.divider()
if st.session_state.chunks:
    st.info(f"🟢 Knowledge Base Active: **{len(st.session_state.chunks)} text chunks** ready.")
    query = st.text_input("Ask a question about your documents:")
    
    if query.strip():
        with st.spinner("Searching and generating answer..."):
            retrieved = hybrid_search(query, top_k=4)
            answer = answer_question(query, retrieved)
            
        st.subheader("Answer")
        st.write(answer)
        
        st.divider()
        st.subheader("Retrieved Sources")
        for idx, chunk in enumerate(retrieved):
            with st.expander(f"Source {idx + 1}: {chunk['filename']} (Page: {chunk['page']}) | Relevance Score: {chunk['score']:.2f}"):
                st.write(chunk["text"])
else:
    st.info("👆 Please upload files or provide a Google Drive link in the sidebar to get started.")
