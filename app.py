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
st.set_page_config(page_title="AI Multi-Document Assistant", page_icon="📚", layout="wide")


@st.cache_resource
def load_embedding_model():
    """Cache the embedding model so it's loaded only once per server runtime."""
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

# Initialize Session State
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []


# -----------------------------------------------------------------------------
# Helper: Clear Knowledge Base State
# -----------------------------------------------------------------------------
def reset_knowledge_base():
    """Resets vector index, stored chunks, embeddings, and suggested questions."""
    st.session_state.chunks = []
    st.session_state.faiss_index = None
    st.session_state.embeddings = None
    st.session_state.suggested_questions = []


# -----------------------------------------------------------------------------
# 1. Document Extraction Functions
# -----------------------------------------------------------------------------
def extract_txt(file_path: str, filename: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text, "filename": filename, "page": "N/A"}] if text.strip() else []


def extract_pdf(file_path: str, filename: str) -> List[Dict[str, Any]]:
    documents = []
    try:
        reader = pypdf.PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append({"text": text, "filename": filename, "page": idx + 1})
    except Exception:
        pass
    return documents


def extract_docx(file_path: str, filename: str) -> List[Dict[str, Any]]:
    try:
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return [{"text": text, "filename": filename, "page": "N/A"}] if text.strip() else []
    except Exception:
        return []


def process_single_file(file_path: str, filename: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(filename)[1].lower()

    if ext in [".txt", ".md"]:
        return extract_txt(file_path, filename)
    elif ext == ".pdf":
        return extract_pdf(file_path, filename)
    elif ext == ".docx":
        return extract_docx(file_path, filename)

    # Fallback auto-detection for missing or unknown extensions
    pdf_docs = extract_pdf(file_path, filename)
    if pdf_docs:
        return pdf_docs

    docx_docs = extract_docx(file_path, filename)
    if docx_docs:
        return docx_docs

    return extract_txt(file_path, filename)


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
# 4. Search Mechanisms (Semantic, Keyword, Hybrid)
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


def hybrid_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
# 5. Groq Integration & Automatic Question Suggestions
# -----------------------------------------------------------------------------
def generate_suggested_questions(chunks: List[Dict[str, Any]]) -> List[str]:
    """Generates 3 practical questions based on representative document chunks."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key or not chunks:
        return []

    step = max(1, len(chunks) // 5)
    sample_chunks = chunks[::step][:5]
    sample_text = "\n\n".join([c["text"] for c in sample_chunks])

    prompt = f"""Based on the following document excerpts, generate 3 clear, concise, and practical questions that a user might ask about this text.
Return ONLY the 3 questions as a numbered list (1., 2., 3.). Do not include any introductory or concluding text.

Document Sample:
{sample_text}"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        lines = response.choices[0].message.content.strip().split("\n")
        questions = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines if line.strip()]
        return questions[:3]
    except Exception:
        return []


def answer_question(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY is missing. Please configure `.streamlit/secrets.toml` or set it in your environment variables."

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
# 6. Ingestion Helpers
# -----------------------------------------------------------------------------
def ingest_uploaded_files(uploaded_files) -> List[Dict[str, Any]]:
    raw_docs = []
    for file in uploaded_files:
        ext = os.path.splitext(file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        extracted = process_single_file(tmp_path, file.name)
        if extracted:
            raw_docs.extend(extracted)
        else:
            st.sidebar.warning(f"Could not extract text from local file: {file.name}")
        os.remove(tmp_path)
    return raw_docs


def ingest_multiple_drive_links(drive_urls_input: str) -> List[Dict[str, Any]]:
    urls = [url.strip() for url in re.split(r'[\n,\s]+', drive_urls_input) if url.strip()]
    raw_docs = []

    for idx, drive_url in enumerate(urls):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                if "folders" in drive_url or "drive/folders" in drive_url:
                    gdown.download_folder(url=drive_url, output=temp_dir, quiet=True, remaining_ok=True)
                else:
                    downloaded_file = gdown.download(url=drive_url, output=os.path.join(temp_dir, ""), quiet=True)
                    if not downloaded_file:
                        fallback_path = os.path.join(temp_dir, f"drive_doc_{idx}")
                        gdown.download(url=drive_url, output=fallback_path, quiet=True)

                found_any = False
                for root, _, files in os.walk(temp_dir):
                    for fname in files:
                        found_any = True
                        fpath = os.path.join(root, fname)
                        extracted = process_single_file(fpath, fname)
                        if extracted:
                            raw_docs.extend(extracted)

                if not found_any:
                    st.sidebar.warning(f"Link #{idx+1}: No downloadable files found.")
        except Exception as e:
            st.sidebar.error(f"Link #{idx+1} Error: {e}")

    return raw_docs


# -----------------------------------------------------------------------------
# 7. Streamlit User Interface
# -----------------------------------------------------------------------------
st.title("📚 AI Document Assistant")
st.markdown("Upload local documents or paste public Google Drive links to build your knowledge base.")

with st.sidebar:
    st.header("1. Upload Local Files")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files (Multiple allowed)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    st.subheader("2. Paste Google Drive Links")
    st.caption("Paste public file/folder links (one per line or comma-separated).")
    drive_urls_input = st.text_area("Folder or File Links:", height=120)

    process_btn = st.button("Process All Documents", type="primary", use_container_width=True)

    st.divider()

    # Clear Knowledge Base Button
    if st.button("🗑️ Clear Knowledge Base", type="secondary", use_container_width=True):
        reset_knowledge_base()
        st.success("Knowledge base cleared successfully!")
        st.rerun()

# Document Ingestion Processing Workflow
if process_btn:
    all_raw_docs = []

    with st.spinner("Extracting text from all documents..."):
        if uploaded_files:
            all_raw_docs.extend(ingest_uploaded_files(uploaded_files))

        if drive_urls_input.strip():
            all_raw_docs.extend(ingest_multiple_drive_links(drive_urls_input))

    if all_raw_docs:
        with st.spinner("Chunking & generating vector embeddings..."):
            chunks = chunk_documents(all_raw_docs)
            if chunks:
                build_vector_store(chunks)
                st.session_state.suggested_questions = generate_suggested_questions(chunks)
                st.sidebar.success(
                    f"Success! Processed {len(all_raw_docs)} document section(s) into {len(chunks)} searchable chunks."
                )
            else:
                st.sidebar.warning("Text extracted, but resulted in 0 chunks.")
    else:
        st.sidebar.error("No valid text extracted. Please check your files or Google Drive permissions.")

# Question Answering Interface
st.divider()
if st.session_state.chunks:
    st.info(f"🟢 Knowledge Base Active: **{len(st.session_state.chunks)} text chunks** ready across all loaded documents.")

    selected_query = ""

    # Display Suggested Quick Questions if available
    if st.session_state.suggested_questions:
        st.subheader("💡 Suggested Quick Questions")
        options = ["-- Select a suggested question --"] + st.session_state.suggested_questions + ["Type my own question..."]
        chosen_option = st.selectbox("Pick an AI-suggested question or select custom input:", options)

        if chosen_option not in ["-- Select a suggested question --", "Type my own question..."]:
            selected_query = chosen_option

    # Show custom text input box if selected or no quick questions exist
    if not selected_query:
        query_input = st.text_input("Ask a question about your documents:")
        if query_input.strip():
            selected_query = query_input

    # Execute Hybrid RAG Search and LLM Answer
    if selected_query:
        with st.spinner("Searching across documents and generating answer..."):
            retrieved = hybrid_search(selected_query, top_k=5)
            answer = answer_question(selected_query, retrieved)

        st.subheader("Answer")
        st.write(answer)

        st.divider()
        st.subheader("Retrieved Sources")
        for idx, chunk in enumerate(retrieved):
            with st.expander(
                f"Source {idx + 1}: {chunk['filename']} (Page: {chunk['page']}) | Relevance Score: {chunk['score']:.2f}"
            ):
                st.write(chunk["text"])
else:
    st.info("👆 Upload files or paste Google Drive links in the sidebar to begin.")
