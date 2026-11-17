import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from app.rag_engine import ScholarRAGEngine
from app.sample import SAMPLE_PAPER

st.set_page_config(page_title="ScholarRAG - Academic Assistant", page_icon="📚", layout="wide")

st.title("📚 ScholarRAG: AI Research Paper & Academic Assistant")
st.markdown("Upload any academic PDF or load the built-in sample paper to retrieve relevant passages (dense embeddings + FAISS, or a TF-IDF keyword baseline) and generate cited answers using **Gemini LLM**.")

if "rag" not in st.session_state:
    st.session_state.rag = ScholarRAGEngine()
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

RETRIEVERS = {
    "Dense embeddings (MiniLM + FAISS)": "dense",
    "TF-IDF keyword baseline": "tfidf",
}

with st.sidebar:
    st.header("🔎 Retrieval Method")
    method_label = st.radio("Retrieve passages with:", list(RETRIEVERS.keys()))
    method = RETRIEVERS[method_label]
    st.markdown("---")

    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Index Uploaded PDF", type="primary"):
            with st.spinner("Extracting text & building indexes (first run downloads the embedding model)..."):
                count = st.session_state.rag.load_pdf(uploaded_file)
                st.session_state.doc_loaded = True
                st.session_state.chunk_count = count
                st.success(f"Indexed {count} text chunks successfully!")

    st.markdown("---")
    st.subheader("⚡ Instant Demo")
    if st.button("Load Sample Research Paper"):
        with st.spinner("Building indexes (first run downloads the embedding model)..."):
            count = st.session_state.rag.process_text(SAMPLE_PAPER, chunk_size=45)
        st.session_state.doc_loaded = True
        st.session_state.chunk_count = count
        st.success(f"Sample paper loaded ({count} chunks indexed)!")

if st.session_state.doc_loaded:
    st.info(f"✅ Active index: **{st.session_state.chunk_count}** searchable text chunks ready.")
    
    query = st.text_input(
        "🔍 Ask a question about the research paper:",
        placeholder="e.g., How many images were used in the dataset and how was explainability achieved?"
    )
    
    if st.button("Search & Generate Answer", type="primary") and query:
        with st.spinner("Retrieving passages & generating cited answer..."):
            retrieved = st.session_state.rag.retrieve(query, top_k=3, method=method)
            answer = st.session_state.rag.generate_answer(query, retrieved)
            
            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("🤖 AI Synthesized Answer")
                st.write(answer)
            with col2:
                st.subheader(f"📌 Retrieved Passages ({method_label})")
                for item in retrieved:
                    with st.expander(f"Passage {item['chunk_id']} (Similarity Score: {item['score']})"):
                        st.write(item['content'])
else:
    st.warning("👈 Please upload a PDF or click **'Load Sample Research Paper'** in the left sidebar to begin.")