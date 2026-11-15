import streamlit as st
from app.rag_engine import ScholarRAGEngine

st.set_page_config(page_title="ScholarRAG - Academic Assistant", page_icon="📚", layout="wide")

st.title("📚 ScholarRAG: AI Research Paper & Academic Assistant")
st.markdown("Upload any academic PDF or load the built-in sample paper to retrieve relevant passages via vector search and generate cited answers using **Gemini LLM**.")

if "rag" not in st.session_state:
    st.session_state.rag = ScholarRAGEngine()
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

SAMPLE_PAPER = """
[Page 1] Title: Deep Learning and Explainable AI in Agricultural Leaf Disease Diagnosis.
Abstract: Plant diseases significantly reduce global crop yield and quality. Recent advances in Convolutional Neural Networks (CNNs) such as ResNet-50, MobileNetV2, and EfficientNetV2 have enabled automated disease classification from RGB leaf images captured in field conditions.
[Page 2] Methodology: A dataset of 4,267 original RGB images across six distinct crop disease categories was curated under natural lighting conditions and standardized to 300x300 pixels. Data augmentation expanded the corpus to 21,335 images to prevent overfitting.
[Page 3] Explainability & Results: To ensure clinical and agricultural trust, Grad-CAM and LIME visual explanation techniques were integrated. These methods generate heatmaps highlighting the exact necrotic lesions and chlorotic spots driving each model's prediction, achieving 96.4% classification accuracy across validation folds.
"""

with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Index Uploaded PDF", type="primary"):
            with st.spinner("Extracting text & building vector index..."):
                count = st.session_state.rag.load_pdf(uploaded_file)
                st.session_state.doc_loaded = True
                st.session_state.chunk_count = count
                st.success(f"Indexed {count} text chunks successfully!")

    st.markdown("---")
    st.subheader("⚡ Instant Demo")
    if st.button("Load Sample Research Paper"):
        count = st.session_state.rag.process_text(SAMPLE_PAPER, chunk_size=45)
        st.session_state.doc_loaded = True
        st.session_state.chunk_count = count
        st.success(f"Sample paper loaded ({count} vector chunks indexed)!")

if st.session_state.doc_loaded:
    st.info(f"✅ Active Vector Index: **{st.session_state.chunk_count}** searchable text chunks ready.")
    
    query = st.text_input(
        "🔍 Ask a question about the research paper:",
        placeholder="e.g., How many images were used in the dataset and how was explainability achieved?"
    )
    
    if st.button("Search & Generate Answer", type="primary") and query:
        with st.spinner("Searching vector space & generating cited answer..."):
            retrieved = st.session_state.rag.retrieve(query, top_k=3)
            answer = st.session_state.rag.generate_answer(query, retrieved)
            
            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("🤖 AI Synthesized Answer")
                st.write(answer)
            with col2:
                st.subheader("📌 Retrieved Vector Passages")
                for item in retrieved:
                    with st.expander(f"Passage {item['chunk_id']} (Similarity Score: {item['score']})"):
                        st.write(item['content'])
else:
    st.warning("👈 Please upload a PDF or click **'Load Sample Research Paper'** in the left sidebar to begin.")