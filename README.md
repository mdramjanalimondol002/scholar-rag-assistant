# ScholarRAG: AI Research Paper & Academic Assistant

A Retrieval-Augmented Generation (RAG) document intelligence system built with **Python**, **Scikit-learn Vector Search**, **Google Gemini LLM**, and **Streamlit**. ScholarRAG ingests academic PDFs, splits text into overlapping semantic chunks, retrieves top-k relevant passages using TF-IDF cosine similarity, and generates source-cited answers.

![ScholarRAG Application Preview](demo.png)

## Key Features
* **Automated PDF Ingestion & Chunking:** Extracts text from multi-page academic PDFs with page tracking and sliding-window chunking.
* **Vector Similarity Search:** Computes TF-IDF vector embeddings and ranks passages via Cosine Similarity scores.
* **Source-Cited LLM Generation:** Integrates Google Gemini (`google-genai`) to synthesize accurate answers strictly grounded in retrieved passages with inline citations (`[Passage 1]`, `[Passage 2]`).
* **Resilient Extractive Fallback:** Works seamlessly offline by returning the highest-scoring vector passages with similarity scores if no API key is configured.

## Tech Stack
* **LLM & GenAI:** Google Gemini API (`google-genai`), Prompt Engineering, RAG
* **Vector Search & NLP:** Scikit-learn (`TfidfVectorizer`, `cosine_similarity`), PyPDF
* **Frontend UI:** Streamlit

---

## Quick Start (Installation & Setup)

1. **Clone the repository & navigate to the project folder:**
   `git clone https://github.com/mdramjanalimondol002/scholar-rag-assistant.git`
   `cd scholar-rag-assistant`

2. **Install required dependencies:**
   `pip install -r requirements.txt`

3. **Configure Gemini API Key (Optional):**
   Create a `.env` file in the root directory and add your key:
   `GEMINI_API_KEY=your_api_key_here`

4. **Run the Streamlit Web Application:**
   `python -m streamlit run main.py`

---

## How to Use the Application (Step-by-Step Guide)

1. **Ingest a Research Paper:**
   * Open `http://localhost:8501` in your browser.
   * Upload any academic research paper (`.pdf`) from the left sidebar and click **"Index Uploaded PDF"**, OR click **"Load Sample Research Paper"** under Instant Demo for immediate testing.

2. **Query the Vector Index:**
   * Type your research question in the search box (e.g., *"How many images were used in the dataset and which explainable AI methods were applied?"*).
   * Click **"Search & Generate Answer"**.

3. **Verify Citations & Similarity Scores:**
   * Read the synthesized answer with passage citations on the left panel.
   * Expand the **Retrieved Vector Passages** on the right panel to inspect the exact source chunks and their cosine similarity scores.