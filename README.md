# ScholarRAG: AI Research Paper & Academic Assistant

A Retrieval-Augmented Generation (RAG) document assistant built with **Python**, **Sentence-Transformers (`all-MiniLM-L6-v2`)**, **FAISS**, **Scikit-learn**, **Google Gemini**, and **Streamlit**. ScholarRAG ingests academic PDFs, splits text into overlapping word windows, retrieves top-k passages using either dense vector embeddings or a lexical TF-IDF baseline, and generates source-cited answers.

![ScholarRAG Application Preview](demo.png)

---

## Key Features
* **PDF Ingestion & Windowed Chunking:** Extracts text from multi-page academic PDFs with page tracking and splits it into overlapping word windows (120 words, 25-word overlap).
* **Dual Retrieval Engine (Dense vs. Lexical):**
  * **Dense Semantic Search:** Encodes chunks with `sentence-transformers/all-MiniLM-L6-v2` (384-dim L2-normalized vectors) and searches via **FAISS** inner-product index (`IndexFlatIP`).
  * **Lexical Baseline:** Computes **TF-IDF** vectors and ranks chunks by cosine similarity for direct keyword comparison.
* **Source-Cited LLM Synthesis:** Uses Google Gemini (`gemini-2.0-flash` via `google-genai`) to synthesize answers strictly grounded in retrieved passages with inline citations (`[Passage 1]`, `[Passage 2]`).
* **Extractive Offline Fallback:** Automatically returns the highest-scoring retrieved passages with similarity scores when run without an API key.

---

## Project Architecture

    scholar-rag-assistant/
    ├── app/
    │   ├── __init__.py
    │   ├── pdf_parser.py    # Multi-page PDF text extraction & overlapping word chunking
    │   ├── retriever.py     # Dual Dense (MiniLM + FAISS) & Lexical (TF-IDF) retrieval engine
    │   └── generator.py     # Google Gemini source-cited synthesis + extractive fallback
    ├── eval/
    │   ├── questions.json           # 16 factual & paraphrased benchmark questions
    │   ├── questions.example.json   # Template for custom PDF evaluation sets
    │   └── results.md               # Generated retrieval comparison report
    ├── evaluate.py          # Automated Hit@1, Hit@3, and MRR benchmark runner
    ├── main.py              # Interactive Streamlit web application
    ├── requirements.txt     # Project dependencies
    ├── demo.png             # Application UI screenshot
    ├── README.md            # Project documentation
    └── .gitignore

---

## Retrieval Benchmark (Dense FAISS vs. TF-IDF Baseline)

Evaluated on the built-in agricultural computer vision sample paper (9 chunks) across 16 factual and paraphrased questions using `evaluate.py`:

| Retrieval method | Hit@1 | Hit@3 | MRR |
| :--- | :---: | :---: | :---: |
| TF-IDF (keyword baseline) | 75% | 88% | 0.82 |
| Dense embeddings (MiniLM + FAISS) | **75%** | **100%** | **0.86** |

* **Observation:** While TF-IDF performs well when query terms match the exact wording in the paper, dense embeddings (`all-MiniLM-L6-v2` + FAISS) consistently rank paraphrased queries higher (e.g., matching *"input photos resized to"* against *"standardized to 300x300 pixels"* at Rank 1 vs. Rank 6 for TF-IDF), achieving **100% Hit@3**.

Reproduce the benchmark locally:

    python evaluate.py --sample
    # Or evaluate on your own PDF and custom question set:
    python evaluate.py --pdf paper.pdf --questions eval/questions.json

---

## Tech Stack
* **Embeddings & Vector Index:** `sentence-transformers` (`all-MiniLM-L6-v2`), `faiss-cpu`, PyTorch (CPU)
* **Lexical Baseline & PDF Parsing:** `scikit-learn` (`TfidfVectorizer`), `pypdf`
* **LLM Integration:** Google Gemini API (`google-genai`), `python-dotenv`
* **Frontend UI:** `streamlit`

---

## Quick Start (Installation & Setup)

1. **Clone the repository:**

       git clone https://github.com/mdramjanalimondol002/scholar-rag-assistant.git
       cd scholar-rag-assistant

2. **Create and activate a virtual environment (Optional):**

       python -m venv .venv
       # On Windows:
       .venv\Scripts\activate
       # On macOS/Linux:
       source .venv/bin/activate

3. **Install dependencies:**

       pip install -r requirements.txt

4. **Configure Gemini API Key (Optional):**
   Create a `.env` file in the project root to enable LLM synthesis (the app works offline via extractive passage retrieval if omitted):

       GEMINI_API_KEY=your_api_key_here

5. **Launch the Streamlit App:**

       python -m streamlit run main.py

---

## How to Use the Application (Step-by-Step Guide)

1. **Start the Web Server:** Run `python -m streamlit run main.py` from the `scholar-rag-assistant` directory and open `http://localhost:8501` in your browser.
2. **Load a Document:** Upload any academic PDF via the left sidebar, or check **Use Built-in Sample Research Paper** to test immediately without uploading a file.
3. **Select Retrieval Mode & Top-K:** Choose between **Dense Embeddings (MiniLM + FAISS)** or **TF-IDF (Keyword Baseline)** and adjust the number of retrieved chunks (`Top-K`).
4. **Ask Research Questions:** Type a specific question about methodology, dataset size, hyperparameters, or results and click **Ask ScholarRAG** to inspect the synthesized answer, inline passage citations, and similarity scores.