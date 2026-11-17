import os
from functools import lru_cache

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Two retrieval methods, so they can be compared side by side:
#   "dense" - sentence embeddings + FAISS (matches meaning, even with different words)
#   "tfidf" - TF-IDF keyword matching (lexical baseline: only matches shared words)
METHODS = ("dense", "tfidf")

# Scores below these are treated as "not relevant" in the app.
MIN_SCORE = {"dense": 0.10, "tfidf": 0.01}


@lru_cache(maxsize=1)
def get_encoder():
    """Load the embedding model once and reuse it (first call downloads ~90 MB)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


class ScholarRAGEngine:
    def __init__(self):
        self.chunks = []
        # Lexical baseline
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        # Dense index
        self.faiss_index = None

    def load_pdf(self, pdf_file):
        reader = PdfReader(pdf_file)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n[Page {page_num + 1}] " + text
        return self.process_text(full_text)

    def process_text(self, text: str, chunk_size: int = 120, overlap: int = 25):
        """Split text into fixed-size word windows with overlap, then build both indexes."""
        words = text.split()
        self.chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                self.chunks.append(chunk)

        self.tfidf_matrix = None
        self.faiss_index = None
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            self._build_dense_index()
        return len(self.chunks)

    def _embed(self, texts):
        vectors = get_encoder().encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")

    def _build_dense_index(self):
        import faiss
        embeddings = self._embed(self.chunks)
        # Vectors are L2-normalised, so inner product == cosine similarity.
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3, method: str = "dense", min_score=None):
        """Return the top_k chunks for a query.

        method:    "dense" or "tfidf"
        min_score: drop results scoring below this. Defaults to MIN_SCORE[method];
                   pass 0 or a negative number to keep everything (used by evaluate.py).
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if not self.chunks:
            return []
        if min_score is None:
            min_score = MIN_SCORE[method]
        top_k = min(top_k, len(self.chunks))

        if method == "tfidf":
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            order = scores.argsort()[::-1][:top_k]
            ranked = [(int(i), float(scores[i])) for i in order]
        else:
            sims, ids = self.faiss_index.search(self._embed([query]), top_k)
            ranked = [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i != -1]

        return [
            {"chunk_id": idx + 1, "score": round(score, 4), "content": self.chunks[idx]}
            for idx, score in ranked
            if score >= min_score
        ]

    def generate_answer(self, query: str, retrieved_chunks: list):
        if not retrieved_chunks:
            return "No relevant passages found in the document for your query."

        context_text = "\n\n".join(
            [f"[Passage {c['chunk_id']}]: {c['content']}" for c in retrieved_chunks]
        )
        api_key = os.getenv("GEMINI_API_KEY")

        if api_key and api_key != "YOUR_COPIED_KEY_HERE":
            try:
                client = genai.Client(api_key=api_key)
                prompt = (
                    "You are an AI academic research assistant. Answer the user's question "
                    "accurately using ONLY the provided research paper passages. "
                    "Always cite the passage numbers like [Passage 1] in your response.\n\n"
                    f"Context Passages:\n{context_text}\n\n"
                    f"Question: {query}"
                )
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
                return response.text
            except Exception:
                pass

        synthesized = "Based on the retrieved passages from the research paper:\n\n"
        for c in retrieved_chunks:
            synthesized += f"• **[Passage {c['chunk_id']}]** (Similarity: `{c['score']}`): {c['content']}\n\n"
        return synthesized