import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from dotenv import load_dotenv

load_dotenv()

class ScholarRAGEngine:
    def __init__(self):
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None

    def load_pdf(self, pdf_file):
        reader = PdfReader(pdf_file)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n[Page {page_num + 1}] " + text
        return self.process_text(full_text)

    def process_text(self, text: str, chunk_size: int = 120):
        words = text.split()
        self.chunks = []
        for i in range(0, len(words), chunk_size - 25):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                self.chunks.append(chunk)

        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3):
        if not self.chunks or self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:
                results.append({
                    "chunk_id": int(idx + 1),
                    "score": round(float(similarities[idx]), 4),
                    "content": self.chunks[idx]
                })
        return results

    def generate_answer(self, query: str, retrieved_chunks: list):
        if not retrieved_chunks:
            return "No relevant passages found in the document for your query."

        context_text = "\n\n".join(
            [f"[Passage {c['chunk_id']}]: {c['content']}" for c in retrieved_chunks]
        )
        api_key = os.getenv("GEMINI_API_KEY")

        if api_key and api_key != "YOUR_COPIED_KEY_HERE":
            client = genai.Client(api_key=api_key)
            prompt = (
                "You are an AI academic research assistant. Answer the user's question "
                "accurately using ONLY the provided research paper passages. "
                "Always cite the passage numbers like [Passage 1] in your response.\n\n"
                f"Context Passages:\n{context_text}\n\n"
                f"Question: {query}"
            )
            # অটোমেটিক সাপোর্টেড মডেল ট্রাই করবে
            candidate_models = [
                "gemini-3-flash-preview",
                "gemini-3.0-flash",
                "gemini-2.0-flash",
                "gemini-1.5-flash"
            ]
            for model_name in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    return response.text
                except Exception:
                    continue

            # যদি লিস্টের বাইরে নতুন কোনো মডেল থাকে, তবে API থেকেই নাম খুঁজে নেবে
            try:
                for m in client.models.list():
                    if "flash" in m.name or "gemini" in m.name:
                        res = client.models.generate_content(
                            model=m.name,
                            contents=prompt
                        )
                        return res.text
            except Exception:
                pass

        # যদি API Key না থাকে বা অফলাইনে চলে, তবে কোনো লাল/কালো এরর না দেখিয়ে ক্লিন উত্তর দেবে!
        synthesized = "Based on the retrieved vector passages from the research paper:\n\n"
        for c in retrieved_chunks:
            synthesized += f"• **[Passage {c['chunk_id']}]** (Similarity: `{c['score']}`): {c['content']}\n\n"
        return synthesized