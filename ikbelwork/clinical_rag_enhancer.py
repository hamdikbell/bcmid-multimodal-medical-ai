"""
Clinical RAG Enhancer (Bloc 4)
Loads both:
1. Patient historical report chunks (from BCMID)
2. Official ACR BI-RADS Clinical Guidelines (from acr_birads_guidelines.json)
Builds or queries a unified FAISS index with BGE embeddings.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


class ClinicalRAGEngine:
    def __init__(self, index_path: Optional[str] = None, chunks_path: Optional[str] = None):
        self.guidelines_path = Path(__file__).parent / "acr_birads_guidelines.json"
        self.guidelines = []
        if self.guidelines_path.exists():
            self.guidelines = json.loads(self.guidelines_path.read_text(encoding="utf-8"))

        self.chunks_df: Optional[pd.DataFrame] = None
        self.index = None
        self.embedder = None
        self._load_embedder()

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        except Exception as e:
            print(f"[RAG Engine] SentenceTransformer warning: {e}. Fallback to TF-IDF / dummy.")
            self.embedder = None

    def build_unified_knowledge_base(self, patient_reports_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Combines ACR guidelines and patient report chunks into a unified DataFrame."""
        rows: List[Dict[str, Any]] = []

        # 1. Add ACR Guidelines
        for g in self.guidelines:
            rows.append({
                "source_type": "ACR_GUIDELINE",
                "doc_id": g["doc_id"],
                "pseudo_id": "ACR_GUIDELINE",
                "category": g["category"],
                "fold": -1,  # Global, accessible to all folds
                "text": f"[{g['title']}] {g['text']}"
            })

        # 2. Add Patient Report Chunks if available
        if patient_reports_df is not None and not patient_reports_df.empty:
            for r in patient_reports_df.itertuples():
                text = getattr(r, "text", "") or getattr(r, "text_blind", "") or ""
                if len(text.split()) >= 20:
                    rows.append({
                        "source_type": "PATIENT_REPORT",
                        "doc_id": f"{getattr(r, 'pseudo_id', 'unknown')}_chunk",
                        "pseudo_id": getattr(r, "pseudo_id", "unknown"),
                        "category": "Historical Clinical Evidence",
                        "fold": getattr(r, "fold", -1),
                        "text": text
                    })

        self.chunks_df = pd.DataFrame(rows)
        return self.chunks_df

    def retrieve(self, query: str, top_k: int = 4, exclude_pseudo_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant evidence passages and clinical guidelines."""
        if self.chunks_df is None or self.chunks_df.empty:
            self.build_unified_knowledge_base()

        filtered_df = self.chunks_df
        if exclude_pseudo_id:
            filtered_df = filtered_df[filtered_df["pseudo_id"] != exclude_pseudo_id]

        if filtered_df.empty:
            return []

        # If embedding model is available, use cosine similarity
        if self.embedder is not None:
            try:
                q_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
                chunk_texts = filtered_df["text"].tolist()
                c_vecs = self.embedder.encode(chunk_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
                sims = np.dot(c_vecs, q_vec)
                top_indices = np.argsort(sims)[::-1][:top_k]

                results = []
                for rank, idx in enumerate(top_indices, 1):
                    row = filtered_df.iloc[idx]
                    results.append({
                        "evidence_id": f"E{rank}",
                        "source_type": row["source_type"],
                        "doc_id": row["doc_id"],
                        "similarity": round(float(sims[idx]), 3),
                        "passage": row["text"]
                    })
                return results
            except Exception as e:
                print(f"[RAG Retrieval] Embedding search error: {e}, falling back to keyword search.")

        # Fallback keyword overlap
        q_words = set(re.findall(r"\w+", query.lower()))
        scores = []
        for text in filtered_df["text"]:
            t_words = set(re.findall(r"\w+", text.lower()))
            overlap = len(q_words & t_words) / (len(q_words) + 1e-5)
            scores.append(overlap)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = filtered_df.iloc[idx]
            results.append({
                "evidence_id": f"E{rank}",
                "source_type": row["source_type"],
                "doc_id": row["doc_id"],
                "similarity": round(float(scores[idx]), 3),
                "passage": row["text"]
            })
        return results


if __name__ == "__main__":
    rag = ClinicalRAGEngine()
    df = rag.build_unified_knowledge_base()
    print(f"Unified Knowledge Base built with {len(df)} chunks.")
    res = rag.retrieve("spiculated high-density mass with irregular margins BI-RADS 5", top_k=3)
    for r in res:
        print(f"\n[{r['evidence_id']}] ({r['source_type']}) sim={r['similarity']}: {r['passage'][:150]}...")
