from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from embedding import KnowledgeBaseEmbedder
from utils import CONFIG, build_document_text, normalize_text, setup_logging

logger = setup_logging(__name__)


@dataclass
class RetrievalResult:
    doc: Dict[str, Any]
    semantic_score: float
    rerank_score: float
    final_score: float


class RAGRetriever:
    def __init__(self) -> None:
        self.embedder = KnowledgeBaseEmbedder()
        self.model = self.embedder.load_model()

        if not CONFIG.INDEX_FILE.exists() or not CONFIG.META_FILE.exists():
            raise FileNotFoundError("Vector store not found. Run embedding.py first.")

        self.index = faiss.read_index(str(CONFIG.INDEX_FILE))
        with open(CONFIG.META_FILE, "rb") as f:
            meta = pickle.load(f)

        self.meta = meta
        self.docs: List[Dict[str, Any]] = meta["docs"]

        if self.index.ntotal != len(self.docs):
            raise RuntimeError(
                f"Index/doc mismatch: vectors={self.index.ntotal}, metadata={len(self.docs)}"
            )

    def embed_query(self, query: str) -> np.ndarray:
        q = CONFIG.QUERY_PREFIX + normalize_text(query)
        vec = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        return vec

    def semantic_search(self, query: str, fetch_k: int = 8) -> List[Tuple[int, float]]:
        qvec = self.embed_query(query)
        scores, ids = self.index.search(qvec, fetch_k)

        results: List[Tuple[int, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def rerank_score(self, query: str, doc: Dict[str, Any], semantic_score: float) -> float:
        q = normalize_text(query)
        doc_text = normalize_text(build_document_text(doc))

        q_tokens = set(q.split())
        d_tokens = set(doc_text.split())

        overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
        condition_hit = 1.0 if normalize_text(doc.get("skin_condition", "")) in q else 0.0
        skin_type_hit = 1.0 if any(normalize_text(t) in q for t in doc.get("skin_types", [])) else 0.0
        symptom_hit = 1.0 if any(normalize_text(s) in q for s in doc.get("symptoms", [])) else 0.0

        lexical = (0.45 * overlap) + (0.25 * condition_hit) + (0.15 * skin_type_hit) + (0.15 * symptom_hit)
        return float((0.75 * semantic_score) + (0.25 * lexical))

    def retrieve(self, query: str, top_k: int = 4, fetch_k: int = 10) -> List[RetrievalResult]:
        candidates = self.semantic_search(query, fetch_k=fetch_k)
        scored: List[RetrievalResult] = []

        for idx, semantic_score in candidates:
            doc = self.docs[idx]
            rerank = self.rerank_score(query, doc, semantic_score)
            scored.append(
                RetrievalResult(
                    doc=doc,
                    semantic_score=semantic_score,
                    rerank_score=rerank,
                    final_score=rerank,
                )
            )

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored[:top_k]

    def build_context(self, results: List[RetrievalResult]) -> str:
        blocks: List[str] = []
        for r in results:
            blocks.append(
                "\n".join(
                    [
                        f"Condition: {r.doc.get('skin_condition', '')}",
                        f"Semantic score: {r.semantic_score:.4f}",
                        f"Rerank score: {r.rerank_score:.4f}",
                        f"Overview: {r.doc.get('overview', '')}",
                        f"Skin types: {', '.join(r.doc.get('skin_types', []))}",
                        f"Causes: {', '.join(r.doc.get('causes', []))}",
                        f"Symptoms: {', '.join(r.doc.get('symptoms', []))}",
                        f"Recommended ingredients: {', '.join(r.doc.get('recommended_ingredients', []))}",
                        f"Ingredients to avoid: {', '.join(r.doc.get('ingredients_to_avoid', []))}",
                        f"Recommended products: {', '.join(r.doc.get('recommended_products', []))}",
                        f"Safety guidance: {', '.join(r.doc.get('safety_guidance', []))}",
                        f"Expected results: {r.doc.get('expected_results', '')}",
                        f"Disclaimer: {r.doc.get('disclaimer', '')}",
                    ]
                )
            )
        return "\n\n---\n\n".join(blocks)