from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import CONFIG, build_document_text, load_json_file, setup_logging

logger = setup_logging(__name__)


class KnowledgeBaseEmbedder:
    def __init__(
        self,
        json_file: str = str(CONFIG.JSON_FILE),
        index_file: str = str(CONFIG.INDEX_FILE),
        meta_file: str = str(CONFIG.META_FILE),
        model_name: str = CONFIG.EMBEDDING_MODEL_NAME,
    ) -> None:
        self.json_file = Path(json_file)
        self.index_file = Path(index_file)
        self.meta_file = Path(meta_file)
        self.model_name = model_name
        self.model: SentenceTransformer | None = None

    def load_knowledge_base(self) -> List[Dict[str, Any]]:
        if not self.json_file.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {self.json_file}")

        logger.info("Loading knowledge base from %s", self.json_file)
        data = load_json_file(self.json_file)

        if not isinstance(data, list) or not data:
            raise ValueError("Knowledge base JSON must be a non-empty list.")

        logger.info("Loaded %d records", len(data))
        return data

    def build_documents(self, data: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        documents: List[str] = []
        metadata: List[Dict[str, Any]] = []

        for item in data:
            try:
                text = build_document_text(item)
                if not text.strip():
                    continue
                documents.append(CONFIG.PASSAGE_PREFIX + text)
                metadata.append(item)
            except Exception as exc:
                logger.warning("Skipping malformed record id=%s: %s", item.get("id", "unknown"), exc)

        logger.info("Built %d documents", len(documents))
        return documents, metadata

    def load_model(self) -> SentenceTransformer:
        if self.model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def generate_embeddings(self, documents: List[str]) -> np.ndarray:
        model = self.load_model()
        logger.info("Generating embeddings for %d documents", len(documents))
        embeddings = model.encode(
            documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=16,
        )
        return embeddings.astype(np.float32)

    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        logger.info("Built FAISS index with %d vectors", index.ntotal)
        return index

    def save(self, index: faiss.IndexFlatIP, metadata: Dict[str, Any]) -> None:
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.meta_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(self.index_file))
        with open(self.meta_file, "wb") as f:
            pickle.dump(metadata, f)

        logger.info("Saved index to %s", self.index_file)
        logger.info("Saved metadata to %s", self.meta_file)

    def run(self) -> None:
        data = self.load_knowledge_base()
        documents, docs = self.build_documents(data)

        if not documents:
            raise ValueError("No valid documents were built from the knowledge base.")

        embeddings = self.generate_embeddings(documents)
        index = self.build_index(embeddings)

        metadata = {
            "model_name": self.model_name,
            "dimension": int(embeddings.shape[1]),
            "count": len(docs),
            "documents": documents,
            "docs": docs,
        }
        self.save(index, metadata)

        logger.info("Embedding pipeline completed successfully.")
        self.health_check()

    def health_check(self, query: str = "acne oily skin and clogged pores") -> None:
        if not self.index_file.exists() or not self.meta_file.exists():
            raise FileNotFoundError("Index or metadata not found. Build the index first.")

        with open(self.meta_file, "rb") as f:
            meta = pickle.load(f)
        index = faiss.read_index(str(self.index_file))

        if index.ntotal != meta["count"]:
            raise RuntimeError(f"Vector count mismatch: index={index.ntotal}, metadata={meta['count']}")

        q = self.load_model().encode(
            [CONFIG.QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, ids = index.search(q, 5)
        logger.info("Health check top results:")
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
            if idx < 0:
                continue
            doc = meta["docs"][idx]
            logger.info("%d. %s | score=%.4f", rank, doc.get("skin_condition", ""), float(score))


def main() -> None:
    KnowledgeBaseEmbedder().run()


if __name__ == "__main__":
    main()