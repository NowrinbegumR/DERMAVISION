from __future__ import annotations

from embedding import KnowledgeBaseEmbedder
from retrieval import RAGRetriever


def main() -> None:
    embedder = KnowledgeBaseEmbedder()
    embedder.health_check("acne oily skin")

    retriever = RAGRetriever()
    results = retriever.retrieve("I have acne and oily skin", top_k=3, fetch_k=8)

    print("\n[Test Retrieval]")
    for r in results:
        print(r.doc.get("skin_condition"), r.semantic_score, r.rerank_score)

    print("\nIf the assistant streams, run rag_assistant.py and ask a query.")


if __name__ == "__main__":
    main()