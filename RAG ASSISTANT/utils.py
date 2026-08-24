from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    JSON_FILE: Path = BASE_DIR / "knowledge_base" / "skin_knowledge.json"
    INDEX_FILE: Path = BASE_DIR / "vector_store" / "faiss.index"
    META_FILE: Path = BASE_DIR / "vector_store" / "metadata.pkl"
    EMBEDDING_MODEL_NAME: str = "intfloat/e5-small-v2"
    CHAT_MODEL_NAME: str = "qwen3:4b"
    OLLAMA_URL: str = "http://localhost:11434"
    PASSAGE_PREFIX: str = "passage: "
    QUERY_PREFIX: str = "query: "


CONFIG = Config()


def setup_logging(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _join(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    if value is None:
        return ""
    return str(value)


def build_document_text(item: Dict[str, Any]) -> str:
    parts = [
        f"Condition: {item.get('skin_condition', '')}",
        f"Skin types: {_join(item.get('skin_types', []))}",
        f"Overview: {item.get('overview', '')}",
        f"Causes: {_join(item.get('causes', []))}",
        f"Symptoms: {_join(item.get('symptoms', []))}",
        f"Recommended ingredients: {_join(item.get('recommended_ingredients', []))}",
        f"Ingredients to avoid: {_join(item.get('ingredients_to_avoid', []))}",
        f"Recommended products: {_join(item.get('recommended_products', []))}",
        f"Safety guidance: {_join(item.get('safety_guidance', []))}",
        f"Expected results: {item.get('expected_results', '')}",
        f"Disclaimer: {item.get('disclaimer', '')}",
    ]

    routine = item.get("skincare_routine")
    if isinstance(routine, dict):
        morning = _join(routine.get("morning", []))
        night = _join(routine.get("night", []))
        if morning:
            parts.append(f"Morning routine: {morning}")
        if night:
            parts.append(f"Night routine: {night}")

    return " | ".join(p for p in parts if p and not p.endswith(": "))


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))