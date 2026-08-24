from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TARGET = ROOT_DIR / "RAG ASSISTANT" / "rag_assistant.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Unable to find launcher target: {TARGET}")

sys.path.insert(0, str(TARGET.parent))
runpy.run_path(str(TARGET), run_name="__main__")
