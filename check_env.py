"""Environment check used by run.sh / run.bat. Exit code 0 means every required package imports."""

from __future__ import annotations

import importlib.util
import sys

REQUIRED = [
    "torch",
    "transformers",
    "accelerate",
    "sentence_transformers",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_huggingface",
    "langchain_chroma",
    "langchain_text_splitters",
    "chromadb",
    "rank_bm25",
    "streamlit",
]
MIN_PYTHON = (3, 10)


def main() -> int:
    if sys.version_info < MIN_PYTHON:
        print(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, found {sys.version.split()[0]}.")
        return 2

    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing packages: " + ", ".join(missing))
        return 1

    import torch

    if torch.cuda.is_available():
        quant = "available" if importlib.util.find_spec("bitsandbytes") else "NOT installed (float16 fallback)"
        print(f"Python {sys.version.split()[0]} | CUDA GPU: {torch.cuda.get_device_name(0)} | 4-bit (bitsandbytes): {quant}")
    else:
        print(f"Python {sys.version.split()[0]} | no CUDA GPU detected, the model will run on CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
