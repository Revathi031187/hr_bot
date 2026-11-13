"""Central configuration and small helpers shared by ingest.py and rag_pipeline.py.

Every setting can be overridden with an environment variable (prefix ``HR_``), which
keeps the code identical across laptops, CPU servers and GPU boxes.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

# Silence telemetry and tokenizer fork warnings before the heavy libraries are imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Behind a TLS-inspecting corporate proxy the Hugging Face download fails with
# CERTIFICATE_VERIFY_FAILED. Trusting the operating system's certificate store fixes it.
# Set HR_USE_SYSTEM_CERTS=0 to turn this off.
if os.getenv("HR_USE_SYSTEM_CERTS", "1") != "0":
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Keep downloaded models next to the project (e.g. on D:) instead of C:\Users\...\.cache.
# Override by setting HF_HOME yourself before starting. Must happen before any HF import.
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_cache"))
DATA_PATH = Path(os.getenv("HR_DATA_PATH", BASE_DIR / "mock_hr_data.json"))
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
MANIFEST_PATH = DATA_DIR / "ingest_manifest.json"
COLLECTION_NAME = "hr_policies"

COMPANY_NAME = os.getenv("HR_COMPANY_NAME", "Meridian Labs")
HR_FALLBACK_CONTACT = os.getenv("HR_FALLBACK_CONTACT", "hr-help@meridianlabs.example")

# Runtime selection: HR_DEVICE=auto|cuda|cpu, HR_QUANTIZE=auto|4bit|none,
# HR_CPU_DTYPE=float32|bfloat16 (used only when running on CPU).
DEVICE = os.getenv("HR_DEVICE", "auto").lower()
QUANTIZE = os.getenv("HR_QUANTIZE", "auto").lower()
CPU_DTYPE = os.getenv("HR_CPU_DTYPE", "float32").lower()


def _default_llm() -> str:
    """3B on a GPU; the 1.5B sibling on CPU, where the 3B model is painfully slow."""
    try:
        import torch

        has_cuda = DEVICE != "cpu" and torch.cuda.is_available()
    except Exception:
        has_cuda = False
    return "Qwen/Qwen2.5-3B-Instruct" if has_cuda else "Qwen/Qwen2.5-1.5B-Instruct"


# Models. Alternatives: sentence-transformers/all-MiniLM-L6-v2, meta-llama/Llama-3.2-3B-Instruct.
# Set HR_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct to force the 3B model on CPU as well.
EMBEDDING_MODEL = os.getenv("HR_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.getenv("HR_LLM_MODEL") or _default_llm()

# CPU backend: a 4-bit GGUF file run by llama.cpp needs ~1 GB of RAM (vs ~6 GB for the float32 PyTorch model)
# and is much faster on CPU. HR_LLM_BACKEND=auto|gguf|transformers (auto: GGUF on CPU when llama-cpp-python is installed).
GGUF_REPO = os.getenv("HR_GGUF_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
GGUF_FILE = os.getenv("HR_GGUF_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
GGUF_CONTEXT = int(os.getenv("HR_GGUF_CONTEXT", "4096"))

UPLOAD_DIR = DATA_DIR / "uploads"
DEFAULT_UPLOAD_CATEGORY = "Uploaded Documents"

# Chunking is measured in embedding-model tokens.
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# Hybrid retrieval.
DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
RETRIEVER_K = 6
DEFAULT_TOP_K = 2
# If even the best retrieved passage is below this cosine similarity, the question is out of scope and
# the pipeline refuses without calling the LLM. (Measured: policy questions 0.76-0.82, off-topic 0.40-0.52.)
MIN_DENSE_SCORE = float(os.getenv("HR_MIN_DENSE_SCORE", "0.60"))

# Generation.
MAX_NEW_TOKENS = int(os.getenv("HR_MAX_NEW_TOKENS", "384"))

# Cross-encoder reranker: re-scores the best retrieval candidates by reading question and passage together.
# Set HR_RERANKER_MODEL=none to disable.
RERANKER_MODEL = os.getenv("HR_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLED = RERANKER_MODEL.strip().lower() not in {"", "none", "off", "0"}
RERANK_CANDIDATES = int(os.getenv("HR_RERANK_CANDIDATES", "8"))

# OCR quality: below this mean confidence an upload is flagged so people double-check names and numbers.
MIN_OCR_CONFIDENCE = float(os.getenv("HR_MIN_OCR_CONFIDENCE", "0.85"))

# Audit log (JSON lines). Text is redacted for emails, phone numbers and ID numbers before it is written.
AUDIT_ENABLED = os.getenv("HR_AUDIT_LOG", "1") != "0"
AUDIT_PATH = DATA_DIR / "audit_log.jsonl"
AUDIT_MAX_BYTES = int(os.getenv("HR_AUDIT_MAX_BYTES", str(5 * 1024 * 1024)))

logger = logging.getLogger("hr_chatbot")


def setup_logging() -> None:
    """Send hr_chatbot.* log records (including tracebacks) to the terminal running Streamlit."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def resolve_device(preference: str | None = None) -> str:
    """Return ``"cuda"`` or ``"cpu"`` according to HR_DEVICE and what is available."""
    import torch

    pref = (preference or DEVICE).lower()
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        logger.warning("HR_DEVICE=cuda requested but CUDA is unavailable; using CPU.")
    return "cpu"


def resolve_backend() -> str:
    """``"gguf"`` (llama.cpp) or ``"transformers"`` (PyTorch), per HR_LLM_BACKEND and the hardware."""
    import importlib.util

    preference = os.getenv("HR_LLM_BACKEND", "auto").lower()
    if preference in ("gguf", "transformers"):
        return preference
    if resolve_device() == "cuda":
        return "transformers"
    return "gguf" if importlib.util.find_spec("llama_cpp") else "transformers"


def build_embeddings():
    """Create the (normalised) HuggingFace embedding model on the best device."""
    from langchain_huggingface import HuggingFaceEmbeddings

    from download_models import ensure_model

    return HuggingFaceEmbeddings(
        model_name=ensure_model(EMBEDDING_MODEL),
        model_kwargs={"device": resolve_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_chroma_client():
    """Persistent Chroma client; one settings object so every caller shares the instance."""
    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def source_fingerprint(path: Path) -> str:
    """SHA-256 over a file, or over every file in a directory (names + bytes)."""
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        digest.update(str(file.relative_to(path if path.is_dir() else path.parent)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()
