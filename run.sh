#!/usr/bin/env bash
# One-click launcher: venv -> dependencies -> ingestion (first run only) -> Streamlit.
# Optional: HR_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./run.sh  (CUDA build of PyTorch)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.10+ was not found. Install it or set PYTHON_BIN." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[1/5] Creating virtual environment in .venv"
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/5] Checking dependencies"
if ! python check_env.py; then
  echo "Installing requirements (this can take several minutes)..."
  python -m pip install --upgrade pip
  if [ -n "${HR_TORCH_INDEX_URL:-}" ]; then
    python -m pip install torch --index-url "$HR_TORCH_INDEX_URL"
  fi
  python -m pip install -r requirements.txt
  python check_env.py
fi

if ! python -c "import llama_cpp" >/dev/null 2>&1; then
  echo "Installing the optional low-memory CPU backend (llama.cpp); skipped if unavailable..."
  python -m pip install -r requirements-gguf.txt --only-binary=:all: --quiet || echo "llama.cpp wheel not available; using the PyTorch backend."
fi

echo "[3/5] Downloading models (resumable, skipped when already cached)"
python download_models.py

echo "[4/5] Checking vector database"
if [ ! -f "data/bm25_index.pkl" ] || [ ! -d "data/chroma_db" ] || [ -z "$(ls -A data/chroma_db 2>/dev/null)" ]; then
  echo "No index found, running ingestion..."
  python ingest.py
else
  echo "Existing index found (run 'python ingest.py' to rebuild after changing the policy data)."
fi

echo "[5/5] Starting Streamlit"
exec streamlit run app.py
