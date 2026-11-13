# HR Policy RAG Chatbot

A local, privacy-friendly HR assistant. It answers employee questions from your HR policy documents using
hybrid retrieval (dense vectors + BM25 keywords), a small open model running on your own machine and a
Streamlit portal that shows exactly which policy each answer came from.

- **Framework:** LangChain 0.3 (LCEL chains, `EnsembleRetriever`)
- **Embeddings:** `BAAI/bge-small-en-v1.5` (swap for `sentence-transformers/all-MiniLM-L6-v2`)
- **LLM:** `Qwen/Qwen2.5-3B-Instruct` on a GPU (4-bit), `Qwen2.5-1.5B-Instruct` (4-bit GGUF via llama.cpp) on CPU
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU, optional)
- **Vector store:** ChromaDB, persisted in `./data/chroma_db`
- **Keyword index:** BM25 pickled to `./data/bm25_index.pkl`
- **UI:** Streamlit with streamed answers, source cards, confidence and grounding tags, chat memory

> For how the system works internally (ingestion, retrieval maths, prompts, safety layers, design decisions) read
> **[docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md)**.

---

## 1. Windows setup (Python 3.10)

These steps target **Windows 10/11 (64-bit)** with **Python 3.10**. Python 3.11 and 3.12 also work, but 3.10 is the
tested baseline (`check_env.py` rejects anything older).

### 1.1 Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Windows | 10 or 11, 64-bit | |
| Python | **3.10.x, 64-bit** | 32-bit Python cannot install PyTorch |
| Free RAM | 8 GB | About 1-2 GB for the default CPU model, 6 GB+ if you force the PyTorch CPU backend |
| Free disk | 10 GB | Models are stored in `data\hf_cache` (about 1 GB on CPU, about 8 GB with the 3B model) |
| Internet | first run only | Downloads packages and models from PyPI and Hugging Face; fully offline afterwards |
| GPU (optional) | NVIDIA, 4 GB+ VRAM | Enables the 3B model in 4-bit. Without it the app runs on CPU |

### 1.2 Install Python 3.10

1. Download the **Windows installer (64-bit)** for Python 3.10 from <https://www.python.org/downloads/windows/>.
2. In the installer tick **Add python.exe to PATH**, then choose *Install Now*.
3. Open a **new** PowerShell window and confirm the version:

```powershell
py -3.10 --version        # Python 3.10.x
```

If you have several Python versions installed, always use `py -3.10 ...` below. The `py` launcher selects the
interpreter explicitly, so the virtual environment is guaranteed to be 3.10.

### 1.3 Get the project

Place the project folder somewhere with a **short path and no spaces**, for example `D:\hr-chatbot`. Deep paths can hit the
Windows 260-character limit while models and packages are unpacked.

```powershell
cd D:\hr-chatbot
```

### 1.4 Create and activate a virtual environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. Confirm that the environment uses 3.10:

```powershell
python --version          # must print Python 3.10.x
```

**If PowerShell says "running scripts is disabled on this system"**, allow scripts for the current window only
(nothing is changed permanently), then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

In classic Command Prompt (`cmd.exe`) use `.venv\Scripts\activate.bat` instead.

### 1.5 Install the dependencies

```powershell
python -m pip install --upgrade pip
```

**CPU only (no NVIDIA GPU)**: install everything from `requirements.txt`. To keep the download smaller you may
first install the CPU build of PyTorch:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu     # optional, smaller download
pip install -r requirements.txt
pip install -r requirements-gguf.txt --only-binary=:all:               # fast, low-memory CPU backend
```

**NVIDIA GPU**: install a CUDA build of PyTorch first. Pick the index that matches your driver (`nvidia-smi` shows
the highest CUDA version supported), then the remaining requirements:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements-gguf.txt` is optional. If no prebuilt `llama-cpp-python` wheel exists for your setup the install
fails harmlessly and the app falls back to the PyTorch backend (slower and uses more memory on CPU).

Verify the installation:

```powershell
python check_env.py
```

Expected output: `Python 3.10.x | no CUDA GPU detected, the model will run on CPU` (or the name of your GPU).

### 1.6 Build the index and download the models

```powershell
python download_models.py     # embedding model, reranker and LLM (resumable, retries automatically)
python ingest.py              # reads mock_hr_data.json, builds ChromaDB + BM25 indexes (about 1 minute)
```

Both steps are safe to repeat. `download_models.py` skips anything already in `data\hf_cache`.

### 1.7 Start the app

```powershell
streamlit run app.py
```

Open <http://localhost:8501> (Streamlit prints the URL; `headless = true` means the browser is not opened for you).
Try: *How many vacation days can I carry over?* The model finishes loading in the background, so the very first
answer can take a little longer.

Stop the server with `Ctrl+C`.

### 1.8 One-click alternative

`run.bat` performs steps 1.4 to 1.7 for you: it creates `.venv` if missing, installs missing packages, downloads
models, runs `ingest.py` on the first start and launches Streamlit.

```powershell
.\run.bat
```

`run.bat` creates the environment with whatever `python` is first on your `PATH`. To force 3.10, create the
environment yourself with `py -3.10 -m venv .venv` (step 1.4) **before** running `run.bat`; it reuses an existing `.venv`.
For a CUDA build of PyTorch set the index first: `$env:HR_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"; .\run.bat`.

### 1.9 Starting it again later

```powershell
cd D:\hr-chatbot
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

### 1.10 Windows troubleshooting

| Symptom | Fix |
|---|---|
| `py` is not recognised | Reinstall Python with the *py launcher* option ticked, or use `python` if it is 3.10 (`python --version`) |
| `python check_env.py` says "Python 3.10+ is required" | The virtual environment was created with an older Python. Delete `.venv` and redo step 1.4 with `py -3.10` |
| Activation blocked by execution policy | See step 1.4 (`Set-ExecutionPolicy -Scope Process ...`) |
| `pip` cannot find a matching `torch` | You are on 32-bit Python or an unsupported version. Check `python -c "import struct;print(struct.calcsize('P')*8)"` prints `64` |
| `OSError: [WinError 1455] The paging file is too small` | Not enough memory to load the model. Close other programs, free space on `C:` so Windows can grow its paging file, or set `HR_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct` |
| `bitsandbytes` errors on Windows or an old GPU | `$env:HR_QUANTIZE="none"` (model loads in float16 instead) |
| `CERTIFICATE_VERIFY_FAILED` while downloading | Corporate TLS proxy. The app already trusts the Windows certificate store (`truststore`); make sure you did not set `HR_USE_SYSTEM_CERTS=0` |
| Downloads keep dropping | Rerun `python download_models.py`; partial files resume where they stopped |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Long-path errors during `pip install` | Use a shorter project path, or enable long paths in Windows (`LongPathsEnabled` registry setting) |
| "ChromaDB collection not found" | Run `python ingest.py` |

### 1.11 Environment variables in PowerShell

PowerShell sets variables with `$env:NAME="value"` (they last for that window only). `cmd.exe` uses `set NAME=value`.

```powershell
$env:HR_DEVICE="cpu"
$env:HR_CPU_DTYPE="bfloat16"
streamlit run app.py
```

---

## 2. Quick start on macOS / Linux

```bash
./run.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ingest.py
streamlit run app.py
```

---

## 3. Architecture

```
                         ┌─────────────────────────── ingest.py (run once) ───────────────────────────┐
 mock_hr_data.json  ───► │ load + validate ─► section-aware chunker (400 / 50) ─┬─► embeddings ─► ChromaDB │
 (or PDF/DOCX folder)    │                                                      └─► BM25Retriever ─► pickle │
                         └─────────────────────────────────────────────────────────────────────────────┘

 Employee ─► Streamlit (app.py)
                │  question + chat history + category filter
                ▼
        rag_pipeline.py
         0. small talk and prompt-injection checks (no model call)
         1. condense follow-up into a standalone question   (LLM, only when there is history)
         2. EnsembleRetriever  ┬─ Chroma similarity search   weight 0.6
                               └─ BM25Retriever              weight 0.4
         3. score each hit (dense cosine, BM25, hybrid), cross-encoder rerank ─► confidence tag
         4. grounded prompt: strict HR system prompt + numbered <passage> context
         5. LLM (Qwen2.5) ─► streamed tokens ─► grounding check ─► policy reference ─► disclaimer if sensitive
                │
                ▼
        answer + source chunks + retrieval latency
```

Behaviour worth knowing:

- If no retrieved passage is relevant (best cosine similarity below `HR_MIN_DENSE_SCORE`, default `0.60`, unless backed by keyword or reranker evidence), the bot refuses without calling the LLM.
- Sensitive legal or medical wording in the question appends a fixed HR disclaimer in code, so it cannot be skipped by the model.
- The confidence tag is a heuristic on the hybrid score: `0.6 × cosine similarity + 0.4 × BM25/(BM25+4)`, bucketed as High (≥ 0.62), Medium (≥ 0.50) or Low.

## 4. Directory tree

```
hr-chatbot/
├── app.py                 Streamlit UI
├── rag_pipeline.py        Retrieval, prompts, LLM loading, streaming
├── ingest.py              Parse -> chunk -> embed -> Chroma + BM25 cache
├── config.py              Paths, model names, weights, env-var overrides
├── guard.py               Prompt-injection guard
├── grounding.py           Answer-grounding check
├── audit.py               Redacted JSON-lines audit log
├── download_models.py     Resumable, retrying model downloader (also used by the app)
├── check_env.py           Dependency check used by the run scripts
├── mock_hr_data.json      Sample policy corpus (5 categories, 28 policies)
├── requirements.txt
├── requirements-gguf.txt  Optional llama.cpp backend for low-memory CPU use
├── run.sh / run.bat       One-click start
├── docs/
│   └── TECHNICAL_DOCUMENT.md   Detailed technical explanation
├── .streamlit/config.toml Theme and server settings
└── data/                  Created by ingest.py
    ├── chroma_db/         Persistent vector store
    ├── bm25_index.pkl     Cached BM25 index
    ├── uploads/           Documents added through the UI (kept across re-ingests)
    ├── hf_cache/          Downloaded models
    ├── audit_log.jsonl    Audit trail
    └── ingest_manifest.json
```

## 5. Configuration

Everything is controlled with environment variables (see `config.py`); the defaults pick the best option automatically.

| Variable | Values | Default | Effect |
|---|---|---|---|
| `HR_DEVICE` | `auto`, `cuda`, `cpu` | `auto` | Device for embeddings and the LLM |
| `HR_QUANTIZE` | `auto`, `4bit`, `none` | `auto` | 4-bit NF4 (bitsandbytes) on CUDA; `none` forces float16 |
| `HR_CPU_DTYPE` | `float32`, `bfloat16` | `float32` | CPU precision for the PyTorch backend; `bfloat16` halves memory on modern CPUs |
| `HR_LLM_MODEL` | any HF causal LM | 3B Qwen on GPU, `Qwen/Qwen2.5-1.5B-Instruct` on CPU | Use `meta-llama/Llama-3.2-3B-Instruct` after accepting its licence and running `huggingface-cli login` |
| `HR_EMBEDDING_MODEL` | any sentence-transformers model | `BAAI/bge-small-en-v1.5` | Re-run `python ingest.py` after changing it |
| `HR_LLM_BACKEND` | `auto`, `gguf`, `transformers` | `auto` | `auto` = GGUF/llama.cpp on CPU (if `llama-cpp-python` is installed), PyTorch on CUDA |
| `HR_GGUF_REPO` / `HR_GGUF_FILE` | HF repo / file | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` / `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 4-bit model for the GGUF backend; use `Qwen/Qwen2.5-0.5B-Instruct-GGUF` / `qwen2.5-0.5b-instruct-q4_k_m.gguf` for ~3x faster answers |
| `HR_GGUF_CONTEXT` | integer | `4096` | llama.cpp context window |
| `HR_MAX_NEW_TOKENS` | integer | `384` | Maximum answer length |
| `HR_MIN_DENSE_SCORE` | 0-1 | `0.60` | Out-of-scope refusal threshold on the best passage's cosine similarity |
| `HR_RERANKER_MODEL` | HF model or `none` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `none` disables reranking |
| `HR_RERANK_CANDIDATES` | integer | `8` | Passages re-scored by the cross-encoder |
| `HR_MIN_OCR_CONFIDENCE` | 0-1 | `0.85` | Below this mean OCR confidence an upload gets a warning |
| `HR_AUDIT_LOG` / `HR_AUDIT_MAX_BYTES` | `0`/`1` / bytes | `1` / 5 MB | Audit log switch and rotation size |
| `HR_COMPANY_NAME` / `HR_FALLBACK_CONTACT` | text | `Meridian Labs` / `hr-help@meridianlabs.example` | Organisation name and HR mailbox used in prompts and citations |
| `HR_USE_SYSTEM_CERTS` | `0`/`1` | `1` | Trust the OS certificate store (corporate proxies) |
| `HF_HOME` | folder | `./data/hf_cache` | Where downloaded models are stored |
| `HR_DATA_PATH` | file or folder | `mock_hr_data.json` | Default source for `ingest.py` |

Loading order on CUDA: 4-bit NF4 first (about 2.5 GB of VRAM for the 3B model), then float16 if bitsandbytes is missing or fails.
Without CUDA the default is the 4-bit GGUF model (about 1 GB of RAM). If llama.cpp is not installed the PyTorch CPU backend is used
(about 6 GB of RAM in float32 for the 1.5B model) and answers are noticeably slower.

## 6. Uploading documents in the UI

Use **Knowledge base** in the sidebar: choose PDF, DOCX, TXT, Markdown or image (PNG, JPG, WEBP, BMP) files; text in images and scanned PDFs is read with OCR, pick a category (or type a new one) and
press *Add to knowledge base*. Each file is split into section-aware chunks of at most 400 tokens, embedded into ChromaDB and added to the BM25
cache immediately, so you can ask about it right away. Answers show an **Uploaded** tag on those sources.

- Uploading a file with the same name replaces the earlier version; *Remove* deletes it everywhere.
- Uploaded documents are also saved in `data\uploads\`, so `python ingest.py` keeps them when it rebuilds the index.
- The upload limit is 25 MB per file (`.streamlit\config.toml`).

## 7. Using your own policy documents

`ingest.py` accepts a JSON file or a folder containing PDF, DOCX, TXT, Markdown, image and JSON files:

```powershell
python ingest.py --data .\policies\          # folder of .pdf / .docx / .json
python ingest.py --data .\my_policies.json   # same schema as mock_hr_data.json
```

For **PDF and DOCX** files each file becomes one policy:

| Field | Taken from |
|---|---|
| `title` | the file name (`parental_leave.pdf` -> "Parental Leave") |
| `category` | the name of the sub-folder it sits in (`policies\benefits\dental.pdf` -> "Benefits"), or "General" |
| `effective_date` | the file's modification date |
| `contact_email` | `HR_FALLBACK_CONTACT` (default `hr-help@meridianlabs.example`) |
| `document_id` | derived from the relative path |

Organise the folder by category (one sub-folder per category) and name files with the exact policy title, because the
chatbot cites these values verbatim. If you need real effective dates or contacts per document, use the JSON format,
which requires `document_id`, `category`, `title`, `effective_date`, `content` and `contact_email`.

After ingesting, restart the app (or use "Rerun" in the Streamlit menu) so that it picks up the new indexes; the sidebar
shows when the source changed after the last ingestion. Also change `HR_COMPANY_NAME` and `HR_FALLBACK_CONTACT` so the
assistant uses your organisation's name and HR mailbox.

## 8. Answer quality and safety features

| Feature | What it does | Settings |
|---|---|---|
| Section-aware chunking | Documents are split at headings first, so a chunk covers one topic; long sections are split further and keep their heading (`section` metadata). Re-run `python ingest.py` to re-chunk existing data. | `CHUNK_SIZE_TOKENS`, `MIN_SECTION_TOKENS` in `config.py` / `ingest.py` |
| Cross-encoder reranker | The top retrieval candidates are re-scored with `cross-encoder/ms-marco-MiniLM-L-6-v2` (about 90 MB, CPU) and blended 70/30 with the hybrid score. Falls back to hybrid order if it cannot load. | `HR_RERANKER_MODEL` (`none` disables), `HR_RERANK_CANDIDATES` |
| Answer-grounding check | Compares the answer's numbers and wording with the passages and tags it *Grounded*, *Partly grounded*, *Check figures* or *Not grounded* (`grounding.py`). | thresholds in `grounding.py` |
| OCR quality warnings | Uploads read by OCR report mean confidence; low confidence, blank pages and empty images produce a warning in the sidebar and in the upload list. | `HR_MIN_OCR_CONFIDENCE` |
| Prompt-injection guard | Instruction-like text is removed from uploads and retrieved passages, passages are wrapped in `<passage>` tags as data, and override attempts in questions are refused (`guard.py`). | patterns in `guard.py` |
| Audit log | JSON lines in `data\audit_log.jsonl` for questions, sources, confidence, grounding, uploads and removals. Emails, phone numbers and IDs are masked; the file rotates at 5 MB. Viewable in the sidebar. | `HR_AUDIT_LOG=0` disables, `HR_AUDIT_MAX_BYTES` |

## 9. General troubleshooting

- **"ChromaDB collection not found"**: run `python ingest.py`.
- **CUDA out of memory**: use 4-bit (default), lower `HR_MAX_NEW_TOKENS`, or reduce "Chunks to retrieve" in the sidebar.
- **Slow first answer**: the LLM loads in the background after the app opens; later answers reuse it.
- **Gated model download fails**: Llama needs a Hugging Face token; Qwen does not.
- **Changed `HR_EMBEDDING_MODEL`**: re-run `python ingest.py`, because vectors from different models are not comparable.

## 10. Disclaimer

The bundled policy data is fictional sample content for demonstration. The chatbot summarises policy text and is not
a substitute for HR, legal or medical advice.
