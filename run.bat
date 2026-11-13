@echo off
rem One-click launcher: venv -> dependencies -> ingestion (first run only) -> Streamlit.
rem Optional: set HR_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 before running (CUDA build of PyTorch).
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10+ was not found on PATH.
  exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
  echo [1/5] Creating virtual environment in .venv
  python -m venv .venv
  if errorlevel 1 exit /b 1
)
call ".venv\Scripts\activate.bat"

echo [2/5] Checking dependencies
python check_env.py
if errorlevel 1 (
  echo Installing requirements, this can take several minutes...
  python -m pip install --upgrade pip
  if defined HR_TORCH_INDEX_URL python -m pip install torch --index-url %HR_TORCH_INDEX_URL%
  python -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
  python check_env.py
  if errorlevel 1 exit /b 1
)

python -c "import llama_cpp" >nul 2>nul
if errorlevel 1 (
  echo Installing the optional low-memory CPU backend ^(llama.cpp^), skipped if unavailable...
  python -m pip install --upgrade pip --quiet
  python -m pip install -r requirements-gguf.txt --only-binary=:all: --quiet
)

echo [3/5] Downloading models, resumable and skipped when already cached
python download_models.py
if errorlevel 1 exit /b 1

echo [4/5] Checking vector database
if not exist "data\bm25_index.pkl" goto ingest
if not exist "data\chroma_db\chroma.sqlite3" goto ingest
echo Existing index found. Run "python ingest.py" to rebuild after changing the policy data.
goto serve

:ingest
echo No index found, running ingestion...
python ingest.py
if errorlevel 1 exit /b 1

:serve
echo [5/5] Starting Streamlit
streamlit run app.py
endlocal
