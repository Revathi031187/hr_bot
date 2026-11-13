"""Resumable, retrying model downloader.

Unstable networks and proxies often cut long downloads with ``IncompleteRead``. Hugging Face resumes partial
files, so simply retrying until the snapshot is complete is enough. The app calls ``ensure_model`` itself,
and run.bat / run.sh run this script up front so the first question does not wait for a download.

Usage:
    python download_models.py              # embedding model + LLM
    python download_models.py --llm-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import config

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

log = logging.getLogger("download_models")

ATTEMPTS = 40
# Skip alternative weight formats we never load.
ALWAYS_IGNORE = ["*.msgpack", "*.h5", "*.ot", "*.tflite", "*.onnx", "onnx/*", "openvino/*", "original/*"]


def _local_snapshot(repo_id: str) -> Path | None:
    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        return None


def _looks_complete(snapshot: Path) -> bool:
    """A cached snapshot counts as complete when config and every weight shard are present."""
    if not (snapshot / "config.json").exists():
        return False
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except (json.JSONDecodeError, KeyError):
            return False
        return all((snapshot / shard).exists() for shard in shards)
    return any(snapshot.glob("*.safetensors")) or any(snapshot.glob("pytorch_model*.bin"))


def _with_retries(label: str, action, attempts: int = ATTEMPTS):
    """Run ``action`` until it succeeds; Hugging Face resumes partial files between attempts."""
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            log.info("Downloading %s (attempt %d/%d, partial files are resumed)", label, attempt, attempts)
            return action()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.warning("Download interrupted (%s). Retrying in %.0fs...", type(exc).__name__, delay)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(
        f"Could not download {label} after {attempts} attempts. Check the network or proxy, then rerun. "
        "Partial files are kept, so the next run resumes where this one stopped."
    )


def ensure_model(repo_id: str, attempts: int = ATTEMPTS) -> str:
    """Return a local path for ``repo_id``, downloading (with resume and retries) if needed."""
    if Path(repo_id).is_dir():
        return repo_id

    cached = _local_snapshot(repo_id)
    if cached and _looks_complete(cached):
        return str(cached)

    def download() -> str:
        ignore = list(ALWAYS_IGNORE)
        if any(name.endswith(".safetensors") for name in HfApi().list_repo_files(repo_id)):
            ignore += ["pytorch_model*.bin", "*.pt"]
        return snapshot_download(repo_id, ignore_patterns=ignore, max_workers=2)

    return _with_retries(repo_id, download, attempts)


def ensure_gguf(repo_id: str, filename: str, attempts: int = ATTEMPTS) -> str:
    """Return the local path of one GGUF file (a file only appears in the cache once fully downloaded)."""
    if Path(filename).is_file():
        return filename
    try:
        return hf_hub_download(repo_id, filename, local_files_only=True)
    except Exception:
        return _with_retries(f"{repo_id}/{filename}", lambda: hf_hub_download(repo_id, filename), attempts)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Pre-download the models used by the HR chatbot.")
    parser.add_argument("--llm-only", action="store_true", help="skip the embedding model")
    args = parser.parse_args(argv)

    if not args.llm_only:
        log.info("Ready: %s -> %s", config.EMBEDDING_MODEL, ensure_model(config.EMBEDDING_MODEL))
        if config.RERANKER_ENABLED:
            log.info("Ready: %s -> %s", config.RERANKER_MODEL, ensure_model(config.RERANKER_MODEL))
    if config.resolve_backend() == "gguf":
        log.info("Ready: %s -> %s", config.GGUF_FILE, ensure_gguf(config.GGUF_REPO, config.GGUF_FILE))
    else:
        log.info("Ready: %s -> %s", config.LLM_MODEL, ensure_model(config.LLM_MODEL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
