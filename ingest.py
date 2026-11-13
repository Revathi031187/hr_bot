"""Ingest HR policy documents into ChromaDB and cache a BM25 index on disk.

Usage:
    python ingest.py                      # ingests mock_hr_data.json
    python ingest.py --data path/to/file  # a .json file, or a folder of .pdf/.docx/.json files

The script rebuilds both indexes from scratch so they can never drift apart.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import config
import guard
from download_models import ensure_model

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

log = logging.getLogger("ingest")

REQUIRED_FIELDS = ("document_id", "category", "title", "effective_date", "content", "contact_email")


# --------------------------------------------------------------------------- loading
def _load_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        # Upload records are a single policy object; the corpus file wraps a list in {"documents": [...]}.
        records = payload["documents"] if "documents" in payload else [payload]
    else:
        records = payload
    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if not str(record.get(field, "")).strip()]
        if missing:
            raise ValueError(f"{path.name}: document #{index} is missing fields {missing}")
    return records


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").upper()


def _file_record(path: Path, text: str, root: Path) -> dict:
    """Wrap raw text from a PDF/DOCX file in the same schema as the JSON corpus."""
    relative = path.relative_to(root) if root.is_dir() else Path(path.name)
    category = relative.parent.name.replace("_", " ").replace("-", " ").title() if relative.parent.name else "General"
    return {
        "document_id": _slug(relative.with_suffix("").as_posix()),
        "category": category or "General",
        "title": path.stem.replace("_", " ").replace("-", " ").title(),
        "effective_date": datetime.fromtimestamp(path.stat().st_mtime).date().isoformat(),
        "content": text,
        "contact_email": config.HR_FALLBACK_CONTACT,
    }


_OCR_ENGINE = None
_OCR_STATS = threading.local()  # per-upload OCR quality, read by record_from_upload
MIN_IMAGE_SIDE = 80  # skip icons and bullets
MAX_IMAGE_SIDE = 2200  # keep OCR time and memory bounded
SPARSE_PAGE_CHARS = 80  # a PDF page with less text than this is OCR'd for embedded images


def _ocr_stats() -> dict:
    if not hasattr(_OCR_STATS, "data"):
        _reset_ocr_stats()
    return _OCR_STATS.data


def _reset_ocr_stats() -> None:
    _OCR_STATS.data = {"images": 0, "empty_images": 0, "lines": 0, "confidence_sum": 0.0, "low_lines": 0, "blank_pages": 0}


def _ocr_warnings() -> tuple[list[str], float | None]:
    """Turn the OCR statistics of one upload into user-facing warnings and a mean confidence."""
    stats = _ocr_stats()
    warnings: list[str] = []
    confidence = stats["confidence_sum"] / stats["lines"] if stats["lines"] else None
    if confidence is not None:
        low_share = stats["low_lines"] / stats["lines"]
        if confidence < config.MIN_OCR_CONFIDENCE or low_share > 0.25:
            warnings.append(
                f"Text recognition (OCR) confidence is low ({confidence:.0%}, {stats['low_lines']} of {stats['lines']} "
                "lines uncertain). Names and numbers may be wrong, so check them against the original."
            )
    if stats["empty_images"]:
        warnings.append(f"{stats['empty_images']} image(s) contained no readable text and were skipped.")
    if stats["blank_pages"]:
        warnings.append(f"{stats['blank_pages']} PDF page(s) have almost no text and no readable image, so they were skipped.")
    return warnings, confidence


def _ocr_image(data: bytes) -> str:
    """Extract text from image bytes with RapidOCR (ONNX, CPU, no system dependencies)."""
    global _OCR_ENGINE
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Image support needs Pillow and numpy (pip install -r requirements.txt).") from exc
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("Image support needs rapidocr-onnxruntime (pip install -r requirements.txt).") from exc

    image = Image.open(io.BytesIO(data))
    image = image.convert("RGB")
    if min(image.size) < MIN_IMAGE_SIDE:
        return ""
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    result, _ = _OCR_ENGINE(np.array(image))
    stats = _ocr_stats()
    stats["images"] += 1
    if not result:
        stats["empty_images"] += 1
        return ""
    scores = [float(score) for _box, _text, score in result]
    stats["lines"] += len(scores)
    stats["confidence_sum"] += sum(scores)
    stats["low_lines"] += sum(1 for score in scores if score < 0.7)
    # Group detected boxes into lines by vertical position, then read each line left to right.
    boxes = sorted(((box[0][1], box[0][0], text) for box, text, _score in result))
    lines, current, last_y = [], [], None
    for y, x, text in boxes:
        if last_y is not None and abs(y - last_y) > 14:
            lines.append(" ".join(t for _x, t in sorted(current)))
            current = []
        current.append((x, text))
        last_y = y
    if current:
        lines.append(" ".join(t for _x, t in sorted(current)))
    return "\n".join(lines)


def _safe_ocr(data: bytes) -> str:
    try:
        return _ocr_image(data)
    except RuntimeError:
        raise
    except Exception as exc:  # unreadable or unsupported embedded image: skip it
        log.warning("Skipped an image that could not be read (%s)", exc)
        return ""


def _read_image(source) -> str:
    """Read text from a photo or screenshot via OCR."""
    data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    return _ocr_image(data)


def _read_pdf(source) -> str:
    """``source`` is a path or a binary file object. Pages with little text (scans) are OCR'd."""
    from pypdf import PdfReader

    pages = []
    for page in PdfReader(source).pages:
        text = (page.extract_text() or "").strip()
        if len(text) < SPARSE_PAGE_CHARS:
            try:
                images = list(page.images)
            except Exception:
                images = []
            ocr = "\n".join(filter(None, (_safe_ocr(img.data) for img in images)))
            text = f"{text}\n{ocr}".strip()
            if not ocr and len(text) < SPARSE_PAGE_CHARS // 4:
                _ocr_stats()["blank_pages"] += 1
        pages.append(text)
    return "\n\n".join(pages)


def _read_docx(source) -> str:
    """``source`` is a path or a binary file object. Embedded pictures are OCR'd."""
    from docx import Document as DocxDocument

    document = DocxDocument(source if hasattr(source, "read") else str(source))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    for rel in document.part.rels.values():
        if "image" in rel.reltype:
            ocr = _safe_ocr(rel.target_part.blob)
            if ocr:
                parts.append(ocr)
    return "\n\n".join(parts)


def _read_text(source) -> str:
    if hasattr(source, "read"):
        return source.read().decode("utf-8", errors="replace")
    return Path(source).read_text(encoding="utf-8", errors="replace")


READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".txt": _read_text,
    ".md": _read_text,
    ".png": _read_image,
    ".jpg": _read_image,
    ".jpeg": _read_image,
    ".bmp": _read_image,
    ".webp": _read_image,
}
UPLOAD_TYPES = [suffix.lstrip(".") for suffix in READERS]


def record_from_upload(filename: str, data: bytes, category: str) -> dict:
    """Turn an uploaded file into a policy record (same schema as the JSON corpus)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in READERS:
        raise ValueError(f"Unsupported file type '{suffix}'. Use: {', '.join(UPLOAD_TYPES)}.")
    _reset_ocr_stats()
    text = READERS[suffix](io.BytesIO(data))
    warnings, ocr_confidence = _ocr_warnings()
    if not text.strip():
        raise ValueError("No readable text found in this file (blank or too low-resolution image?). " + " ".join(warnings))
    text, removed = guard.neutralize(text)
    if removed:
        log.warning("Removed %d instruction-like passage(s) from upload %s", len(removed), filename)
        warnings.append(
            f"{len(removed)} instruction-like passage(s) (for example '{removed[0][:60]}...') were removed "
            "from this document so they cannot steer the assistant."
        )
    stem = Path(filename).stem
    return {
        "document_id": f"UPL-{_slug(stem)[:40]}",
        "category": category.strip() or config.DEFAULT_UPLOAD_CATEGORY,
        "title": stem.replace("_", " ").replace("-", " ").strip().title(),
        "effective_date": datetime.now().date().isoformat(),
        "content": text,
        "contact_email": config.HR_FALLBACK_CONTACT,
        "source": "upload",
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warnings": warnings,
        "ocr_confidence": round(ocr_confidence, 3) if ocr_confidence is not None else None,
    }


def save_upload_record(record: dict) -> None:
    """Persist an uploaded record so a full re-ingest keeps it."""
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / f"{record['document_id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_upload_record(document_id: str) -> None:
    (config.UPLOAD_DIR / f"{document_id}.json").unlink(missing_ok=True)


def load_upload_records() -> list[dict]:
    """Records previously added through the UI."""
    if not config.UPLOAD_DIR.is_dir():
        return []
    records: list[dict] = []
    for file in sorted(config.UPLOAD_DIR.glob("*.json")):
        records.extend(_load_json(file))
    return records


def load_documents(source: Path) -> list[dict]:
    """Return policy records from a JSON file or a folder of JSON/PDF/DOCX/TXT/MD files."""
    if not source.exists():
        raise FileNotFoundError(f"Data source not found: {source}")
    if source.is_file() and source.suffix.lower() == ".json":
        return _load_json(source)

    readers = {suffix: reader for suffix, reader in READERS.items()}
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    records: list[dict] = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".json":
            records.extend(_load_json(file))
        elif suffix in readers:
            text = readers[suffix](file)
            if text.strip():
                records.append(_file_record(file, text, source))
            else:
                log.warning("Skipping %s: no extractable text (scanned PDF? run OCR first)", file.name)
    if not records:
        raise ValueError(f"No ingestible documents found in {source}")
    return records


# --------------------------------------------------------------------------- chunking
def build_splitter() -> RecursiveCharacterTextSplitter:
    """Token-based splitter that counts tokens with the embedding model's own tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(ensure_model(config.EMBEDDING_MODEL))
    tokenizer.model_max_length = 10**6  # we only count tokens; silence the 512-token warning
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)]|[IVX]+[.)])\s+\S")
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with", "&", "-"}
MIN_SECTION_TOKENS = 80  # a section at least this long gets a chunk of its own; shorter ones are merged


def _is_heading(line: str) -> bool:
    """Heuristic for a heading line in text extracted from PDF, DOCX, Markdown or plain text."""
    text = line.strip()
    if not text or len(text) > 90:
        return False
    if text.startswith("#"):
        return True
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    if len(words) > 10 or len(letters) < 3 or text[-1] in ".,;?!":
        return False
    if text.isupper() or (text.endswith(":") and len(words) <= 8):
        return True
    if _NUMBERED_HEADING.match(text):
        return True
    return len(words) <= 8 and all(
        w[0].isupper() or not w[0].isalpha() or w.lower() in _SMALL_WORDS for w in words
    )


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split text at headings into ``(heading, section_text)`` pairs; text without headings is one section."""
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []
    body_started = False  # becomes True once a non-heading line follows the current heading

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((heading, content))

    for line in text.splitlines():
        if _is_heading(line):
            clean = line.strip().lstrip("#").strip()
            if body_started or not body:
                flush()
                heading, body, body_started = clean, [line], False
            else:  # consecutive heading lines (for example a wrapped title) belong together
                heading = f"{heading} {clean}".strip()
                body.append(line)
        else:
            body.append(line)
            if line.strip():
                body_started = True
    flush()
    return sections


def split_text_by_section(text: str, splitter: RecursiveCharacterTextSplitter) -> list[tuple[str, str]]:
    """Section-aware chunking: ``(section_heading, chunk_text)`` pairs of at most the configured token size.

    Chunks do not straddle sections unless a section is tiny; an oversized section is split by the token
    splitter and every piece keeps the section heading.
    """
    size = splitter._length_function
    limit = config.CHUNK_SIZE_TOKENS
    chunks: list[tuple[str, str]] = []
    label, parts, used = "", [], 0

    def flush() -> None:
        nonlocal label, parts, used
        if parts:
            chunks.append((label, "\n".join(parts)))
        label, parts, used = "", [], 0

    for heading, body in split_sections(text):
        tokens = size(body)
        if tokens > limit:
            flush()
            chunks.extend((heading, piece.strip()) for piece in splitter.split_text(body) if piece.strip())
            continue
        if parts and (used + tokens > limit or used >= MIN_SECTION_TOKENS):
            flush()
        if not parts:
            label = heading
        parts.append(body)
        used += tokens
    flush()
    return chunks


def chunk_documents(records: list[dict], splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    """Split each policy into section-aware chunks that keep the full policy metadata."""
    chunks: list[Document] = []
    for record in records:
        pieces = split_text_by_section(record["content"], splitter)
        for index, (section, piece) in enumerate(pieces):
            # A short header lets both BM25 and the embedder see which policy and section a chunk belongs to.
            header = f"{record['title']} ({record['category']})"
            if section and not piece.lstrip("# ").startswith(section):
                header += f" - {section}"
            chunks.append(
                Document(
                    page_content=f"{header}\n{piece}",
                    metadata={
                        "chunk_id": f"{record['document_id']}::{index:03d}",
                        "document_id": record["document_id"],
                        "category": record["category"],
                        "title": record["title"],
                        "section": section[:120],
                        "effective_date": record["effective_date"],
                        "contact_email": record["contact_email"],
                        "chunk_index": index,
                        "total_chunks": len(pieces),
                        "source": record.get("source", "corpus"),
                        "uploaded_at": record.get("uploaded_at", ""),
                        "warnings": " | ".join(record.get("warnings", [])),
                    },
                )
            )
    return chunks


# --------------------------------------------------------------------------- persistence
def write_chroma(chunks: list[Document]) -> int:
    client = config.get_chroma_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
        log.info("Dropped existing collection '%s'", config.COLLECTION_NAME)
    except Exception:  # collection does not exist yet
        pass

    store = Chroma(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding_function=config.build_embeddings(),
    )
    store.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])
    return client.get_collection(config.COLLECTION_NAME).count()


def write_bm25(chunks: list[Document]) -> None:
    retriever = BM25Retriever.from_documents(chunks, k=config.RETRIEVER_K)
    config.BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.BM25_PATH.open("wb") as handle:
        pickle.dump(retriever, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_manifest(source: Path, documents: int, chunks: int) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
        "source_sha256": config.source_fingerprint(source),
        "documents": documents,
        "chunks": chunks,
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
    }
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- entry point
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the HR chatbot vector and keyword indexes.")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH, help="JSON file or folder of PDF/DOCX/JSON files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    source = args.data.resolve()

    started = time.perf_counter()
    records = load_documents(source)
    log.info("Loaded %d policy documents from %s", len(records), source)
    uploads = load_upload_records()
    if uploads:
        log.info("Keeping %d document(s) previously uploaded through the UI", len(uploads))
        replaced = {r["document_id"] for r in uploads}
        records = [r for r in records if r["document_id"] not in replaced] + uploads

    chunks = chunk_documents(records, build_splitter())
    log.info(
        "Created %d chunks (%d-token window, %d-token overlap)",
        len(chunks),
        config.CHUNK_SIZE_TOKENS,
        config.CHUNK_OVERLAP_TOKENS,
    )

    stored = write_chroma(chunks)
    log.info("Persisted %d vectors to %s", stored, config.CHROMA_DIR)

    write_bm25(chunks)
    log.info("Cached BM25 index at %s", config.BM25_PATH)

    write_manifest(source, len(records), len(chunks))
    log.info("Ingestion finished in %.1fs", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
