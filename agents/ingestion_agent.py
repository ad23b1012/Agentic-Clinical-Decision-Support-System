#Step 1: File Type Detection
# Step 2: Text Extraction
# Step 3: OCR Cleanup
# Step 4: Metadata Extraction


"""
Ingestion Agent
----------------
Responsibility:
- Convert raw patient inputs (PDFs, images, text files) into
  clean, normalized, timestamped clinical text documents.

Guarantees:
- No medical reasoning
- No hallucination
- No LLM usage
- Deterministic output

This module is the ONLY entry point for raw data.
"""

from typing import List, Dict, Optional
from pathlib import Path
import re
import pdfplumber
import pytesseract
from PIL import Image
from datetime import datetime


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff"}
SUPPORTED_TEXT_EXTENSIONS = {".txt"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}


# -----------------------------
# Public API (used by orchestrator)
# -----------------------------

def ingest_files(file_paths: List[str]) -> List[Dict]:
    """
    Main ingestion entry point.

    Input:
        List of file paths (PDF, image, text)

    Output:
        List of normalized clinical documents:
        [
          {
            "text": "...",
            "doc_type": "clinical_note",
            "date": "YYYY-MM-DD" | None,
            "source": "filename.ext"
          }
        ]
    """

    documents: List[Dict] = []

    for path in file_paths:
        file_path = Path(path)

        if not file_path.exists():
            continue

        ext = file_path.suffix.lower()

        raw_text: Optional[str] = None

        try:
            if ext in SUPPORTED_PDF_EXTENSIONS:
                raw_text = _extract_pdf_text(file_path)

            elif ext in SUPPORTED_IMAGE_EXTENSIONS:
                raw_text = _extract_image_text(file_path)

            elif ext in SUPPORTED_TEXT_EXTENSIONS:
                raw_text = file_path.read_text(encoding="utf-8", errors="ignore")

            else:
                continue

            if not raw_text or len(raw_text.strip()) < 20:
                continue

            cleaned_text = _clean_text(raw_text)

            documents.append({
                "text": cleaned_text,
                "doc_type": _infer_doc_type(cleaned_text),
                "date": _extract_date(cleaned_text),
                "source": file_path.name
            })

        except Exception:
            # Fail-safe: ingestion should NEVER crash pipeline
            continue

    return documents


# -----------------------------
# Extraction helpers
# -----------------------------

def _extract_pdf_text(path: Path) -> str:
    """
    Extract text from PDFs.
    Handles both text-based and scanned PDFs.
    """

    text_chunks = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)

    text = "\n".join(text_chunks)

    # Fallback to OCR if PDF text extraction fails
    if len(text.strip()) < 50:
        text = _ocr_pdf(path)

    return text


def _ocr_pdf(path: Path) -> str:
    """
    OCR fallback for scanned PDFs.
    """
    text_chunks = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            image = page.to_image(resolution=300).original
            text_chunks.append(pytesseract.image_to_string(image))

    return "\n".join(text_chunks)


def _extract_image_text(path: Path) -> str:
    """
    OCR for images (scanned notes, reports).
    """
    image = Image.open(path)
    return pytesseract.image_to_string(image)


# -----------------------------
# Cleaning & normalization
# -----------------------------

def _clean_text(text: str) -> str:
    """
    Cleans OCR noise and normalizes formatting
    while preserving clinical meaning.
    """

    # Normalize line breaks
    text = text.replace("\r", "\n")

    # Remove page numbers / headers / footers (heuristic)
    text = re.sub(r"\n\s*Page\s+\d+\s*\n", "\n", text, flags=re.IGNORECASE)

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix broken OCR words (simple heuristic)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text.strip()


# -----------------------------
# Metadata inference
# -----------------------------

def _infer_doc_type(text: str) -> str:
    """
    Infer document type using keyword heuristics.
    """

    lowered = text.lower()

    if "discharge summary" in lowered:
        return "discharge_summary"
    if "radiology" in lowered or "ct scan" in lowered or "x-ray" in lowered:
        return "radiology_report"
    if "lab results" in lowered or "hemoglobin" in lowered:
        return "lab_report"
    if "history of present illness" in lowered or "chief complaint" in lowered:
        return "clinical_note"

    return "unknown"


def _extract_date(text: str) -> Optional[str]:
    """
    Extracts date from clinical text.
    Returns ISO format YYYY-MM-DD if found.
    """

    date_patterns = [
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{2}-\d{2}-\d{4})\b",
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            raw_date = match.group(1)
            try:
                parsed = datetime.strptime(raw_date, "%d/%m/%Y")
                return parsed.strftime("%Y-%m-%d")
            except Exception:
                try:
                    parsed = datetime.strptime(raw_date, "%Y-%m-%d")
                    return parsed.strftime("%Y-%m-%d")
                except Exception:
                    pass

    return None


def main():
    """
    Manual test entry point for ingestion agent.
    Usage:
        python ingestion_agent.py /path/to/file1 /path/to/file2
    """

    import sys
    from pprint import pprint

    if len(sys.argv) < 2:
        print("Usage: python ingestion_agent.py <file1> <file2> ...")
        return

    file_paths = sys.argv[1:]

    print("\n[INFO] Starting ingestion...\n")
    documents = ingest_files(file_paths)

    print(f"[INFO] Ingested {len(documents)} document(s)\n")

    for idx, doc in enumerate(documents, start=1):
        print("=" * 60)
        print(f"Document {idx}")
        print(f"Source   : {doc['source']}")
        print(f"Type     : {doc['doc_type']}")
        print(f"Date     : {doc['date']}")
        print("-" * 60)
        print(doc["text"][:1000])  # preview first 1000 chars
        print("\n")


if __name__ == "__main__":
    main()
