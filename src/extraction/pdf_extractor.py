"""Extract text from PDF (PyMuPDF) and TXT uploads. No OCR."""

from __future__ import annotations

from pathlib import Path

from src.models.schemas import AnalysisError

PDF_EMPTY_MESSAGE = (
    "Could not extract text from this PDF. Please use a text-based PDF or paste the resume text."
)


def extract_text_from_pdf(data: bytes) -> tuple[str, AnalysisError | None]:
    if not data:
        return "", AnalysisError(
            category="invalid_pdf",
            message=PDF_EMPTY_MESSAGE,
            details="Empty file bytes.",
        )
    try:
        import fitz
    except ImportError as exc:
        return "", AnalysisError(
            category="extraction_failed",
            message="PDF support is unavailable because PyMuPDF is not installed.",
            details=str(exc),
        )
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        return "", AnalysisError(
            category="invalid_pdf",
            message=PDF_EMPTY_MESSAGE,
            details=str(exc),
        )
    try:
        parts: list[str] = []
        for page in document:
            parts.append(page.get_text("text") or "")
        text = "\n".join(parts).strip()
        if not text:
            return "", AnalysisError(
                category="invalid_pdf",
                message=PDF_EMPTY_MESSAGE,
                details="PDF opened but contained no extractable text.",
            )
        return text, None
    finally:
        document.close()


def extract_text_from_txt(data: bytes) -> tuple[str, AnalysisError | None]:
    if not data:
        return "", AnalysisError(
            category="malformed_input",
            message="The uploaded text file is empty.",
        )
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = data.decode(encoding).strip()
            if not text:
                return "", AnalysisError(
                    category="malformed_input",
                    message="The uploaded text file is empty.",
                )
            return text, None
        except UnicodeDecodeError:
            continue
    return "", AnalysisError(
        category="malformed_input",
        message="Could not decode the text file. Please upload UTF-8 text.",
    )


def extract_text(data: bytes, filename: str) -> tuple[str, AnalysisError | None]:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(data)
    if suffix in {".txt", ".md"}:
        return extract_text_from_txt(data)
    sniff = data[:8]
    if sniff.startswith(b"%PDF"):
        return extract_text_from_pdf(data)
    return extract_text_from_txt(data)
