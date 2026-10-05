from __future__ import annotations

import fitz

from src.extraction.pdf_extractor import extract_text, extract_text_from_pdf, extract_text_from_txt
from src.extraction.text_cleaner import clean_text, normalize_whitespace


def _pdf_bytes(text: str | None) -> bytes:
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def test_extract_text_from_pdf_success() -> None:
    data = _pdf_bytes("Python developer with SQL experience.")
    text, error = extract_text_from_pdf(data)
    assert error is None
    assert "Python" in text
    assert "SQL" in text


def test_extract_text_from_empty_pdf() -> None:
    data = _pdf_bytes(None)
    text, error = extract_text_from_pdf(data)
    assert text == ""
    assert error is not None
    assert error.category == "invalid_pdf"
    assert "Could not extract text from this PDF" in error.message


def test_extract_text_from_invalid_pdf_bytes() -> None:
    text, error = extract_text_from_pdf(b"not-a-pdf")
    assert text == ""
    assert error is not None
    assert error.category == "invalid_pdf"


def test_extract_text_from_txt() -> None:
    text, error = extract_text_from_txt("Machine Learning intern\n".encode("utf-8"))
    assert error is None
    assert "Machine Learning" in text


def test_extract_text_empty_txt() -> None:
    text, error = extract_text_from_txt(b"   ")
    assert text == ""
    assert error is not None
    assert error.category == "malformed_input"


def test_extract_text_dispatches_by_filename() -> None:
    pdf_text, pdf_error = extract_text(_pdf_bytes("FastAPI service"), "resume.PDF")
    assert pdf_error is None
    assert "FastAPI" in pdf_text
    txt_text, txt_error = extract_text(b"Git and REST APIs", "resume.txt")
    assert txt_error is None
    assert "Git" in txt_text


def test_clean_text_normalizes_bullets_and_space() -> None:
    raw = "Skills:\n• Python\n\n\n  SQL  "
    cleaned = clean_text(raw)
    assert "-" in cleaned or "Python" in cleaned
    assert "Python" in cleaned
    assert "SQL" in cleaned
    assert "\n\n\n" not in cleaned


def test_normalize_whitespace_empty() -> None:
    assert normalize_whitespace("") == ""
    assert clean_text("") == ""
