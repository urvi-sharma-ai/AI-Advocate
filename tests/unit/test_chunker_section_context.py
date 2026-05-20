from __future__ import annotations

from ai_advocate.domain.models import LegalDocument
from ai_advocate.infrastructure.pdf.chunker import chunk_document


def test_chunker_prefix_repeated_for_long_section() -> None:
    doc = LegalDocument(
        id="doc1",
        act_title="Income Tax Act, 2025",
        domain="Finance",
        source_path="/tmp/it_2025.pdf",
        relative_file="Finance/Income_Tax_Act_2025.pdf",
        act_version="IT_2025",
        effective_date=None,
        supersedes=None,
    )

    long_body = ("(1) This is a clause.\n" * 500) + ("x" * 5000)
    page_text = (
        "Section 393 Deduction of tax at source\n"
        + long_body
        + "\nSection 394 Next section\nShort."
    )

    chunks = chunk_document(doc, pages=[(1, page_text)])

    s393 = [c for c in chunks if c.section_number == "393"]
    assert len(s393) >= 2  # split inside the mega-section

    for c in s393:
        assert c.text.startswith("[S393] Deduction of tax at source\n---\n")
        assert c.section_label == "Section 393"
