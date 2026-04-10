"""Generate demo files under data/ when missing: invoice PNGs and a minimal sample PDF."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path


def _minimal_pdf_bytes() -> bytes:
    """Single blank-page PDF (valid structure, no external deps)."""
    buf = BytesIO()
    obj_offsets: dict[int, int] = {}

    def w(line: str) -> None:
        buf.write(line.encode("ascii"))

    w("%PDF-1.4\n")

    def obj(n: int, body: str) -> None:
        obj_offsets[n] = buf.tell()
        w(f"{n} 0 obj\n{body}\nendobj\n")

    obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
    obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    obj(3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")

    xref_pos = buf.tell()
    w("xref\n0 4\n")
    w("0000000000 65535 f \n")
    for n in range(1, 4):
        w(f"{obj_offsets[n]:010d} 00000 n \n")
    w("trailer << /Size 4 /Root 1 0 R >>\n")
    w(f"startxref\n{xref_pos}\n")
    w("%%EOF\n")
    return buf.getvalue()


def ensure_sample_pdf(doc_path: Path | None = None) -> Path:
    doc_path = doc_path or Path(__file__).parent / "documents" / "report_sample.pdf"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    if not doc_path.exists():
        doc_path.write_bytes(_minimal_pdf_bytes())
        print(f"Created {doc_path}")
    return doc_path


def ensure_invoices() -> Path:
    inv_dir = Path(__file__).parent / "invoices"
    if any(inv_dir.glob("*.png")):
        return inv_dir
    import runpy

    runpy.run_path(str(Path(__file__).parent / "generate_invoices.py"), run_name="__main__")
    return inv_dir


def main() -> None:
    ensure_sample_pdf()
    ensure_invoices()
    print("Demo data ready.")


if __name__ == "__main__":
    main()
