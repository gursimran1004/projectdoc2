from __future__ import annotations

import csv
import io
from datetime import datetime

try:
    from fpdf import FPDF
except Exception:
    FPDF = None


def build_report_text(doc_name: str, query: str, answer: str, source_label: str, confidence: float) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "AI Document Analysis Report\n"
        "===========================\n\n"
        f"Generated At: {timestamp}\n"
        f"Document: {doc_name}\n"
        f"Question: {query}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Source: {source_label}\n"
        f"Confidence: {confidence:.2f}\n"
    )


def build_report_pdf_bytes(report_text: str) -> bytes | None:
    if FPDF is None:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in report_text.splitlines():
        pdf.multi_cell(0, 8, txt=line)
    output = pdf.output(dest="S")
    if isinstance(output, bytearray):
        return bytes(output)
    if isinstance(output, str):
        return output.encode("latin-1", errors="ignore")
    return bytes(output)


def build_comparison_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["document", "answer", "source", "confidence"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def build_comparison_pdf_bytes(question: str, rows: list[dict[str, str]]) -> bytes | None:
    if FPDF is None:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Multi-Document Comparison Report", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, txt=f"Question: {question}")
    pdf.ln(1)

    for idx, row in enumerate(rows, start=1):
        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 8, txt=f"{idx}. {row['document']}")
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 7, txt=f"Answer: {row['answer']}")
        pdf.multi_cell(0, 7, txt=f"Source: {row['source']}")
        pdf.multi_cell(0, 7, txt=f"Confidence: {row['confidence']}")
        pdf.ln(2)

    output = pdf.output(dest="S")
    if isinstance(output, bytearray):
        return bytes(output)
    if isinstance(output, str):
        return output.encode("latin-1", errors="ignore")
    return bytes(output)
