"""Document Processor: PDF/image → structured data extraction pipeline.

Inspired by Unstract — defines extraction schemas via Pydantic models
and uses LLM Vision to extract structured data from documents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Type

import polars as pl
from pydantic import BaseModel

from src.engine.models import InvoiceData
from src.llm.gateway import LLMGateway
from src.utils.logging import get_logger

logger = get_logger(__name__)


def process_documents(
    document_paths: list[str | Path],
    response_model: Type[BaseModel] = InvoiceData,
    llm: LLMGateway | None = None,
    validate: bool = True,
    prompt: str | None = None,
) -> tuple[pl.DataFrame, list[dict]]:
    """Process a batch of document images into structured data.

    Pipeline:
    1. Read each document image
    2. Use LLM Vision to extract fields per schema
    3. Validate extracted data (cross-field checks)
    4. Return unified DataFrame + validation issues

    Args:
        document_paths: Paths to image/PDF files.
        response_model: Pydantic model defining extraction schema.
        llm: LLMGateway instance.
        validate: Run cross-field validation after extraction.
        prompt: Custom extraction prompt.
    """
    if llm is None:
        raise ValueError("LLM gateway required for document processing")

    schema = response_model.model_json_schema()
    default_prompt = (
        "You are an expert document data extractor.\n"
        "Extract ALL visible data from this document image into structured JSON.\n"
        f"Use this exact schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
        "Rules:\n"
        "- Be precise with all numbers, dates, and text\n"
        "- For line items in tables, extract every row\n"
        "- If a field is not visible, use empty string or 0\n"
        "- Preserve original formatting of invoice/reference numbers\n"
        "- Detect and convert currency symbols\n"
        "\nOutput ONLY valid JSON, no explanation."
    )

    records: list[dict] = []
    validation_issues: list[dict] = []

    for doc_path in document_paths:
        doc_path = Path(doc_path)
        if not doc_path.exists():
            logger.warning(f"Document not found: {doc_path}")
            continue

        logger.info(f"Processing document: {doc_path.name}")
        image_bytes = doc_path.read_bytes()

        try:
            parsed, resp = llm.complete_vision_structured(
                prompt=prompt or default_prompt,
                images=[image_bytes],
                response_model=response_model,
            )

            record = parsed.model_dump()
            record["_source_file"] = doc_path.name
            record["_extraction_tokens"] = resp.total_tokens

            # Validate if it's an invoice
            if validate and isinstance(parsed, InvoiceData):
                issues = _validate_invoice(parsed, doc_path.name)
                if issues:
                    record["_validation_issues"] = json.dumps(issues)
                    validation_issues.extend(issues)

            records.append(record)
            logger.info(f"Extracted from {doc_path.name}: {len(record)} fields")

        except Exception as e:
            logger.error(f"Extraction failed for {doc_path.name}: {e}")
            records.append({
                "_source_file": doc_path.name,
                "_error": str(e),
            })

    if not records:
        return pl.DataFrame(), validation_issues

    # Flatten nested structures for DataFrame
    flat_records = _flatten_records(records, response_model)
    df = pl.DataFrame(flat_records)

    logger.info(f"Document processing complete: {len(records)} documents → {len(df)} records")
    return df, validation_issues


def _validate_invoice(invoice: InvoiceData, source: str) -> list[dict]:
    """Cross-field validation for extracted invoice data."""
    issues = []

    # Check total matches sum of line items
    if invoice.line_items:
        computed_subtotal = sum(item.total for item in invoice.line_items)
        if invoice.subtotal > 0 and abs(computed_subtotal - invoice.subtotal) > 0.01:
            issues.append({
                "source": source,
                "type": "total_mismatch",
                "message": f"Line items sum ({computed_subtotal:.2f}) ≠ subtotal ({invoice.subtotal:.2f})",
                "severity": "warning",
            })

        # Check each line item: quantity × price ≈ total
        for i, item in enumerate(invoice.line_items):
            expected = item.quantity * item.unit_price
            if abs(expected - item.total) > 0.01:
                issues.append({
                    "source": source,
                    "type": "line_item_mismatch",
                    "message": f"Item {i+1}: qty({item.quantity}) × price({item.unit_price}) = {expected:.2f} ≠ {item.total:.2f}",
                    "severity": "warning",
                })

    # Check total = subtotal + tax
    if invoice.subtotal > 0 and invoice.tax >= 0:
        expected_total = invoice.subtotal + invoice.tax
        if abs(expected_total - invoice.total) > 0.01:
            issues.append({
                "source": source,
                "type": "total_calculation",
                "message": f"subtotal({invoice.subtotal:.2f}) + tax({invoice.tax:.2f}) = {expected_total:.2f} ≠ total({invoice.total:.2f})",
                "severity": "critical",
            })

    # Check for missing required fields
    if not invoice.vendor_name:
        issues.append({"source": source, "type": "missing_field", "message": "Missing vendor name", "severity": "warning"})
    if not invoice.invoice_number:
        issues.append({"source": source, "type": "missing_field", "message": "Missing invoice number", "severity": "warning"})
    if invoice.total == 0:
        issues.append({"source": source, "type": "zero_total", "message": "Invoice total is zero", "severity": "critical"})

    return issues


def _flatten_records(records: list[dict], model: Type[BaseModel]) -> list[dict]:
    """Flatten nested records for DataFrame compatibility."""
    flat = []
    for record in records:
        flat_record = {}
        for key, value in record.items():
            if isinstance(value, list) and key == "line_items":
                flat_record["line_items_count"] = len(value)
                flat_record["line_items_json"] = json.dumps(value, default=str)
            elif isinstance(value, (dict, list)):
                flat_record[key] = json.dumps(value, default=str)
            else:
                flat_record[key] = value
        flat.append(flat_record)
    return flat
