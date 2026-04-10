"""Hybrid data validation: rule-based + LLM semantic validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

from src.llm.gateway import LLMGateway
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_checked: int = 0
    rows_failed: int = 0


# ---------------------------------------------------------------------------
# Rule-based validation
# ---------------------------------------------------------------------------

def validate_dataframe(
    df: pl.DataFrame,
    rules: dict[str, list[Callable[[pl.Series], pl.Series]]] | None = None,
    not_null: list[str] | None = None,
    unique: list[str] | None = None,
    value_ranges: dict[str, tuple[float, float]] | None = None,
    custom_checks: list[Callable[[pl.DataFrame], str | None]] | None = None,
) -> tuple[pl.DataFrame, ValidationResult]:
    """Validate a DataFrame with rule-based checks.

    Args:
        df: Input DataFrame.
        rules: Dict of column→list of validation functions (Series→bool Series).
        not_null: Columns that must not contain nulls.
        unique: Columns that must contain unique values.
        value_ranges: Dict of column→(min, max) for numeric range checks.
        custom_checks: List of functions that take a DataFrame and return error message or None.
    """
    result = ValidationResult(is_valid=True, rows_checked=len(df))
    invalid_mask = pl.lit(False)

    # Not-null checks
    for col in (not_null or []):
        if col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                result.errors.append(f"Column '{col}' has {null_count} null values")
                invalid_mask = invalid_mask | pl.col(col).is_null()

    # Uniqueness checks
    for col in (unique or []):
        if col in df.columns:
            dup_count = len(df) - df[col].n_unique()
            if dup_count > 0:
                result.errors.append(f"Column '{col}' has {dup_count} duplicate values")

    # Range checks
    for col, (min_val, max_val) in (value_ranges or {}).items():
        if col in df.columns:
            out_of_range = df.filter(
                (pl.col(col) < min_val) | (pl.col(col) > max_val)
            )
            if len(out_of_range) > 0:
                result.errors.append(
                    f"Column '{col}': {len(out_of_range)} values outside range [{min_val}, {max_val}]"
                )
                invalid_mask = invalid_mask | (pl.col(col) < min_val) | (pl.col(col) > max_val)

    # Custom column rules
    for col, rule_list in (rules or {}).items():
        if col in df.columns:
            for rule_fn in rule_list:
                try:
                    valid_mask = rule_fn(df[col])
                    failed = (~valid_mask).sum()
                    if failed > 0:
                        result.errors.append(
                            f"Column '{col}': {failed} rows failed rule {rule_fn.__name__}"
                        )
                        invalid_mask = invalid_mask | ~valid_mask
                except Exception as e:
                    result.warnings.append(f"Rule check failed on '{col}': {e}")

    # Custom DataFrame checks
    for check_fn in (custom_checks or []):
        error = check_fn(df)
        if error:
            result.errors.append(error)

    # Mark invalid rows
    df = df.with_columns(
        invalid_mask.alias("_validation_failed")
    )
    result.rows_failed = df.filter(pl.col("_validation_failed")).height
    result.is_valid = len(result.errors) == 0

    logger.info(
        f"Validation: {'PASSED' if result.is_valid else 'FAILED'} | "
        f"{result.rows_failed}/{result.rows_checked} rows failed | "
        f"{len(result.errors)} errors, {len(result.warnings)} warnings"
    )
    return df, result


# ---------------------------------------------------------------------------
# Semantic validation with LLM
# ---------------------------------------------------------------------------

def validate_semantic(
    df: pl.DataFrame,
    llm: LLMGateway,
    sample_size: int = 20,
    context: str = "",
) -> ValidationResult:
    """Use LLM to perform semantic validation on a data sample.

    The LLM checks for logical inconsistencies, implausible values,
    and cross-column constraint violations that rules cannot catch.
    """
    sample = df.head(sample_size).to_dicts()

    prompt = (
        "You are a data quality expert. Analyze this data sample for semantic issues:\n"
        "- Values that don't make logical sense (e.g., age=200, future dates for past events)\n"
        "- Cross-column inconsistencies (e.g., total ≠ quantity × price)\n"
        "- Implausible combinations (e.g., country=Japan but currency=EUR)\n"
        "- Format inconsistencies across rows\n"
        f"\nContext: {context}\n" if context else ""
        f"\nData sample (JSON):\n```json\n{json.dumps(sample, indent=2, default=str)}\n```\n\n"
        'Return JSON: {"errors": ["description of each issue"], "warnings": ["potential issues"]}'
    )

    try:
        resp = llm.complete(prompt, temperature=0.0)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result_data = json.loads(text)

        errors = result_data.get("errors", [])
        warnings = result_data.get("warnings", [])

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            rows_checked=len(sample),
        )
    except Exception as e:
        logger.error(f"Semantic validation failed: {e}")
        return ValidationResult(
            is_valid=True,
            warnings=[f"Semantic validation could not complete: {e}"],
            rows_checked=0,
        )
