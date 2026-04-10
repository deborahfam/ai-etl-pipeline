"""Pydantic models for pipeline execution, data profiling, and AI outputs."""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pipeline execution models
# ---------------------------------------------------------------------------

class StepType(str, Enum):
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    AI_TRANSFORM = "ai_transform"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepResult(BaseModel):
    step_name: str
    step_type: StepType
    status: StepStatus
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    duration_seconds: float = 0.0
    rows_in: int = 0
    rows_out: int = 0
    columns_added: list[str] = Field(default_factory=list)
    columns_removed: list[str] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_display(self) -> str:
        if self.duration_seconds < 1:
            return f"{self.duration_seconds * 1000:.0f}ms"
        return f"{self.duration_seconds:.2f}s"


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineRun(BaseModel):
    pipeline_name: str
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    steps: list[StepResult] = Field(default_factory=list)
    total_rows_processed: int = 0
    total_llm_calls: int = 0
    total_llm_cost_usd: float = 0.0
    total_tokens_used: int = 0

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0

    @property
    def failed_steps(self) -> list[StepResult]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    @property
    def success(self) -> bool:
        return self.status == PipelineStatus.COMPLETED and not self.failed_steps


# ---------------------------------------------------------------------------
# Data profiling models
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: str = ""
    description: str = ""
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0
    min_value: str | None = None
    max_value: str | None = None
    mean_value: float | None = None
    std_value: float | None = None
    sample_values: list[str] = Field(default_factory=list)
    quality_score: int = Field(default=100, ge=0, le=100)
    issues: list[str] = Field(default_factory=list)


class DataProfile(BaseModel):
    dataset_name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    overall_quality_score: int = Field(default=100, ge=0, le=100)
    summary: str = ""
    relationships: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Anomaly detection models
# ---------------------------------------------------------------------------

class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AnomalyType(str, Enum):
    DATA_ERROR = "data_error"
    POTENTIAL_FRAUD = "potential_fraud"
    LEGITIMATE_OUTLIER = "legitimate_outlier"
    SYSTEM_ERROR = "system_error"
    INCONSISTENCY = "inconsistency"


class Anomaly(BaseModel):
    row_index: int
    column: str
    value: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    explanation: str = ""
    recommended_action: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AnomalyReport(BaseModel):
    dataset_name: str
    total_rows: int
    anomalies: list[Anomaly] = Field(default_factory=list)
    summary: str = ""
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0


# ---------------------------------------------------------------------------
# LLM enrichment models
# ---------------------------------------------------------------------------

class SentimentResult(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    score: float = Field(ge=-1.0, le=1.0)
    key_phrases: list[str] = Field(default_factory=list)
    language: str = ""


class EntityResult(BaseModel):
    text: str
    entity_type: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EnrichmentResult(BaseModel):
    sentiment: SentimentResult | None = None
    entities: list[EntityResult] = Field(default_factory=list)
    category: str = ""
    summary: str = ""
    language: str = ""


# ---------------------------------------------------------------------------
# Invoice / Document extraction models
# ---------------------------------------------------------------------------

class InvoiceLineItem(BaseModel):
    description: str
    quantity: float = 1.0
    unit_price: float = 0.0
    total: float = 0.0


class InvoiceData(BaseModel):
    vendor_name: str = ""
    vendor_address: str = ""
    invoice_number: str = ""
    invoice_date: str = ""
    due_date: str = ""
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    currency: str = "USD"
    notes: str = ""


# ---------------------------------------------------------------------------
# Quality scoring models
# ---------------------------------------------------------------------------

class QualityDimension(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)
    details: str = ""


class QualityReport(BaseModel):
    dataset_name: str
    overall_score: int = Field(ge=0, le=100)
    dimensions: list[QualityDimension] = Field(default_factory=list)
    summary: str = ""
    critical_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PII detection models
# ---------------------------------------------------------------------------

class PIIEntity(BaseModel):
    text: str
    pii_type: str
    location: str
    row_index: int
    column: str
    redacted_value: str = "***REDACTED***"


class PIIReport(BaseModel):
    total_pii_found: int = 0
    entities: list[PIIEntity] = Field(default_factory=list)
    columns_affected: list[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Pipeline advisor models
# ---------------------------------------------------------------------------

class TransformSuggestion(BaseModel):
    step_name: str
    description: str
    priority: str = "medium"
    reason: str = ""


class PipelineAdvice(BaseModel):
    dataset_summary: str = ""
    suggested_transforms: list[TransformSuggestion] = Field(default_factory=list)
    suggested_validations: list[str] = Field(default_factory=list)
    suggested_schema: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
