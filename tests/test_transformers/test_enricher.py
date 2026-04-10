"""Tests for LLM-powered enrichment."""

import json

import polars as pl
import pytest


class TestEnrichTextColumn:
    def test_enrichment_adds_columns(self, mock_gateway):
        from src.transformers.enricher import enrich_text_column

        # Mock LLM returns structured enrichment
        mock_gateway.adapters["mock"].responses["Analyze each text"] = json.dumps({
            "results": [
                {"sentiment": {"label": "positive", "score": 0.9}, "entities": [], "category": "electronics", "language": "en"},
                {"sentiment": {"label": "negative", "score": -0.5}, "entities": [], "category": "service", "language": "es"},
            ]
        })

        df = pl.DataFrame({
            "id": [1, 2],
            "text": ["Great product!", "Mal servicio"],
        })

        result = enrich_text_column(
            df, text_column="text", llm=mock_gateway,
            operations=["sentiment", "category", "language"],
            batch_size=10,
        )

        assert "sentiment_label" in result.columns
        assert "sentiment_score" in result.columns
        assert "category" in result.columns
        assert "detected_language" in result.columns

    def test_missing_column_raises(self, mock_gateway):
        from src.transformers.enricher import enrich_text_column

        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            enrich_text_column(df, text_column="nonexistent", llm=mock_gateway)
