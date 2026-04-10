"""Shared test fixtures with mock LLM responses."""

from __future__ import annotations

from typing import Any, Type
from unittest.mock import MagicMock

import polars as pl
import pytest
from pydantic import BaseModel

from src.llm.adapters.base import LLMAdapter, LLMResponse
from src.llm.gateway import LLMGateway


class MockAdapter(LLMAdapter):
    """Mock LLM adapter that returns predefined responses."""

    provider_name = "mock"
    supports_vision = True
    supports_structured_output = True

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []
        self._default_response = '{"result": "mock response"}'
        self.model = "mock-model"

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 4096) -> LLMResponse:
        self.calls.append({"method": "complete", "prompt": prompt, "system": system})
        content = self._find_response(prompt)
        return LLMResponse(content=content, input_tokens=100, output_tokens=50, model="mock-model", provider="mock")

    def complete_structured(self, prompt: str, response_model: Type[BaseModel], system: str = "", temperature: float = 0.0, max_tokens: int = 4096) -> tuple[BaseModel, LLMResponse]:
        self.calls.append({"method": "complete_structured", "prompt": prompt})
        content = self._find_response(prompt)
        import json
        parsed = response_model.model_validate(json.loads(content))
        resp = LLMResponse(content=content, input_tokens=100, output_tokens=50, model="mock-model", provider="mock")
        return parsed, resp

    def complete_vision(self, prompt: str, images: list[bytes], system: str = "", temperature: float = 0.0, max_tokens: int = 4096) -> LLMResponse:
        self.calls.append({"method": "complete_vision", "prompt": prompt, "n_images": len(images)})
        content = self._find_response(prompt)
        return LLMResponse(content=content, input_tokens=200, output_tokens=100, model="mock-model", provider="mock")

    def is_available(self) -> bool:
        return True

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens + output_tokens) * 0.000001

    def _find_response(self, prompt: str) -> str:
        for key, value in self.responses.items():
            if key in prompt:
                return value
        return self._default_response


@pytest.fixture
def mock_adapter():
    return MockAdapter()


@pytest.fixture
def mock_gateway(mock_adapter):
    return LLMGateway(
        adapters={"mock": mock_adapter},
        primary_provider="mock",
        cache_enabled=False,
    )


@pytest.fixture
def sample_sales_df():
    return pl.DataFrame({
        "transaction_id": [f"T{i:03d}" for i in range(1, 21)],
        "date": ["2026-01-15"] * 20,
        "customer_id": [f"C{i:03d}" for i in range(1, 21)],
        "product_id": [f"P{(i % 5) + 1:03d}" for i in range(20)],
        "quantity": [1, 2, 3, 1, 5, 2, 1, 3, 2, 1, 99999, -5, 2, 3, 1, 2, 1, 3, 2, 1],
        "unit_price": [10.0, 25.0, 15.0, 50.0, 8.0, 30.0, 45.0, 12.0, 20.0, 35.0,
                       10.0, 25.0, -15.0, 50.0, 8.0, 30.0, 45.0, 12.0, 20.0, 35.0],
        "total": [10.0, 50.0, 45.0, 50.0, 40.0, 60.0, 45.0, 36.0, 40.0, 35.0,
                  10.0, 50.0, 30.0, 150.0, 8.0, 999.99, 45.0, 36.0, 40.0, 35.0],
        "payment_method": ["credit_card"] * 10 + ["debit", "cash", "credit_card", "paypal"] * 2 + ["credit_card"] * 2,
        "store_location": (["New York", "Los Angeles", "Chicago", "Miami", "Houston"] * 4),
        "salesperson": ["Alice", "Bob", None, "Diana", "Eve"] * 4,
        "notes": [""] * 15 + ["Urgent", "VIP customer", None, "Return", "Gift wrap"],
    })


@pytest.fixture
def sample_reviews_df():
    return pl.DataFrame({
        "review_id": [f"R{i:03d}" for i in range(1, 11)],
        "product_id": [f"P{(i % 3) + 1:03d}" for i in range(10)],
        "customer_name": ["John Smith", "Maria Garcia", "Hans Mueller", "Sophie Dubois",
                          "James Wilson", "Ana Lopez", "Pierre Martin", "Klaus Schmidt",
                          "Emily Brown", "Carlos Ruiz"],
        "email": ["john@example.com", "maria@test.com", "hans@example.de", "sophie@test.fr",
                  "james@", "ana@example.es", "pierre@test.fr", "klaus@example.de",
                  "emily@example.com", "carlos@test.es"],
        "rating": [5, 4, 3, 5, 1, 4, 2, 5, 3, 1],
        "review_text": [
            "Amazing product! Call me at 555-123-4567 for details.",
            "Buen producto, muy satisfecha con la compra.",
            "Durchschnittlich, nichts Besonderes.",
            "This is terrible, worst purchase ever!",  # Contradicts 5-star rating
            "Love it! Best thing I ever bought!",  # Contradicts 1-star rating
            "Excelente calidad y precio justo.",
            "Pas terrible, je ne recommande pas.",
            "Buy now! Best price! Click here! Amazing deal!!!",  # Spam
            "ok",
            "Horrible experiencia. Mi email es carlos@personal.com y necesito un reembolso.",
        ],
        "date": ["2026-01-15"] * 10,
        "language": ["en", "es", "de", "en", "en", "es", "fr", "en", "en", "es"],
    })
