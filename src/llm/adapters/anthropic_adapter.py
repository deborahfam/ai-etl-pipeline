"""Anthropic (Claude) adapter with structured outputs and vision."""

from __future__ import annotations

import base64
import json
import os
from typing import Type

from pydantic import BaseModel

from src.llm.adapters.base import LLMAdapter, LLMResponse


class AnthropicAdapter(LLMAdapter):
    provider_name = "anthropic"
    supports_vision = True
    supports_structured_output = True

    # Cost per 1M tokens (input, output) by model prefix
    PRICING = {
        "claude-opus-4": (15.0, 75.0),
        "claude-sonnet-4": (3.0, 15.0),
        "claude-haiku-4": (0.80, 4.0),
        "claude-3-5-sonnet": (3.0, 15.0),
        "claude-3-5-haiku": (0.80, 4.0),
    }

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self, prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 4096
    ) -> LLMResponse:
        client = self._get_client()
        kwargs = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature}
        if system:
            kwargs["system"] = system
        kwargs["messages"] = [{"role": "user", "content": prompt}]

        resp = client.messages.create(**kwargs)
        return LLMResponse(
            content=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
            provider=self.provider_name,
        )

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[BaseModel, LLMResponse]:
        client = self._get_client()
        schema = response_model.model_json_schema()

        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Output ONLY the JSON, no other text."
        )

        kwargs = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature}
        if system:
            kwargs["system"] = system
        kwargs["messages"] = [{"role": "user", "content": full_prompt}]

        resp = client.messages.create(**kwargs)
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        parsed = response_model.model_validate(json.loads(text))
        llm_resp = LLMResponse(
            content=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
            provider=self.provider_name,
        )
        return parsed, llm_resp

    def complete_vision(
        self,
        prompt: str,
        images: list[bytes],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        client = self._get_client()
        content: list[dict] = []
        for img in images:
            b64 = base64.b64encode(img).decode()
            media_type = _detect_media_type(img)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            })
        content.append({"type": "text", "text": prompt})

        kwargs = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature}
        if system:
            kwargs["system"] = system
        kwargs["messages"] = [{"role": "user", "content": content}]

        resp = client.messages.create(**kwargs)
        return LLMResponse(
            content=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
            provider=self.provider_name,
        )

    def is_available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        for prefix, (inp_cost, out_cost) in self.PRICING.items():
            if self.model.startswith(prefix):
                return (input_tokens * inp_cost + output_tokens * out_cost) / 1_000_000
        return (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000


def _detect_media_type(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"
