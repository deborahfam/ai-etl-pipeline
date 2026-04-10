"""OpenRouter adapter — multi-model gateway via OpenAI-compatible API."""

from __future__ import annotations

import base64
import json
import os
from typing import Type

from pydantic import BaseModel

from src.llm.adapters.base import LLMAdapter, LLMResponse


class OpenRouterAdapter(LLMAdapter):
    provider_name = "openrouter"
    supports_vision = True
    supports_structured_output = True

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv(
            "OPENROUTER_MODEL", "anthropic/claude-sonnet-4-20250514"
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.BASE_URL,
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                default_headers={
                    "HTTP-Referer": "https://github.com/flowai-etl",
                    "X-Title": "FlowAI ETL",
                },
            )
        return self._client

    def complete(
        self, prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 4096
    ) -> LLMResponse:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        usage = resp.usage
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
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
        schema = response_model.model_json_schema()
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Output ONLY the JSON, no other text."
        )
        resp = self.complete(full_prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = response_model.model_validate(json.loads(text))
        return parsed, resp

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
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        resp = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        usage = resp.usage
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
            provider=self.provider_name,
        )

    def is_available(self) -> bool:
        return bool(os.getenv("OPENROUTER_API_KEY"))

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # OpenRouter pricing varies by model; approximate with Claude Sonnet rates
        return (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
