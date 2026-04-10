"""LM Studio adapter — local models via OpenAI-compatible API."""

from __future__ import annotations

import json
import os
from typing import Type

from pydantic import BaseModel

from src.llm.adapters.base import LLMAdapter, LLMResponse


class LMStudioAdapter(LLMAdapter):
    provider_name = "lmstudio"
    supports_vision = False  # Depends on loaded model
    supports_structured_output = True

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("LM_STUDIO_MODEL", "local-model")
        self.base_url = base_url or os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="lm-studio")
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
            f"You MUST respond with valid JSON matching this exact schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Output ONLY the raw JSON object, no markdown, no explanation."
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
        raise NotImplementedError(
            "Vision is not supported by LM Studio adapter. "
            "Use a cloud provider (Anthropic or OpenAI) for vision tasks."
        )

    def is_available(self) -> bool:
        if not os.getenv("LM_STUDIO_URL"):
            return False
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/models", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0  # Local models are free
