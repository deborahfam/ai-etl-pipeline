"""OpenAI (GPT-4o) adapter with structured outputs and vision."""

from __future__ import annotations

import base64
import json
import os
from typing import Type

from pydantic import BaseModel

from src.llm.adapters.base import LLMAdapter, LLMResponse


class OpenAIAdapter(LLMAdapter):
    provider_name = "openai"
    supports_vision = True
    supports_structured_output = True

    PRICING = {
        "gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.0, 30.0),
    }

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
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
        client = self._get_client()
        schema = response_model.model_json_schema()

        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Output ONLY the JSON, no other text."
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": full_prompt})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        usage = resp.usage
        text = resp.choices[0].message.content or "{}"
        parsed = response_model.model_validate(json.loads(text))
        llm_resp = LLMResponse(
            content=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
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
        return bool(os.getenv("OPENAI_API_KEY"))

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        for prefix, (inp_cost, out_cost) in self.PRICING.items():
            if self.model.startswith(prefix):
                return (input_tokens * inp_cost + output_tokens * out_cost) / 1_000_000
        return (input_tokens * 2.5 + output_tokens * 10.0) / 1_000_000
