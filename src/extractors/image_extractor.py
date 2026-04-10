"""Image extractor — uses LLM Vision to extract structured data from images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Type

import polars as pl
from pydantic import BaseModel

from src.llm.gateway import LLMGateway
from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_from_images(
    image_paths: list[str | Path],
    response_model: Type[BaseModel],
    llm: LLMGateway,
    prompt: str | None = None,
    system: str = "You are an expert document data extractor. Extract all structured data accurately.",
) -> pl.DataFrame:
    """Extract structured data from a list of images using LLM Vision.

    Each image is processed independently. The LLM extracts data
    matching the provided Pydantic model schema.

    Args:
        image_paths: Paths to image files.
        response_model: Pydantic model defining the extraction schema.
        llm: LLMGateway instance for vision calls.
        prompt: Custom extraction prompt (auto-generated if None).
        system: System prompt for the LLM.
    """
    if not llm.has_vision:
        raise RuntimeError(
            "No vision-capable LLM provider available. "
            "Configure Anthropic or OpenAI API keys."
        )

    schema = response_model.model_json_schema()
    default_prompt = (
        "Extract ALL data from this document image into structured JSON.\n"
        f"Use this exact schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n"
        "Be precise with numbers, dates, and text. "
        "If a field is not visible, use an empty string or 0."
        "\nOutput ONLY valid JSON."
    )

    extraction_prompt = prompt or default_prompt
    records: list[dict] = []

    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            logger.warning(f"Image not found, skipping: {img_path}")
            continue

        logger.info(f"Extracting data from: {img_path.name}")
        image_bytes = img_path.read_bytes()

        try:
            parsed, resp = llm.complete_vision_structured(
                prompt=extraction_prompt,
                images=[image_bytes],
                response_model=response_model,
                system=system,
            )
            record = parsed.model_dump()
            record["_source_file"] = img_path.name
            records.append(record)
            logger.info(f"Successfully extracted from {img_path.name}")
        except Exception as e:
            logger.error(f"Failed to extract from {img_path.name}: {e}")
            records.append({"_source_file": img_path.name, "_error": str(e)})

    if not records:
        return pl.DataFrame()

    return pl.DataFrame(records)


def extract_single_image(
    image_path: str | Path,
    response_model: Type[BaseModel],
    llm: LLMGateway,
    prompt: str | None = None,
) -> BaseModel:
    """Extract structured data from a single image."""
    image_path = Path(image_path)
    image_bytes = image_path.read_bytes()
    schema = response_model.model_json_schema()

    default_prompt = (
        "Extract ALL data from this document into structured JSON.\n"
        f"Schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n"
        "Output ONLY valid JSON."
    )

    parsed, _ = llm.complete_vision_structured(
        prompt=prompt or default_prompt,
        images=[image_bytes],
        response_model=response_model,
    )
    return parsed
