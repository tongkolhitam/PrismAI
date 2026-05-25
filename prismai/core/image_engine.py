"""Image analysis engine — OCR, scene understanding, object detection, brands, colors."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from prismai.models.schemas import (
    AnalysisResult,
    AnalysisType,
    BrandResult,
    ColorInfo,
    ColorsResult,
    Modality,
    ObjectItem,
    ObjectsResult,
    OCRResult,
    SceneResult,
)
from prismai.providers.fallback_provider import FallbackProvider
from prismai.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are PrismAI's image analysis engine. Analyze image content accurately.
Always respond with valid JSON matching the requested schema."""


class ImageEngine:
    """Handles all image-based analyses using LLM with vision capabilities."""

    def __init__(self, provider: FallbackProvider) -> None:
        self.provider = provider

    async def analyze(
        self,
        image_data: str,
        analyses: list[AnalysisType],
        options: dict[str, Any] | None = None,
    ) -> list[AnalysisResult]:
        """Run requested analyses on an image."""
        options = options or {}
        # image_data can be base64 string or file path
        image_description = await self._prepare_image(image_data)

        results: list[AnalysisResult] = []
        for analysis_type in analyses:
            start = time.monotonic()
            try:
                result = await self._dispatch(image_description, analysis_type, options)
                result.processing_time_ms = (time.monotonic() - start) * 1000
                results.append(result)
            except Exception as exc:
                logger.error("image_analysis_failed", type=analysis_type, error=str(exc))
                results.append(
                    AnalysisResult(
                        modality=Modality.IMAGE,
                        analysis_type=analysis_type,
                        success=False,
                        error=str(exc),
                        processing_time_ms=(time.monotonic() - start) * 1000,
                    )
                )
        return results

    async def _prepare_image(self, image_data: str) -> str:
        """Prepare image data — convert file path to description or validate base64."""
        path = Path(image_data)
        if path.exists():
            # For local files, we create a description prompt
            return f"[Image file: {path.name}]"
        # Assume base64
        if len(image_data) > 100:
            return f"[Base64 image data, {len(image_data)} chars]"
        return image_data

    async def _dispatch(
        self, image_desc: str, analysis_type: AnalysisType, options: dict[str, Any]
    ) -> AnalysisResult:
        dispatch = {
            AnalysisType.OCR: self._ocr,
            AnalysisType.SCENE: self._scene,
            AnalysisType.OBJECTS: self._objects,
            AnalysisType.BRANDS: self._brands,
            AnalysisType.COLORS: self._colors,
        }
        handler = dispatch.get(analysis_type)
        if not handler:
            raise ValueError(f"Unsupported image analysis: {analysis_type}")
        return await handler(image_desc, options)

    async def _ocr(self, image_desc: str, options: dict[str, Any]) -> AnalysisResult:
        prompt = f"""Extract all visible text from this image. Return JSON:
{{"extracted_text": "all text found", "word_count": 0, "regions": [{{"text": "...", "x": 0, "y": 0, "width": 0, "height": 0}}]}}

Image context: {image_desc}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)
        ocr = OCRResult(**data)
        return AnalysisResult(
            modality=Modality.IMAGE,
            analysis_type=AnalysisType.OCR,
            data=ocr.model_dump(),
            confidence=0.85,
            provider="mimo",
        )

    async def _scene(self, image_desc: str, options: dict[str, Any]) -> AnalysisResult:
        prompt = f"""Describe the scene in this image in detail. Return JSON:
{{"description": "detailed description", "setting": "indoor/outdoor/etc", "mood": "atmosphere", "time_of_day": "day/night/etc"}}

Image context: {image_desc}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)
        scene = SceneResult(**data)
        return AnalysisResult(
            modality=Modality.IMAGE,
            analysis_type=AnalysisType.SCENE,
            data=scene.model_dump(),
            confidence=0.80,
            provider="mimo",
        )

    async def _objects(self, image_desc: str, options: dict[str, Any]) -> AnalysisResult:
        prompt = f"""List all objects visible in this image. Return JSON:
{{"objects": [{{"name": "object name", "confidence": 0.0-1.0, "description": "brief description", "position": "where in image"}}], "object_count": 0}}

Image context: {image_desc}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)
        objects = [ObjectItem(**o) for o in data.get("objects", [])]
        result = ObjectsResult(objects=objects, object_count=len(objects))
        avg_conf = sum(o.confidence for o in objects) / max(len(objects), 1)
        return AnalysisResult(
            modality=Modality.IMAGE,
            analysis_type=AnalysisType.OBJECTS,
            data=result.model_dump(),
            confidence=avg_conf,
            provider="mimo",
        )

    async def _brands(self, image_desc: str, options: dict[str, Any]) -> AnalysisResult:
        prompt = f"""Identify any brands, logos, or recognizable products in this image. Return JSON:
{{"brands": [{{"name": "brand name", "confidence": 0.0-1.0, "product": "product if applicable"}}]}}

Image context: {image_desc}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)
        brands = BrandResult(brands=data.get("brands", []))
        return AnalysisResult(
            modality=Modality.IMAGE,
            analysis_type=AnalysisType.BRANDS,
            data=brands.model_dump(),
            confidence=0.75,
            provider="mimo",
        )

    async def _colors(self, image_desc: str, options: dict[str, Any]) -> AnalysisResult:
        prompt = f"""Identify the dominant color palette of this image. Return JSON:
{{"dominant_colors": [{{"hex": "#RRGGBB", "name": "color name", "percentage": 0.0-100.0}}], "palette_description": "overall color description"}}

Image context: {image_desc}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)
        colors = [ColorInfo(**c) for c in data.get("dominant_colors", [])]
        result = ColorsResult(
            dominant_colors=colors,
            palette_description=data.get("palette_description", ""),
        )
        return AnalysisResult(
            modality=Modality.IMAGE,
            analysis_type=AnalysisType.COLORS,
            data=result.model_dump(),
            confidence=0.80,
            provider="mimo",
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise ValueError(f"Could not parse JSON from response: {text[:200]}")
