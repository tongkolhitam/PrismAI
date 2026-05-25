"""Multi-modal fusion — cross-modal reasoning combining text, image, and audio insights."""

from __future__ import annotations

import json
import time
from typing import Any

from prismai.models.schemas import (
    AnalysisResult,
    AnalysisType,
    FusionInsight,
    Modality,
    MultimodalResult,
)
from prismai.providers.fallback_provider import FallbackProvider
from prismai.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are PrismAI's multi-modal fusion engine. Your job is to combine 
insights from text, image, and audio analyses to produce a comprehensive understanding 
of the content. Always respond with valid JSON."""


class MultimodalFusionEngine:
    """Combines insights from multiple modalities for unified understanding."""

    def __init__(self, provider: FallbackProvider) -> None:
        self.provider = provider

    async def fuse(
        self,
        text_results: list[AnalysisResult] | None = None,
        image_results: list[AnalysisResult] | None = None,
        audio_results: list[AnalysisResult] | None = None,
    ) -> AnalysisResult:
        """Fuse results from multiple modalities into unified insights."""
        start = time.monotonic()

        text_results = text_results or []
        image_results = image_results or []
        audio_results = audio_results or []

        # Build context from all results
        context = self._build_context(text_results, image_results, audio_results)

        prompt = f"""Given the following analysis results from multiple modalities, 
provide a comprehensive multi-modal fusion analysis.

{context}

Return JSON:
{{
    "unified_summary": "comprehensive summary combining all modalities",
    "cross_modal_insights": [
        {{"category": "theme|pattern|contradiction|correlation", "insight": "description", "confidence": 0.0-1.0, "contributing_modalities": ["text", "image", "audio"]}}
    ],
    "overall_sentiment": {{"label": "positive|negative|neutral", "confidence": 0.0-1.0}},
    "content_topics": ["topic1", "topic2"],
    "confidence_score": 0.0-1.0
}}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)

        # Build fusion result
        insights = [
            FusionInsight(**i) for i in data.get("cross_modal_insights", [])
        ]

        fusion = MultimodalResult(
            unified_summary=data.get("unified_summary", ""),
            cross_modal_insights=insights,
            content_topics=data.get("content_topics", []),
            confidence_score=float(data.get("confidence_score", 0.5)),
        )

        # Add overall sentiment if present
        if "overall_sentiment" in data:
            from prismai.models.schemas import SentimentLabel, SentimentResult

            fusion.overall_sentiment = SentimentResult(
                label=SentimentLabel(data["overall_sentiment"].get("label", "neutral")),
                confidence=float(data["overall_sentiment"].get("confidence", 0.5)),
            )

        elapsed_ms = (time.monotonic() - start) * 1000
        return AnalysisResult(
            modality=Modality.MULTIMODAL,
            analysis_type=AnalysisType.SENTIMENT,  # Generic type for fusion
            data=fusion.model_dump(),
            confidence=fusion.confidence_score,
            processing_time_ms=elapsed_ms,
            provider="mimo",
        )

    def _build_context(
        self,
        text_results: list[AnalysisResult],
        image_results: list[AnalysisResult],
        audio_results: list[AnalysisResult],
    ) -> str:
        """Build analysis context string from all results."""
        parts = []

        if text_results:
            parts.append("=== TEXT ANALYSIS RESULTS ===")
            for r in text_results:
                if r.success:
                    parts.append(f"- {r.analysis_type.value}: {json.dumps(r.data, default=str)[:500]}")

        if image_results:
            parts.append("\n=== IMAGE ANALYSIS RESULTS ===")
            for r in image_results:
                if r.success:
                    parts.append(f"- {r.analysis_type.value}: {json.dumps(r.data, default=str)[:500]}")

        if audio_results:
            parts.append("\n=== AUDIO ANALYSIS RESULTS ===")
            for r in audio_results:
                if r.success:
                    parts.append(f"- {r.analysis_type.value}: {json.dumps(r.data, default=str)[:500]}")

        return "\n".join(parts) if parts else "No analysis results provided."

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
