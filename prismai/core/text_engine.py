"""Text analysis engine — sentiment, NER, classification, summarization, language detection."""

from __future__ import annotations

import json
import time
from typing import Any

from prismai.models.schemas import (
    AnalysisResult,
    AnalysisType,
    ClassificationResult,
    Entity,
    LanguageResult,
    Modality,
    NERResult,
    SentimentLabel,
    SentimentResult,
    SummaryResult,
)
from prismai.providers.fallback_provider import FallbackProvider
from prismai.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are PrismAI's text analysis engine. Analyze text content accurately.
Always respond with valid JSON matching the requested schema."""


class TextEngine:
    """Handles all text-based analyses using LLM."""

    def __init__(self, provider: FallbackProvider) -> None:
        self.provider = provider

    async def analyze(
        self,
        text: str,
        analyses: list[AnalysisType],
        options: dict[str, Any] | None = None,
    ) -> list[AnalysisResult]:
        """Run requested analyses on text."""
        options = options or {}
        results: list[AnalysisResult] = []

        for analysis_type in analyses:
            start = time.monotonic()
            try:
                result = await self._dispatch(text, analysis_type, options)
                result.processing_time_ms = (time.monotonic() - start) * 1000
                results.append(result)
            except Exception as exc:
                logger.error("text_analysis_failed", type=analysis_type, error=str(exc))
                results.append(
                    AnalysisResult(
                        modality=Modality.TEXT,
                        analysis_type=analysis_type,
                        success=False,
                        error=str(exc),
                        processing_time_ms=(time.monotonic() - start) * 1000,
                    )
                )
        return results

    async def _dispatch(
        self, text: str, analysis_type: AnalysisType, options: dict[str, Any]
    ) -> AnalysisResult:
        dispatch = {
            AnalysisType.SENTIMENT: self._sentiment,
            AnalysisType.NER: self._ner,
            AnalysisType.CLASSIFICATION: self._classification,
            AnalysisType.SUMMARY: self._summary,
            AnalysisType.LANGUAGE: self._language,
        }
        handler = dispatch.get(analysis_type)
        if not handler:
            raise ValueError(f"Unsupported text analysis: {analysis_type}")
        return await handler(text, options)

    async def _sentiment(self, text: str, options: dict[str, Any]) -> AnalysisResult:
        prompt = f"""Analyze the sentiment of the following text. Return JSON:
{{"label": "positive|negative|neutral", "confidence": 0.0-1.0, "scores": {{"positive": 0.0, "negative": 0.0, "neutral": 0.0}}}}

Text: {text[:3000]}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)
        sentiment = SentimentResult(
            label=SentimentLabel(data.get("label", "neutral")),
            confidence=float(data.get("confidence", 0.5)),
            scores=data.get("scores", {}),
        )
        return AnalysisResult(
            modality=Modality.TEXT,
            analysis_type=AnalysisType.SENTIMENT,
            data=sentiment.model_dump(),
            confidence=sentiment.confidence,
            provider="mimo",
        )

    async def _ner(self, text: str, options: dict[str, Any]) -> AnalysisResult:
        prompt = f"""Extract named entities from the text. Return JSON:
{{"entities": [{{"text": "...", "entity_type": "PERSON|ORG|LOC|DATE|PRODUCT|EVENT|MONEY", "confidence": 0.0-1.0}}]}}

Text: {text[:3000]}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)
        entities = [Entity(**e) for e in data.get("entities", [])]
        ner = NERResult(entities=entities, entity_count=len(entities))
        avg_conf = (
            sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        )
        return AnalysisResult(
            modality=Modality.TEXT,
            analysis_type=AnalysisType.NER,
            data=ner.model_dump(),
            confidence=avg_conf,
            provider="mimo",
        )

    async def _classification(self, text: str, options: dict[str, Any]) -> AnalysisResult:
        prompt = f"""Classify the topic/category of the following text. Return JSON:
{{"primary_topic": "...", "topics": [{{"name": "...", "confidence": 0.0-1.0}}], "categories": ["..."]}}

Text: {text[:3000]}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)
        classification = ClassificationResult(**data)
        top_conf = classification.topics[0]["confidence"] if classification.topics else 0.5
        return AnalysisResult(
            modality=Modality.TEXT,
            analysis_type=AnalysisType.CLASSIFICATION,
            data=classification.model_dump(),
            confidence=float(top_conf),
            provider="mimo",
        )

    async def _summary(self, text: str, options: dict[str, Any]) -> AnalysisResult:
        max_length = options.get("max_summary_length", 200)
        prompt = f"""Summarize the following text in {max_length} words or fewer. Return JSON:
{{"summary": "..."}}

Text: {text[:5000]}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)
        summary_text = data.get("summary", "")
        summary = SummaryResult(
            summary=summary_text,
            original_length=len(text),
            summary_length=len(summary_text),
            compression_ratio=len(summary_text) / max(len(text), 1),
        )
        return AnalysisResult(
            modality=Modality.TEXT,
            analysis_type=AnalysisType.SUMMARY,
            data=summary.model_dump(),
            confidence=0.85,
            provider="mimo",
        )

    async def _language(self, text: str, options: dict[str, Any]) -> AnalysisResult:
        prompt = f"""Detect the language of the following text. Return JSON:
{{"language": "English", "language_code": "en", "confidence": 0.0-1.0}}

Text: {text[:1000]}"""

        raw = await self.provider.complete(prompt, system=SYSTEM_PROMPT)
        data = self._parse_json(raw)
        language = LanguageResult(**data)
        return AnalysisResult(
            modality=Modality.TEXT,
            analysis_type=AnalysisType.LANGUAGE,
            data=language.model_dump(),
            confidence=language.confidence,
            provider="mimo",
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Extract JSON from LLM response (handles markdown fences)."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise ValueError(f"Could not parse JSON from response: {text[:200]}")
