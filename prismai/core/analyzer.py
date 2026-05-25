import uuid
import time
from datetime import datetime
from ..providers.mimo_provider import MiMoProvider
from ..models.schemas import AnalysisResult, Modality, TextAnalysis, ImageAnalysis, AudioAnalysis
from ..utils.logger import get_logger

logger = get_logger("analyzer")


class PrismAnalyzer:
    def __init__(self):
        self.provider = MiMoProvider()
        self._history: list[AnalysisResult] = []

    async def analyze_text(self, text: str) -> AnalysisResult:
        start = time.time()
        data = await self.provider.analyze_text(text)
        elapsed = (time.time() - start) * 1000

        result = AnalysisResult(
            id=str(uuid.uuid4())[:8],
            modality=Modality.TEXT,
            text=TextAnalysis(**data),
            confidence=data.get("confidence", 0.5),
            processing_time_ms=round(elapsed, 1),
        )
        self._history.append(result)
        logger.info(f"Text analysis done: {result.confidence:.0%} confidence in {elapsed:.0f}ms")
        return result

    async def analyze_image(self, image_path: str = None, description: str = None) -> AnalysisResult:
        start = time.time()
        if not description:
            description = f"Image at {image_path}"
        data = await self.provider.analyze_image_description(description)
        elapsed = (time.time() - start) * 1000

        result = AnalysisResult(
            id=str(uuid.uuid4())[:8],
            modality=Modality.IMAGE,
            image=ImageAnalysis(**data),
            confidence=0.8,
            processing_time_ms=round(elapsed, 1),
        )
        self._history.append(result)
        logger.info(f"Image analysis done in {elapsed:.0f}ms")
        return result

    async def analyze_audio(self, transcription: str, duration: float = 0) -> AnalysisResult:
        start = time.time()
        data = await self.provider.analyze_audio_description(transcription)
        elapsed = (time.time() - start) * 1000

        result = AnalysisResult(
            id=str(uuid.uuid4())[:8],
            modality=Modality.AUDIO,
            audio=AudioAnalysis(**data, transcription=transcription, duration_seconds=duration),
            confidence=0.75,
            processing_time_ms=round(elapsed, 1),
        )
        self._history.append(result)
        logger.info(f"Audio analysis done in {elapsed:.0f}ms")
        return result

    async def analyze_multimodal(self, text: str = None, image_desc: str = None, audio_transcript: str = None) -> AnalysisResult:
        start = time.time()
        text_data, image_data, audio_data = None, None, None

        if text:
            text_data = await self.provider.analyze_text(text)
        if image_desc:
            image_data = await self.provider.analyze_image_description(image_desc)
        if audio_transcript:
            audio_data = await self.provider.analyze_audio_description(audio_transcript)

        fusion = await self.provider.fusion_analysis(text_data, image_data, audio_data)
        elapsed = (time.time() - start) * 1000

        result = AnalysisResult(
            id=str(uuid.uuid4())[:8],
            modality=Modality.MULTIMODAL,
            text=TextAnalysis(**text_data) if text_data else None,
            image=ImageAnalysis(**image_data) if image_data else None,
            audio=AudioAnalysis(**audio_data) if audio_data else None,
            fusion_summary=fusion,
            confidence=0.85,
            processing_time_ms=round(elapsed, 1),
        )
        self._history.append(result)
        logger.info(f"Multi-modal analysis done in {elapsed:.0f}ms")
        return result

    def get_stats(self) -> dict:
        if not self._history:
            return {"total": 0, "avg_confidence": 0, "avg_ms": 0, "modalities": {}, "today": 0}
        today = datetime.utcnow().date()
        return {
            "total": len(self._history),
            "avg_confidence": round(sum(r.confidence for r in self._history) / len(self._history), 3),
            "avg_ms": round(sum(r.processing_time_ms for r in self._history) / len(self._history), 1),
            "modalities": {m.value: sum(1 for r in self._history if r.modality == m) for m in Modality},
            "today": sum(1 for r in self._history if r.timestamp.date() == today),
        }

    def get_history(self, limit: int = 20) -> list:
        return self._history[-limit:]
