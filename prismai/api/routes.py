from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..core.analyzer import PrismAnalyzer

router = APIRouter()
analyzer = PrismAnalyzer()


class TextRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    image_path: Optional[str] = None
    description: Optional[str] = None

class AudioRequest(BaseModel):
    transcription: str
    duration_seconds: float = 0

class MultiModalRequest(BaseModel):
    text: Optional[str] = None
    image_description: Optional[str] = None
    audio_transcription: Optional[str] = None


@router.post("/analyze/text")
async def analyze_text(req: TextRequest):
    try:
        result = await analyzer.analyze_text(req.text)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/analyze/image")
async def analyze_image(req: ImageRequest):
    try:
        result = await analyzer.analyze_image(req.image_path, req.description)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/analyze/audio")
async def analyze_audio(req: AudioRequest):
    try:
        result = await analyzer.analyze_audio(req.transcription, req.duration_seconds)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/analyze/multimodal")
async def analyze_multimodal(req: MultiModalRequest):
    try:
        result = await analyzer.analyze_multimodal(req.text, req.image_description, req.audio_transcription)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/stats")
async def get_stats():
    return analyzer.get_stats()


@router.get("/history")
async def get_history(limit: int = 20):
    return [r.model_dump() for r in analyzer.get_history(limit)]


@router.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
