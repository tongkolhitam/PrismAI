from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class TextAnalysis(BaseModel):
    sentiment: Sentiment
    sentiment_confidence: float = Field(ge=0, le=1)
    entities: List[dict] = []
    topics: List[str] = []
    summary: Optional[str] = None
    language: str = "en"
    word_count: int = 0


class ImageAnalysis(BaseModel):
    description: str
    objects: List[dict] = []
    ocr_text: Optional[str] = None
    colors: List[str] = []
    scene_type: Optional[str] = None
    brands: List[str] = []


class AudioAnalysis(BaseModel):
    transcription: Optional[str] = None
    speaker_count: int = 1
    emotion: Optional[str] = None
    duration_seconds: float = 0
    key_topics: List[str] = []


class AnalysisResult(BaseModel):
    id: str
    modality: Modality
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    text: Optional[TextAnalysis] = None
    image: Optional[ImageAnalysis] = None
    audio: Optional[AudioAnalysis] = None
    fusion_summary: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    processing_time_ms: float = 0
    model_used: str = "mimo-v2.5-pro"


class BatchRequest(BaseModel):
    items: List[dict]
    priority: int = Field(default=5, ge=1, le=10)


class AnalysisStats(BaseModel):
    total_analyses: int = 0
    avg_confidence: float = 0
    avg_processing_ms: float = 0
    modality_breakdown: dict = {}
    analyses_today: int = 0
