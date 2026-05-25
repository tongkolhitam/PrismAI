import httpx
import json
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger("mimo_provider")


class MiMoProvider:
    """OpenAI-compatible provider with cascade fallback."""

    def __init__(self):
        self.config = Config()

    async def chat(self, messages: list, temperature: float = 0.3, max_tokens: int = 2000) -> str:
        for provider in self.config.PROVIDER_CASCADE:
            if not provider["api_key"]:
                continue
            try:
                return await self._call(provider, messages, temperature, max_tokens)
            except Exception as e:
                logger.warning(f"Provider {provider['name']} failed: {e}")
                continue
        raise RuntimeError("All providers failed")

    async def _call(self, provider: dict, messages: list, temperature: float, max_tokens: int) -> str:
        url = f"{provider['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": provider["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def analyze_text(self, text: str) -> dict:
        messages = [
            {"role": "system", "content": """Analyze the text and return JSON:
{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0, "entities": [{"text": "...", "type": "person|org|location|other"}], "topics": ["topic1", "topic2"], "summary": "one sentence summary", "language": "en|id|...", "word_count": N}"""},
            {"role": "user", "content": f"Analyze this text:\n\n{text[:3000]}"}
        ]
        result = await self.chat(messages)
        try:
            return json.loads(result.strip().strip("```json").strip("```"))
        except:
            return {"sentiment": "neutral", "confidence": 0.5, "entities": [], "topics": [], "summary": text[:100], "language": "en", "word_count": len(text.split())}

    async def analyze_image_description(self, description: str) -> dict:
        messages = [
            {"role": "system", "content": """Based on image description, return JSON:
{"description": "detailed description", "objects": [{"name": "object", "confidence": 0.9}], "scene_type": "indoor|outdoor|...", "colors": ["#hex1", "#hex2"], "brands": ["brand1"]}"""},
            {"role": "user", "content": f"Analyze this image description:\n\n{description}"}
        ]
        result = await self.chat(messages)
        try:
            return json.loads(result.strip().strip("```json").strip("```"))
        except:
            return {"description": description, "objects": [], "scene_type": "unknown", "colors": [], "brands": []}

    async def analyze_audio_description(self, transcription: str) -> dict:
        messages = [
            {"role": "system", "content": """Based on audio transcription, return JSON:
{"emotion": "happy|sad|angry|neutral|excited", "key_topics": ["topic1", "topic2"], "speaker_count": N}"""},
            {"role": "user", "content": f"Analyze this transcription:\n\n{transcription[:3000]}"}
        ]
        result = await self.chat(messages)
        try:
            return json.loads(result.strip().strip("```json").strip("```"))
        except:
            return {"emotion": "neutral", "key_topics": [], "speaker_count": 1}

    async def fusion_analysis(self, text_data: dict = None, image_data: dict = None, audio_data: dict = None) -> str:
        context_parts = []
        if text_data:
            context_parts.append(f"Text analysis: {json.dumps(text_data)}")
        if image_data:
            context_parts.append(f"Image analysis: {json.dumps(image_data)}")
        if audio_data:
            context_parts.append(f"Audio analysis: {json.dumps(audio_data)}")

        messages = [
            {"role": "system", "content": "You are a multi-modal content analyst. Synthesize all available analysis into a comprehensive 2-3 sentence insight."},
            {"role": "user", "content": "\n\n".join(context_parts)}
        ]
        return await self.chat(messages, max_tokens=500)
