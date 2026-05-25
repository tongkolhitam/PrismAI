# 🔮 PrismAI — Multi-Modal Content Intelligence Platform

> AI-powered analysis for text, images, and audio — powered by MiMo v2.5 Pro

## Features

- **📝 Text Analysis** — Sentiment detection, NER, topic classification, summarization
- **🖼️ Image Analysis** — OCR, scene understanding, object detection, brand recognition
- **🎵 Audio Analysis** — Transcription, emotion detection, speaker identification
- **🔮 Multi-Modal Fusion** — Cross-modal reasoning for comprehensive insights
- **⚡ Batch Processing** — Parallel analysis with progress tracking
- **📊 Real-time Dashboard** — Live analytics, confidence metrics, modality breakdown

## Architecture

```
Input → Modality Detection → Provider Cascade (MiMo → DeepSeek → OpenRouter)
      → Analysis Engine → Fusion Layer → Structured Output
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your MiMo API key
python main.py
```

Open `http://localhost:8000` for the dashboard.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze/text` | Analyze text content |
| POST | `/api/v1/analyze/image` | Analyze image |
| POST | `/api/v1/analyze/audio` | Analyze audio transcription |
| POST | `/api/v1/analyze/multimodal` | Multi-modal fusion analysis |
| GET | `/api/v1/stats` | Platform analytics |
| GET | `/api/v1/history` | Recent analyses |
| GET | `/api/v1/health` | Health check |

## Tech Stack

- **AI Engine**: MiMo v2.5 Pro (Xiaomi)
- **Backend**: Python 3.11+, FastAPI, Pydantic v2
- **Async**: httpx, asyncio for concurrent processing
- **Provider Cascade**: MiMo → OpenRouter → DeepSeek (auto-fallback)
- **Dashboard**: Tailwind CSS, Chart.js

## License

MIT
