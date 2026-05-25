"""WebSocket endpoint for real-time analysis streaming."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from prismai.models.schemas import AnalysisRequest
from prismai.utils.logger import get_logger

logger = get_logger(__name__)
ws_router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("ws_connected", total=len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)
        logger.info("ws_disconnected", total=len(self.active_connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@ws_router.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis.
    
    Send JSON messages with analysis requests and receive results in real-time.
    
    Example message:
    {
        "content": "Your text to analyze",
        "modality": "text",
        "analyses": ["sentiment", "ner"]
    }
    """
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                request = AnalysisRequest(**payload)

                # Send acknowledgment
                await websocket.send_json({
                    "type": "status",
                    "status": "processing",
                    "message": f"Analyzing {request.modality.value} content...",
                })

                from prismai.api.server import get_analyzer

                analyzer = get_analyzer()

                # Stream results as they come
                results = await analyzer.analyze(request)
                for result in results:
                    await websocket.send_json({
                        "type": "result",
                        "data": result.model_dump(mode="json"),
                    })

                await websocket.send_json({
                    "type": "status",
                    "status": "completed",
                    "message": f"Analysis complete. {len(results)} results.",
                })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                })
            except Exception as exc:
                logger.error("ws_analysis_error", error=str(exc))
                await websocket.send_json({
                    "type": "error",
                    "message": str(exc),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@ws_router.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis feed.
    
    Connect to receive all analyses performed on the platform in real-time.
    """
    await manager.connect(websocket)
    try:
        # Just keep connection open for broadcasts
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
