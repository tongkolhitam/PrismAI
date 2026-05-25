"""Parallel batch analysis with progress tracking."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Coroutine

from prismai.models.schemas import (
    AnalysisRequest,
    AnalysisResult,
    BatchRequest,
    BatchResult,
    Priority,
    TaskStatus,
)
from prismai.pipeline.cache import AnalysisCache
from prismai.utils.logger import get_logger

logger = get_logger(__name__)


class BatchProcessor:
    """Processes multiple analysis requests in parallel."""

    def __init__(self, max_workers: int = 5) -> None:
        self.max_workers = max_workers
        self._batches: dict[str, BatchResult] = {}
        self._cache = AnalysisCache()
        self._analyzers: dict[str, Callable[..., Coroutine[Any, Any, list[AnalysisResult]]]] = {}

    def register_analyzer(
        self, name: str, fn: Callable[..., Coroutine[Any, Any, list[AnalysisResult]]]
    ) -> None:
        """Register an analysis function for batch processing."""
        self._analyzers[name] = fn

    async def process_batch(
        self,
        batch_request: BatchRequest,
        analyze_fn: Callable[[AnalysisRequest], Coroutine[Any, Any, list[AnalysisResult]]],
    ) -> BatchResult:
        """Process a batch of analysis requests in parallel."""
        batch_id = str(uuid.uuid4())
        batch_result = BatchResult(
            batch_id=batch_id,
            total=len(batch_request.items),
            status=TaskStatus.PROCESSING,
        )
        self._batches[batch_id] = batch_result

        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_item(item: AnalysisRequest) -> list[AnalysisResult]:
            async with semaphore:
                # Check cache first
                cache_key = AnalysisCache.make_key(
                    item.content[:500],
                    item.modality.value,
                    [a.value for a in item.analyses],
                )
                cached = await self._cache.get(cache_key)
                if cached:
                    logger.info("cache_hit", batch_id=batch_id)
                    return [AnalysisResult(**r) for r in cached]

                results = await analyze_fn(item)
                # Cache results
                await self._cache.set(
                    cache_key, [r.model_dump() for r in results]
                )
                return results

        # Run all items concurrently
        start = time.monotonic()
        tasks = [process_item(item) for item in batch_request.items]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                batch_result.failed += 1
                batch_result.results.append(
                    AnalysisResult(
                        success=False,
                        error=str(result),
                    )
                )
            elif isinstance(result, list):
                batch_result.completed += sum(1 for r in result if r.success)
                batch_result.failed += sum(1 for r in result if not r.success)
                batch_result.results.extend(result)

        elapsed_ms = (time.monotonic() - start) * 1000
        batch_result.status = (
            TaskStatus.COMPLETED if batch_result.failed == 0 else TaskStatus.COMPLETED
        )

        from datetime import datetime
        batch_result.completed_at = datetime.utcnow()

        logger.info(
            "batch_complete",
            batch_id=batch_id,
            total=batch_result.total,
            completed=batch_result.completed,
            failed=batch_result.failed,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return batch_result

    def get_batch(self, batch_id: str) -> BatchResult | None:
        return self._batches.get(batch_id)

    def get_all_batches(self) -> list[BatchResult]:
        return list(self._batches.values())
