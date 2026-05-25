import pytest
import asyncio
from prismai.core.analyzer import PrismAnalyzer


def test_analyzer_init():
    a = PrismAnalyzer()
    assert a.provider is not None
    assert len(a._history) == 0


def test_stats_empty():
    a = PrismAnalyzer()
    stats = a.get_stats()
    assert stats["total"] == 0


def test_history_empty():
    a = PrismAnalyzer()
    assert a.get_history() == []
