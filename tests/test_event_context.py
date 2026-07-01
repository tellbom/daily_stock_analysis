# -*- coding: utf-8 -*-
"""Tests for lightweight stock event context construction."""

from __future__ import annotations

from src.services.event_context import build_event_context


class _FakeFetcher:
    def __init__(self, items=None, *, fail: bool = False):
        self.items = items or []
        self.fail = fail

    def fetch(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("boom")
        return {
            "items": list(self.items),
            "warnings": [],
            "source_chain": [{"provider": "fake.events", "result": "ok"}],
        }


def test_event_context_filters_future_items_and_keeps_required_fields() -> None:
    context = build_event_context(
        "600519",
        "贵州茅台",
        as_of_date="2026-05-24",
        lookback_days=7,
        fetcher=_FakeFetcher(
            [
                {
                    "type": "announcement",
                    "title": "贵州茅台发布利润增长公告",
                    "publish_time": "2026-05-24 10:00:00",
                    "source": "交易所公告",
                    "url": "https://example.test/a",
                    "summary": "归母净利润增长",
                    "tags": ["announcement", "earnings", "catalyst"],
                    "risk_level": "low",
                    "is_confirmed": True,
                },
                {
                    "type": "news",
                    "title": "未来新闻不得注入",
                    "publish_time": "2026-05-25",
                    "source": "媒体",
                    "summary": "未来发布时间",
                    "is_confirmed": False,
                },
            ]
        ),
    )

    assert context["status"] == "partial"
    assert len(context["items"]) == 1
    event = context["items"][0]
    assert event["type"] == "announcement"
    assert event["publish_time"] == "2026-05-24"
    assert event["source"] == "交易所公告"
    assert event["url"] == "https://example.test/a"
    assert event["is_confirmed"] is True
    assert "event_future_filtered" in context["warnings"]
    assert context["event_digest"]["positive_catalysts"]


def test_event_context_keeps_future_unlock_risk_events() -> None:
    context = build_event_context(
        "600519",
        "贵州茅台",
        as_of_date="2026-05-24",
        lookback_days=7,
        fetcher=_FakeFetcher(
            [
                {
                    "type": "risk_event",
                    "title": "贵州茅台未来12天存在限售股解禁",
                    "publish_time": "2026-06-05",
                    "source": "akshare.stock_restricted_release_detail_em",
                    "summary": "解禁日期=2026-06-05；占解禁前流通市值比例=6.2%",
                    "tags": ["risk_event", "unlock", "risk"],
                    "risk_level": "high",
                    "is_confirmed": True,
                },
                {
                    "type": "news",
                    "title": "未来新闻仍应过滤",
                    "publish_time": "2026-06-05",
                    "source": "媒体",
                    "summary": "未来新闻",
                    "is_confirmed": False,
                },
            ]
        ),
    )

    assert len(context["items"]) == 1
    assert context["items"][0]["type"] == "risk_event"
    assert context["items"][0]["publish_time"] == "2026-06-05"
    assert context["items"][0]["risk_level"] == "high"
    assert context["event_digest"]["event_bias"] == "negative"
    assert context["event_digest"]["negative_risks"]
    assert "event_future_filtered" in context["warnings"]


def test_event_context_dedupes_similar_titles_with_announcement_priority() -> None:
    context = build_event_context(
        "600519",
        as_of_date="2026-05-24",
        fetcher=_FakeFetcher(
            [
                {
                    "type": "news",
                    "title": "贵州茅台发布业绩预增公告",
                    "publish_time": "2026-05-24",
                    "source": "财经媒体",
                    "summary": "媒体转载",
                    "is_confirmed": False,
                },
                {
                    "type": "announcement",
                    "title": "贵州茅台发布业绩预增公告",
                    "publish_time": "2026-05-24",
                    "source": "交易所公告",
                    "summary": "公司公告",
                    "is_confirmed": True,
                },
            ]
        ),
    )

    assert len(context["items"]) == 1
    assert context["items"][0]["type"] == "announcement"
    assert context["items"][0]["is_confirmed"] is True


def test_event_context_no_recent_events_and_fetch_failure_are_fail_open() -> None:
    empty = build_event_context(
        "600519",
        as_of_date="2026-05-24",
        fetcher=_FakeFetcher([]),
    )
    assert empty["status"] == "missing"
    assert empty["event_digest"]["status"] == "no_recent_events_found"
    assert empty["event_digest"]["uncertainties"] == ["no_recent_events_found"]

    failed = build_event_context(
        "600519",
        as_of_date="2026-05-24",
        fetcher=_FakeFetcher(fail=True),
    )
    assert failed["status"] == "failed"
    assert failed["items"] == []
    assert "event_fetcher_failed:RuntimeError" in failed["warnings"]
