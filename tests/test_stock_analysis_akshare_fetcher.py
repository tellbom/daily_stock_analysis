# -*- coding: utf-8 -*-
"""Tests for the stock-analysis style AKShare fetcher."""

import pandas as pd
import pytest

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from data_provider.base import DataFetchError, DataFetcherManager
from data_provider.stock_analysis_akshare_fetcher import StockAnalysisAkshareFetcher


def test_stock_analysis_akshare_fetcher_prefers_sina_then_eastmoney(monkeypatch) -> None:
    fetcher = StockAnalysisAkshareFetcher(sleep_min=0, sleep_max=0)
    calls = []
    em_df = pd.DataFrame({"日期": ["2026-06-29"], "收盘": [10.0]})

    def fake_sina(*args):
        calls.append("sina")
        return pd.DataFrame()

    def fake_em(*args):
        calls.append("em")
        return em_df

    monkeypatch.setattr(fetcher, "_fetch_stock_data_sina", fake_sina)
    monkeypatch.setattr(fetcher, "_fetch_stock_data_em", fake_em)

    result = fetcher._fetch_stock_data("600519", "2026-06-01", "2026-06-29")

    assert result is em_df
    assert calls == ["sina", "em"]


def test_stock_analysis_akshare_fetcher_fails_loudly_when_all_endpoints_empty(monkeypatch) -> None:
    fetcher = StockAnalysisAkshareFetcher(sleep_min=0, sleep_max=0)

    monkeypatch.setattr(fetcher, "_fetch_stock_data_sina", lambda *args: pd.DataFrame())
    monkeypatch.setattr(fetcher, "_fetch_stock_data_em", lambda *args: pd.DataFrame())

    with pytest.raises(DataFetchError, match="StockAnalysisAkshare 所有渠道获取失败"):
        fetcher._fetch_stock_data("600519", "2026-06-01", "2026-06-29")


def test_manager_filters_stock_analysis_fetcher_to_cn_daily_route() -> None:
    stock_analysis = StockAnalysisAkshareFetcher(sleep_min=0, sleep_max=0)

    cn_fetchers = DataFetcherManager._filter_daily_fetchers_for_market([stock_analysis], "cn")
    hk_fetchers = DataFetcherManager._filter_daily_fetchers_for_market([stock_analysis], "hk")

    assert cn_fetchers == [stock_analysis]
    assert hk_fetchers == []
