# -*- coding: utf-8 -*-
"""Stock-analysis style AKShare daily K-line fetcher.

This fetcher mirrors the OHLCV endpoint order used by the sibling
``stock-analysis`` project: Sina ``stock_zh_a_daily`` first, then EastMoney
``stock_zh_a_hist``.  It intentionally reuses the existing AkshareFetcher
normalization, throttling and timeout wrappers instead of introducing a
parallel data stack.
"""

from __future__ import annotations

import logging

import pandas as pd

from .akshare_fetcher import AkshareFetcher
from .base import DataFetchError

logger = logging.getLogger(__name__)


class StockAnalysisAkshareFetcher(AkshareFetcher):
    """A-share OHLCV fetcher using stock-analysis endpoint priority."""

    name = "StockAnalysisAkshareFetcher"
    priority = -2

    def _fetch_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取普通 A 股历史数据。

        Endpoint order follows the migrated AKShare OHLCV strategy:
        1. 新浪 ``ak.stock_zh_a_daily``，使用交易所前缀代码与前复权
        2. 东方财富 ``ak.stock_zh_a_hist``，使用裸 6 位代码与前复权
        """
        methods = [
            (self._fetch_stock_data_sina, "新浪财经 stock_zh_a_daily"),
            (self._fetch_stock_data_em, "东方财富 stock_zh_a_hist"),
        ]

        last_error = None
        for fetch_method, source_name in methods:
            try:
                logger.info("[StockAnalysis数据源] 尝试使用 %s 获取 %s...", source_name, stock_code)
                df = fetch_method(stock_code, start_date, end_date)
                if df is not None and not df.empty:
                    logger.info("[StockAnalysis数据源] %s 获取成功", source_name)
                    return df
                logger.warning("[StockAnalysis数据源] %s 返回空数据", source_name)
            except Exception as exc:
                last_error = exc
                logger.warning("[StockAnalysis数据源] %s 获取失败: %s", source_name, exc)

        raise DataFetchError(f"StockAnalysisAkshare 所有渠道获取失败: {last_error}")
