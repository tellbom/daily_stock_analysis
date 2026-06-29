# -*- coding: utf-8 -*-
"""
===================================
数据源策略层 - 包初始化
===================================

本包实现策略模式管理多个数据源，实现：
1. 统一的数据获取接口
2. 自动故障切换
3. 防封禁流控策略

数据源优先级（动态调整）：
【默认 A 股日线路径】
1. StockAnalysisAkshareFetcher (Priority -2) - stock-analysis 风格 AKShare：新浪 -> 东方财富
2. EfinanceFetcher (Priority 0) - 来自 efinance 库
3. TencentFetcher (Priority 0) - 腾讯直连接口
4. AkshareFetcher (Priority 1) - 来自 akshare 库：东方财富 -> 新浪 -> 腾讯
5. PytdxFetcher (Priority 2) - 来自 pytdx 库（通达信）
6. BaostockFetcher (Priority 3) - 来自 baostock 库
7. YfinanceFetcher (Priority 4) - 来自 yfinance 库
8. LongbridgeFetcher (Priority 5) - 长桥 OpenAPI（美股/港股兜底）

【配置了 TUSHARE_TOKEN 时】
TushareFetcher 会作为可选后备源实例化；StockAnalysisAkshareFetcher 仍排在 Tushare 之前。

提示：优先级数字越小越优先，同优先级按初始化顺序排列
"""

from .base import BaseFetcher, DataFetcherManager
from .stock_analysis_akshare_fetcher import StockAnalysisAkshareFetcher
from .efinance_fetcher import EfinanceFetcher
from .tencent_fetcher import TencentFetcher
from .akshare_fetcher import AkshareFetcher, is_hk_stock_code
from .tushare_fetcher import TushareFetcher
from .pytdx_fetcher import PytdxFetcher
from .baostock_fetcher import BaostockFetcher
from .yfinance_fetcher import YfinanceFetcher
from .longbridge_fetcher import LongbridgeFetcher
from .finnhub_fetcher import FinnhubFetcher
from .alphavantage_fetcher import AlphaVantageFetcher
from .us_index_mapping import is_us_index_code, is_us_stock_code, get_us_index_yf_symbol, US_INDEX_MAPPING

__all__ = [
    'BaseFetcher',
    'DataFetcherManager',
    'StockAnalysisAkshareFetcher',
    'EfinanceFetcher',
    'TencentFetcher',
    'AkshareFetcher',
    'TushareFetcher',
    'PytdxFetcher',
    'BaostockFetcher',
    'YfinanceFetcher',
    'LongbridgeFetcher',
    'FinnhubFetcher',
    'AlphaVantageFetcher',
    'is_us_index_code',
    'is_us_stock_code',
    'is_hk_stock_code',
    'get_us_index_yf_symbol',
    'US_INDEX_MAPPING',
]
