# -*- coding: utf-8 -*-
"""Regression tests for conditional short-term decision reports."""

import json
from datetime import date
from unittest.mock import MagicMock, patch

from src.analyzer import AnalysisResult, GeminiAnalyzer
from src.config import Config
from src.market_analyzer import MarketAnalyzer, MarketOverview
from src.notification import NotificationService
from src.services.daily_market_context import (
    DailyMarketContextService,
    format_daily_market_context_prompt_section,
)


def _analyzer() -> GeminiAnalyzer:
    config = MagicMock()
    config.report_language = "zh"
    config.enable_phase_classification = False
    config.enable_pre_judge = False
    config.pre_judge_decision_filter = False
    config.enable_knowledge_base = False
    config.use_agent_analysis = False
    config.use_multi_agent = False
    config.enable_stagewise_analysis = False
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer.config = config
    analyzer.phase_classifier = None
    analyzer.pre_judge = None
    return analyzer


def test_parse_response_mirrors_conditional_decision_into_dashboard() -> None:
    response = json.dumps(
        {
            "stock_name": "中远海能",
            "sentiment_score": 42,
            "trend_prediction": "震荡",
            "operation_advice": "观望",
            "decision_type": "hold",
            "confidence_level": "中",
            "decision": "观望",
            "score": 42,
            "trend": "震荡",
            "confidence": "中",
            "one_sentence_summary": "等放量修复再考虑。",
            "core_reasons": ["短线资金未确认"],
            "tomorrow_watchlist": ["能否站回5日线"],
            "trigger_strategy": {
                "bearish_continue": "跌破支撑减仓",
                "neutral_observe": "缩量震荡观望",
                "bullish_repair": "放量站回5日线试错",
            },
            "position_advice": {
                "empty_position": "空仓等待",
                "light_position": "轻仓跟踪",
                "heavy_position": "重仓降至轻仓",
            },
            "risk_alerts": ["资金流未确认"],
            "positive_catalysts": ["航运景气改善"],
            "external_market_reference": {
                "status": "not_relevant",
                "summary": "非科技链标的",
                "impact": "无直接影响",
                "data_as_of": "N/A",
            },
            "data_completeness": {
                "used": ["daily_bars", "quant_factor_context"],
                "missing": ["margin"],
                "impact": "融资融券缺失降低风险判断精度",
            },
            "dashboard": {
                "core_conclusion": {"one_sentence": "观望等待"},
                "intelligence": {"risk_alerts": []},
                "battle_plan": {"sniper_points": {"stop_loss": "跌破8元"}},
                "phase_decision": {
                    "phase_context": {"phase": "postmarket"},
                    "action_window": "盘后复盘",
                    "immediate_action": "观察",
                    "watch_conditions": ["站回5日线"],
                    "next_check_time": "明日开盘",
                    "confidence_reason": "数据基本可用",
                    "data_limitations": [],
                },
            },
            "analysis_summary": "短线观望",
            "key_points": "资金未确认",
            "risk_warning": "跌破支撑",
            "buy_reason": "等待确认",
            "search_performed": False,
            "data_sources": "本地行情",
        },
        ensure_ascii=False,
    )

    result = _analyzer()._parse_response(response, "600026", "中远海能")

    assert result.conditional_decision["decision"] == "观望"
    assert result.dashboard["conditional_decision"]["trigger_strategy"]["bullish_repair"] == "放量站回5日线试错"
    assert result.external_market_reference["status"] == "not_relevant"
    assert result.to_dict()["conditional_decision"]["data_completeness"]["missing"] == ["margin"]


@patch("src.notification.get_config")
def test_wechat_dashboard_uses_conditional_entry_and_limits_risks(mock_get_config: MagicMock) -> None:
    mock_get_config.return_value = Config(stock_list=[], report_renderer_enabled=False)
    result = AnalysisResult(
        code="600026",
        name="中远海能",
        sentiment_score=38,
        trend_prediction="看空",
        operation_advice="减仓/观望",
        decision_type="sell",
        confidence_level="中",
        analysis_summary="风险偏高",
        dashboard={
            "core_conclusion": {"one_sentence": "先控仓，等修复。"},
            "conditional_decision": {"tomorrow_watchlist": ["是否止跌", "资金是否回流"]},
            "intelligence": {
                "risk_alerts": ["风险1", "风险2", "风险3", "风险4"],
                "positive_catalysts": [],
            },
            "battle_plan": {
                "sniper_points": {
                    "ideal_buy": "理想买入点：放量站回9.5元后再看",
                    "stop_loss": "跌破8.8元",
                    "take_profit": "反弹看10元压力",
                }
            },
        },
    )

    content = NotificationService().generate_wechat_dashboard([result])

    assert "潜在买点条件" in content
    assert "风控参考" in content
    assert "反弹/目标参考" in content
    assert "更多风险见完整报告" in content
    assert "风险4" not in content


def test_market_review_payload_includes_us_tech_summary() -> None:
    analyzer = MarketAnalyzer(region="cn", config=Config(stock_list=[]))
    with patch("src.market_analyzer.build_us_tech_summary") as summary:
        summary.return_value = {
            "status": "available",
            "as_of": "2026-07-01",
            "summary": "纳指+1.00%。",
            "possible_impact_on_a_share": "利于A股科技映射。",
        }
        payload = analyzer.build_market_review_payload(MarketOverview(date="2026-07-02"), [], "## 复盘")

    assert payload["us_tech_summary"]["status"] == "available"
    assert payload["us_tech_summary"]["summary"] == "纳指+1.00%。"


def test_daily_market_context_keeps_us_tech_summary_in_safe_prompt() -> None:
    context = DailyMarketContextService(
        db_manager=MagicMock(),
        today_fn=lambda: date(2026, 7, 2),
    )._build_context_from_payload(
        region="cn",
        trade_date=date(2026, 7, 2),
        payload={
            "region": "cn",
            "summary": "A股震荡。",
            "us_tech_summary": {
                "status": "available",
                "as_of": "2026-07-01",
                "summary": "费半+1.2%。",
                "possible_impact_on_a_share": "半导体情绪映射偏正面。",
                "api_key": "secret",
            },
        },
        source="test",
    )

    safe = context.to_safe_dict()
    section = format_daily_market_context_prompt_section(safe, report_language="zh")

    assert safe["us_tech_summary"]["status"] == "available"
    assert "美股科技链参考" in section
    assert "费半+1.2%" in section
    assert "secret" not in section
