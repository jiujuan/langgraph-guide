from __future__ import annotations

from typing import Any


def evaluate_report(report: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
    """规则评测保护结构与可追溯性，不代替人工质量审查。"""

    required_sections = ("模块边界", "共享 State", "长任务生命周期", "未来扩展")
    missing = [section for section in required_sections if section not in report]
    return {
        "passed": not missing and bool(metadata),
        "missing_sections": missing,
        "has_metadata": bool(metadata),
    }
