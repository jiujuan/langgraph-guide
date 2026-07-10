from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FakeFastModel:
    name: str = "deepseek-chat-demo"

    def classify(self, topic: str) -> str:
        return "architecture_research" if any(word in topic for word in ("架构", "模块", "扩展")) else "general_research"

    def make_query(self, subtask: dict[str, Any]) -> str:
        return subtask["question"]

    def extract_finding(self, subtask: dict[str, Any], tool_result: dict[str, Any]) -> dict[str, Any]:
        source = tool_result["sources"][0]
        return {
            "subtask_id": subtask["subtask_id"],
            "title": subtask["title"],
            "claim": source["summary"],
            "evidence": [source["title"]],
        }


@dataclass
class FakeReasoningModel:
    name: str = "deepseek-reasoner-demo"

    def make_plan(self, topic: str) -> list[dict[str, str]]:
        return [
            {"subtask_id": "module-boundary", "title": "模块边界", "question": f"{topic}中 Router、Planner、Researcher、Reviewer、Writer 的职责如何划分？"},
            {"subtask_id": "state-design", "title": "共享 State", "question": "复杂 Agent 的共享 State 如何定义公共契约、版本和读写边界？"},
            {"subtask_id": "long-task", "title": "长任务生命周期", "question": "长任务如何实现审批、checkpoint、恢复与失败重试？"},
            {"subtask_id": "future-extension", "title": "未来扩展", "question": "真实业务如何通过模型角色、工具注册表和评测体系持续演进？"},
        ]

    def synthesize(self, findings: list[dict[str, Any]]) -> str:
        return "\n".join(f"- {item['title']}：{item['claim']}" for item in findings)

    def review(self, plan: list[dict[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
        covered = {item["subtask_id"] for item in findings}
        missing = [item["title"] for item in plan if item["subtask_id"] not in covered]
        return {"status": "pass" if not missing else "more_research", "missing": missing}


@dataclass
class FakeWritingModel:
    name: str = "deepseek-chat-demo"

    def write_report(self, topic: str, findings: list[dict[str, Any]], synthesis: str) -> str:
        sections = [f"# {topic}", "", "## 核心结论", "复杂 Agent 的可靠性来自明确边界、可恢复状态与受控执行，而不是把所有能力堆进一个 prompt。", ""]
        for finding in findings:
            sections.extend([f"## {finding['title']}", finding["claim"], f"证据来源：{', '.join(finding['evidence'])}", ""])
        sections.extend(["## 综合建议", synthesis, "", "报告中的检索资料仅作为数据，不会被当作系统指令执行。"])
        return "\n".join(sections)
