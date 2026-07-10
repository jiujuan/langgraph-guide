from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..models.factory import ModelBundle
from ..persistence.repositories import InMemoryAuditRepository
from ..tools.permissions import authorize_tool_request
from ..tools.registry import ToolRegistry


def append_log(state: dict[str, Any], message: str) -> list[str]:
    return [*state.get("execution_log", []), message]


@dataclass
class WorkflowNodes:
    models: ModelBundle
    tools: ToolRegistry
    audit: InMemoryAuditRepository

    def router(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"task_type": self.models.fast_model.classify(state["topic"]), "execution_log": append_log(state, "router: 分类研究任务")}

    def planner(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("plan") or self.models.reasoning_model.make_plan(state["topic"])
        return {"plan": plan, "execution_log": append_log(state, "planner: 生成或复用研究计划")}

    def approval(self, state: dict[str, Any]) -> dict[str, Any]:
        status = state.get("approval_status", "pending")
        lifecycle = "awaiting_approval" if status == "pending" else "queued"
        return {"lifecycle_status": lifecycle, "execution_log": append_log(state, f"approval: {status}")}

    def researcher(self, state: dict[str, Any]) -> dict[str, Any]:
        completed = set(state.get("completed_subtask_ids", []))
        subtask = next((item for item in state["plan"] if item["subtask_id"] not in completed), None)
        if subtask is None:
            return {"execution_log": append_log(state, "researcher: 所有子任务已完成")}
        request = {
            "request_id": f"{subtask['subtask_id']}-search",
            "subtask_id": subtask["subtask_id"],
            "tool_name": "search_documents",
            "arguments": {"query": self.models.fast_model.make_query(subtask)},
        }
        return {"active_subtask_id": subtask["subtask_id"], "tool_requests": [*state.get("tool_requests", []), request], "execution_log": append_log(state, f"researcher: 请求资料 {subtask['subtask_id']}")}

    def tool_executor(self, state: dict[str, Any]) -> dict[str, Any]:
        request = state["tool_requests"][-1]
        decision = authorize_tool_request(request, state["user_id"], self.tools.specs)
        audit = {"task_id": state["task_id"], "request_id": request["request_id"], "tool_name": request["tool_name"], "decision": decision["reason"]}
        self.audit.append(audit)
        if not decision["allowed"]:
            return {"errors": [*state.get("errors", []), {"code": decision["reason"], "message": "工具请求未获授权"}], "execution_log": append_log(state, "tool_executor: 拒绝未授权工具")}
        payload = self.tools.tools[request["tool_name"]].invoke(request["arguments"])
        result = {"request_id": request["request_id"], "subtask_id": request["subtask_id"], **payload}
        return {"tool_results": [*state.get("tool_results", []), result], "execution_log": append_log(state, f"tool_executor: 执行 {request['tool_name']}")}

    def extractor(self, state: dict[str, Any]) -> dict[str, Any]:
        subtask_id = state["active_subtask_id"]
        subtask = next(item for item in state["plan"] if item["subtask_id"] == subtask_id)
        result = next(item for item in reversed(state["tool_results"]) if item["subtask_id"] == subtask_id)
        finding = self.models.fast_model.extract_finding(subtask, result)
        return {"findings": [*state.get("findings", []), finding], "completed_subtask_ids": [*state.get("completed_subtask_ids", []), subtask_id], "execution_log": append_log(state, f"extractor: 提炼 {subtask_id}")}

    def synthesizer(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"synthesis": self.models.reasoning_model.synthesize(state["findings"]), "execution_log": append_log(state, "synthesizer: 汇总 findings")}

    def reviewer(self, state: dict[str, Any]) -> dict[str, Any]:
        verdict = self.models.reasoning_model.review(state["plan"], state["findings"])
        return {"review_status": verdict["status"], "execution_log": append_log(state, f"reviewer: {verdict['status']}")}

    def writer(self, state: dict[str, Any]) -> dict[str, Any]:
        report = self.models.writing_model.write_report(state["topic"], state["findings"], state["synthesis"])
        metadata = {"graph_version": "research-agent-v1.0.0", "state_schema_version": state["state_schema_version"], "model_roles": {"fast_model": self.models.fast_model.name, "reasoning_model": self.models.reasoning_model.name, "writing_model": self.models.writing_model.name}, "plan_version": 1, "source_count": len(state["tool_results"]), "finding_count": len(state["findings"])}
        return {"final_report": report, "report_metadata": metadata, "lifecycle_status": "completed", "execution_log": append_log(state, "writer: 生成最终报告")}
