from __future__ import annotations

from typing import Any, TypedDict


class ToolDecision(TypedDict):
    allowed: bool
    reason: str
    requires_human_approval: bool


def authorize_tool_request(
    request: dict[str, Any], user_id: str, registry: dict[str, dict[str, Any]]
) -> ToolDecision:
    """程序而非模型决定一个工具请求能否执行。"""

    del user_id
    spec = registry.get(request["tool_name"])
    if spec is None:
        return {"allowed": False, "reason": "tool_not_registered", "requires_human_approval": False}
    if spec.get("risk_level") == "high":
        return {"allowed": False, "reason": "human_approval_required", "requires_human_approval": True}
    return {"allowed": True, "reason": "allowed", "requires_human_approval": False}
