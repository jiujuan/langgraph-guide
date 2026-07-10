from __future__ import annotations

from typing import Any, Literal, TypedDict


ApprovalStatus = Literal["pending", "approved", "rejected"]
LifecycleStatus = Literal["queued", "awaiting_approval", "running", "completed", "failed"]


class ResearchState(TypedDict, total=False):
    """图内共享 State：只放节点协作所需的数据。"""

    state_schema_version: int
    task_id: str
    thread_id: str
    user_id: str
    topic: str
    lifecycle_status: LifecycleStatus
    approval_status: ApprovalStatus
    plan: list[dict[str, Any]]
    active_subtask_id: str
    completed_subtask_ids: list[str]
    tool_requests: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    synthesis: str
    review_status: Literal["pass", "more_research"]
    final_report: str
    report_metadata: dict[str, Any]
    execution_log: list[str]
    errors: list[dict[str, str]]
    warnings: list[str]


def migrate_state(state: ResearchState | dict[str, Any]) -> ResearchState:
    """把示例 v1 的 approvalState 字段迁移到 v2 的统一契约。"""

    migrated: ResearchState = dict(state)
    if migrated.get("state_schema_version", 1) < 2:
        legacy_status = migrated.pop("approvalState", None)
        if legacy_status is not None:
            migrated["approval_status"] = legacy_status
        migrated["state_schema_version"] = 2
    migrated.setdefault("state_schema_version", 2)
    migrated.setdefault("completed_subtask_ids", [])
    migrated.setdefault("tool_requests", [])
    migrated.setdefault("tool_results", [])
    migrated.setdefault("findings", [])
    migrated.setdefault("execution_log", [])
    migrated.setdefault("errors", [])
    migrated.setdefault("warnings", [])
    return migrated
