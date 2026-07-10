from __future__ import annotations

from copy import deepcopy
from uuid import uuid4
from typing import Any

from ..graph.builder import build_research_graph
from ..graph.state import migrate_state
from ..models.factory import ModelBundle
from ..nodes.workflow import WorkflowNodes
from ..persistence.checkpointer import InMemoryCheckpointStore
from ..persistence.repositories import InMemoryAuditRepository, InMemoryTaskRepository
from ..runtime.worker import ResearchWorker
from ..tools.registry import ToolRegistry


class ResearchService:
    def __init__(self, tasks: InMemoryTaskRepository, audit: InMemoryAuditRepository, checkpoints: InMemoryCheckpointStore, graph: Any) -> None:
        self.tasks = tasks
        self.audit = audit
        self.checkpoints = checkpoints
        self.worker = ResearchWorker(graph, tasks, checkpoints)
        self.graph = graph

    @classmethod
    def demo(cls) -> "ResearchService":
        tasks, audit, checkpoints = InMemoryTaskRepository(), InMemoryAuditRepository(), InMemoryCheckpointStore()
        graph = build_research_graph(WorkflowNodes(ModelBundle.demo(), ToolRegistry.demo(), audit))
        return cls(tasks, audit, checkpoints, graph)

    def create_task(self, user_id: str, topic: str) -> dict[str, Any]:
        task_id, thread_id = f"task-{uuid4().hex[:8]}", f"thread-{uuid4().hex[:8]}"
        initial = migrate_state({"task_id": task_id, "thread_id": thread_id, "user_id": user_id, "topic": topic, "approval_status": "pending", "lifecycle_status": "queued"})
        state = self.graph.invoke(initial)
        task = {"task_id": task_id, "thread_id": thread_id, "user_id": user_id, "topic": topic, "approval_status": state["approval_status"], "lifecycle_status": state["lifecycle_status"], "state": state, "final_report": None, "report_metadata": None}
        self.checkpoints.save(thread_id, state)
        return self.tasks.create(task)

    def approve_plan(self, task_id: str, reviewer_id: str) -> dict[str, Any]:
        task = self.tasks.get(task_id)
        task["approval_status"] = "approved"
        task["lifecycle_status"] = "queued"
        task["state"] = {**deepcopy(task["state"]), "approval_status": "approved", "lifecycle_status": "queued"}
        self.audit.append({"task_id": task_id, "event": "plan_approved", "reviewer_id": reviewer_id})
        return self.tasks.save(task)

    def run_task(self, task_id: str) -> dict[str, Any]:
        return self.worker.run(task_id)

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self.tasks.get(task_id)

    def audit_records(self, task_id: str) -> list[dict[str, Any]]:
        return self.audit.for_task(task_id)
