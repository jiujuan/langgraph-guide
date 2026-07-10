from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..persistence.checkpointer import InMemoryCheckpointStore
from ..persistence.repositories import InMemoryTaskRepository


class ResearchWorker:
    def __init__(self, graph: Any, tasks: InMemoryTaskRepository, checkpoints: InMemoryCheckpointStore) -> None:
        self.graph = graph
        self.tasks = tasks
        self.checkpoints = checkpoints

    def run(self, task_id: str) -> dict[str, Any]:
        task = self.tasks.get(task_id)
        if task["approval_status"] != "approved":
            raise ValueError("task_requires_approval")
        state = deepcopy(task["state"])
        state["lifecycle_status"] = "running"
        result = self.graph.invoke(state)
        self.checkpoints.save(task["thread_id"], result)
        task.update({"state": result, "lifecycle_status": result["lifecycle_status"], "final_report": result.get("final_report"), "report_metadata": result.get("report_metadata")})
        return self.tasks.save(task)
