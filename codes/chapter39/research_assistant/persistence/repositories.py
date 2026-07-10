from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InMemoryTaskRepository:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def create(self, task: dict[str, Any]) -> dict[str, Any]:
        self.records[task["task_id"]] = deepcopy(task)
        return self.get(task["task_id"])

    def get(self, task_id: str) -> dict[str, Any]:
        if task_id not in self.records:
            raise KeyError(f"task_not_found:{task_id}")
        return deepcopy(self.records[task_id])

    def save(self, task: dict[str, Any]) -> dict[str, Any]:
        self.records[task["task_id"]] = deepcopy(task)
        return self.get(task["task_id"])


@dataclass
class InMemoryAuditRepository:
    records: list[dict[str, Any]] = field(default_factory=list)

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(deepcopy(record))

    def for_task(self, task_id: str) -> list[dict[str, Any]]:
        return [deepcopy(record) for record in self.records if record["task_id"] == task_id]
