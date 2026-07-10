from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InMemoryCheckpointStore:
    """教学实现；生产环境替换为 Postgres checkpointer。"""

    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)

    def save(self, thread_id: str, state: dict[str, Any]) -> None:
        self.snapshots[thread_id] = deepcopy(state)

    def load(self, thread_id: str) -> dict[str, Any] | None:
        state = self.snapshots.get(thread_id)
        return deepcopy(state) if state else None
