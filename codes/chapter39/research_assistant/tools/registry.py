from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .search_docs import SearchDocumentsTool


class Tool(Protocol):
    name: str

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class ToolRegistry:
    specs: dict[str, dict[str, Any]]
    tools: dict[str, Tool]

    @classmethod
    def demo(cls) -> "ToolRegistry":
        search = SearchDocumentsTool()
        return cls(
            specs={"search_documents": {"risk_level": "low", "description": "检索已授权的公开文档"}},
            tools={search.name: search},
        )
