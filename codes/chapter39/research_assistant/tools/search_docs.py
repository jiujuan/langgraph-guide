from __future__ import annotations

from typing import Any


class SearchDocumentsTool:
    """可预测的文档搜索适配器，便于本章离线运行与测试。"""

    name = "search_documents"

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments["query"]
        return {
            "sources": [
                {
                    "title": "第 39 章生产演进实践",
                    "summary": f"围绕“{query}”的实践表明：通过职责单一的节点、版本化 State 和可注入依赖，可以在新增模型或工具时保持系统可维护。",
                }
            ]
        }
