from __future__ import annotations

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class RuntimeConfig:
    """配置集中在运行时边界；节点不直接读取环境变量。"""

    graph_version: str = "research-agent-v1.0.0"
    max_model_calls: int = 20
    deepseek_api_key: str | None = None

    @classmethod
    def from_environment(cls) -> "RuntimeConfig":
        return cls(deepseek_api_key=getenv("DEEPSEEK_API_KEY"))
