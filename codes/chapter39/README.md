# 第 39 章：从示例到生产

这是第 39 章对应的可运行工程示例。它保留了 LangGraph 图的核心流程，同时把教学版单文件拆成明确的运行时边界：State、节点、模型、工具、任务服务和持久化存储各自独立。

## 运行

在仓库根目录执行：

```powershell
python codes\chapter39\run_demo.py
python -m unittest codes\chapter39\tests\test_chapter39.py
```

示例默认使用确定性的 `Fake*Model` 和离线文档搜索工具，不需要配置 `DEEPSEEK_API_KEY`。真实接入时，只需要在 `models/` 中实现同一角色方法，再通过 `ModelBundle` 注入；节点和图的契约不变。

## 一次完整任务

1. `ResearchService.create_task()` 创建业务任务，并调用图完成 Router、Planner 和 Approval 节点。
2. 图以 `awaiting_approval` 结束，State 被保存到任务记录与 checkpoint。
3. `approve_plan()` 写入审批事件并保持同一个 `thread_id`。
4. `ResearchWorker.run()` 恢复 State，完成 Researcher -> Tool Executor -> Extractor 循环。
5. Synthesizer、Reviewer、Writer 生成最终报告、版本元数据和审计可追溯信息。

## 目录与章节映射

| 目录 | 负责内容 | 对应生产演进 |
| --- | --- | --- |
| `graph/` | State 契约、迁移和 LangGraph 组装 | State 版本化、图不依赖具体模型 |
| `nodes/` | Router、Planner、Approval、Researcher、Reviewer、Writer | 节点职责单一、只写自己的 State 字段 |
| `models/` | fast/reasoning/writing 角色 | 模型可替换，节点契约稳定 |
| `tools/` | 注册表、授权、离线搜索工具 | 程序决定工具是否执行 |
| `persistence/` | 任务记录、审计记录、checkpoint | 业务任务与图内 State 分开保存 |
| `api/` 与 `runtime/` | 任务服务与 worker | HTTP/API 可只提交任务，耗时图在后台恢复 |
| `evals/` | 报告的规则评测 | 测试保护结构，评测保护输出质量 |

## 教学实现与生产替换点

- `InMemoryTaskRepository`、`InMemoryAuditRepository` 和 `InMemoryCheckpointStore` 便于本地理解。生产环境应分别替换为业务数据库、审计存储和 Postgres checkpointer。
- `ResearchService` 是 API 层的可复用服务对象，刻意没有绑定 FastAPI。实际项目可在 HTTP handler 中调用它，并将 `run_task()` 投递给 Celery、RQ、Temporal 或企业已有队列。
- `RuntimeConfig` 集中读取环境变量。真实模型适配器只能在模型工厂读取密钥，节点不得散落读取 `DEEPSEEK_API_KEY`。
- 规则评测只检查报告结构与元数据；业务质量仍要维护真实样例集，并增加人工或模型评审。
