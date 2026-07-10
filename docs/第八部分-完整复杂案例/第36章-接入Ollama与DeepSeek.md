# 第36章-接入 Ollama 与 DeepSeek

## 36.1 模型分工，不等于一定要部署多个模型服务

第 35 章已经把智能研究助理的 State 和模块边界设计好了。

现在我们进入一个新的问题：

> 这些模块应该分别使用什么模型？

前面第 12-15 章讲过一种经典分工：

```text
Ollama：本地、便宜、适合轻量分类、摘要、预处理。
DeepSeek：远程、推理更强、适合规划、审查、复杂写作。
```

这个分工很有价值。

因为真实 Agent 里，不同节点的任务难度并不一样。

例如：

```text
Router 判断任务类型，通常不需要最强模型。
Planner 拆研究计划，需要更强推理。
Researcher 提取发现，可以中等模型完成。
Reviewer 检查质量，需要更强判断。
Writer 生成最终报告，需要稳定长文能力。
```

但是，如果读者只是想跑通第八部分完整案例，部署 Ollama 可能会增加门槛。

需要安装 Ollama。

需要拉取模型。

需要确认本地服务启动。

需要处理本地模型能力差异。

这些事情对学习模型分工不是没有价值，但会让完整案例的主线变重。

所以本章采用一个更低门槛的做法：

> 架构上保留“轻模型 / 强模型”的分工思想，实际示例中统一使用 DeepSeek API，通过不同模型角色和不同 prompt 约束来完成分工。

也就是说，本章标题仍然是“接入 Ollama 与 DeepSeek”，因为我们要讲清两类模型能力的架构位置。

但实际代码可以先这样落地：

```text
fast_model：DeepSeek API，用于轻量分类、格式转换、短摘要。
reasoning_model：DeepSeek API，用于规划、审查、最终报告。
```

以后如果你愿意部署 Ollama，只需要把 `fast_model` 的实现替换成本地模型。

图结构、State 契约和节点边界都不需要大改。

这就是本章的核心：

> 模型供应方可以先简化，但模型角色分工不能丢。

## 36.2 本章要解决的问题

本章采用“模型分工法”。

核心问题是：

> 哪些任务适合本地轻模型，哪些必须交给强推理模型？

但结合本书完整案例，我们会把问题稍微改写成：

> 哪些节点只需要轻量模型角色，哪些节点需要强推理模型角色？如果暂时不用 Ollama，如何用 DeepSeek API 保持这种分工？

本章会回答四个问题。

第一，智能研究助理里每个模块需要什么模型能力？

第二，为什么不能所有节点都直接使用同一个 `llm`？

第三，如何设计 `fast_model`、`reasoning_model`、`writing_model` 这类模型角色？

第四，如何把 DeepSeek 接入第 35 章的 Router、Planner、Researcher、Reviewer、Writer，同时不破坏 State 边界？

读完本章，读者应该建立一个判断：

> 模型不是 Agent 架构的中心，模型是节点能力的实现。好的 Agent 架构应该允许模型被替换，而不改变任务生命周期。

## 36.3 先看错误写法：所有节点共用一个 llm

很多项目一开始会这样写：

```python
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


def router(state):
    response = llm.invoke(...)
    return {"task_type": ...}


def planner(state):
    response = llm.invoke(...)
    return {"research_plan": ...}


def reviewer(state):
    response = llm.invoke(...)
    return {"review_feedback": ...}


def writer(state):
    response = llm.invoke(...)
    return {"final_report": ...}
```

这段代码能跑。

但它有几个问题。

第一，节点能力差异被抹平。

Router、Planner、Reviewer、Writer 都叫同一个 `llm`，读者看不出哪个节点需要快速响应，哪个节点需要复杂推理。

第二，成本和延迟不可控。

简单分类和复杂报告都走同一个模型、同一个 prompt、同一种调用策略。

以后想优化成本时，很难只替换某些节点。

第三，测试不清楚。

如果最终报告质量不好，你不知道是 Planner 没拆好，Reviewer 没审好，还是 Writer 没写好。

第四，未来替换模型困难。

如果以后把 Router 换成本地 Ollama，或者把 Writer 换成更适合长文的模型，就要到每个节点里改。

所以本章不建议用一个泛泛的 `llm` 贯穿所有节点。

更好的方式是按模型角色命名。

```text
fast_model
reasoning_model
writing_model
```

这些名字表达的不是供应商，而是节点需要的能力。

## 36.4 模型角色，而不是模型品牌

模型分工最重要的不是：

```text
这个节点用 Ollama，那个节点用 DeepSeek。
```

而是：

```text
这个节点需要什么能力？
```

本案例先定义三类模型角色。

| 模型角色 | 适合任务 | 设计目标 |
| --- | --- | --- |
| `fast_model` | 路由、短分类、格式清洗、简单摘要 | 快、便宜、输出短 |
| `reasoning_model` | 规划、架构判断、审查、复杂推理 | 质量高、能处理多约束 |
| `writing_model` | 最终报告、长文组织、结构化表达 | 输出稳定、结构清楚 |

如果你部署了 Ollama，可以这样实现：

```text
fast_model -> Ollama 本地模型
reasoning_model -> DeepSeek
writing_model -> DeepSeek
```

如果你不想部署 Ollama，可以这样实现：

```text
fast_model -> DeepSeek API，使用更短 prompt、更低温度、更严格输出格式
reasoning_model -> DeepSeek API，使用复杂推理 prompt
writing_model -> DeepSeek API，使用长文写作 prompt
```

也就是说：

```text
分工是架构层概念。
模型供应方是实现层选择。
```

先把分工想清楚，再选择具体模型，系统才不会被某个模型品牌绑死。

## 36.5 哪些节点适合轻量模型角色

轻量模型角色不代表模型一定很弱。

它代表这个节点不应该消耗太多推理预算。

在智能研究助理里，适合 `fast_model` 的节点包括：

| 节点 | 为什么适合轻量模型角色 |
| --- | --- |
| `router` | 只需要判断任务类型和给出短理由 |
| `memory_loader` 的偏好摘要 | 只是把长期偏好整理成短上下文 |
| `researcher` 的简单 query 生成 | 多数情况下是把子任务改写成检索词 |
| `tool_executor` 的结果清洗 | 如果需要模型参与，也只是短文本整理 |
| `memory_writer` 的偏好提取 | 从用户反馈中提取简短偏好 |

这些任务的共同特点是：

```text
输入短。
输出短。
结构固定。
错误可兜底。
不需要复杂多步推理。
```

例如 Router 只需要输出：

```json
{
  "task_type": "architecture_research",
  "route_reason": "用户关注模块化、可扩展性和生产化"
}
```

这种任务不应该每次都使用最重的推理链。

如果用 DeepSeek API 实现 `fast_model`，也要让它保持轻：

```text
prompt 短。
只返回 JSON。
temperature 低。
失败时用规则兜底。
```

## 36.6 哪些节点需要强推理模型角色

强推理模型角色适合处理多约束、多步骤、质量要求高的节点。

在智能研究助理里，最适合 `reasoning_model` 的节点包括：

| 节点 | 为什么需要强推理 |
| --- | --- |
| `planner` | 要把大主题拆成合理子任务，并考虑用户偏好 |
| `reviewer` | 要发现遗漏、判断证据是否支撑结论 |
| `state_aggregator` | 如果聚合逻辑复杂，需要把多条发现合成架构判断 |
| `writer` 的报告结构规划 | 要把材料组织成有说服力的长文结构 |

这些任务的共同特点是：

```text
要同时考虑多个约束。
输出会影响后续流程。
错误会让整个任务偏航。
需要判断“够不够好”，而不是只做格式转换。
```

例如 Reviewer 不能只说：

```text
看起来不错。
```

它要判断：

```text
计划是否执行完整。
发现是否覆盖用户重点。
结论是否有证据支撑。
是否缺少未来扩展分析。
下一步应该写报告、修订计划，还是补充研究。
```

这类任务应该交给更强的推理模型角色。

## 36.7 最终分工表

把第 35 章的模块和本章模型角色对应起来，可以得到这张表。

| 模块 | 推荐模型角色 | 本章实际实现 | 原因 |
| --- | --- | --- | --- |
| `memory_loader` | 无模型或 `fast_model` | 先用固定偏好 | 本章重点不是长期记忆 |
| `router` | `fast_model` | DeepSeek API 轻量调用 | 只做分类和短理由 |
| `planner` | `reasoning_model` | DeepSeek API 强推理调用 | 需要拆解复杂主题 |
| `human_approval` | 不需要模型 | 第 37 章用 interrupt | 人类决策，不交给模型 |
| `dispatch_research` | 不需要模型 | 规则函数 | 只选择下一个子任务 |
| `researcher` | `fast_model` 或 `reasoning_model` | 先用轻量调用生成 query / finding | 子任务解释可逐步增强 |
| `tool_executor` | 不需要模型 | 工具或模拟工具 | 工具执行应确定 |
| `state_aggregator` | 规则或 `reasoning_model` | 简化可用规则，复杂可用 DeepSeek | 聚合质量决定审查效果 |
| `reviewer` | `reasoning_model` | DeepSeek API 强推理调用 | 质量闸门，需要严谨判断 |
| `writer` | `writing_model` | DeepSeek API 写作调用 | 最终报告需要结构稳定 |
| `memory_writer` | `fast_model` | 可选轻量调用 | 提取偏好，输出短 |

这张表有一个隐藏原则：

> 不要因为模型方便，就让模型接管所有逻辑。

例如 `dispatch_research` 完全可以用规则。

`tool_executor` 也不应该让模型“假装调用工具”。

模型应该用在理解、判断、总结和表达上。

确定性控制和工具执行仍然应该由程序完成。

## 36.8 DeepSeek-only 的模型工厂

本章实际示例采用 DeepSeek API。

为了保留模型分工，我们不直接在节点里创建模型，而是写一个模型工厂。

```python
from dataclasses import dataclass
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


@dataclass(frozen=True)
class ModelConfig:
    fast_model_name: str = "deepseek-chat"
    reasoning_model_name: str = "deepseek-reasoner"
    writing_model_name: str = "deepseek-chat"
    temperature: float = 0


def load_model_config() -> ModelConfig:
    load_dotenv()
    return ModelConfig(
        fast_model_name=os.getenv("DEEPSEEK_FAST_MODEL", "deepseek-chat"),
        reasoning_model_name=os.getenv("DEEPSEEK_REASONING_MODEL", "deepseek-reasoner"),
        writing_model_name=os.getenv("DEEPSEEK_WRITING_MODEL", "deepseek-chat"),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0")),
    )


def build_deepseek_model(model_name: str, temperature: float = 0):
    return ChatDeepSeek(
        model=model_name,
        temperature=temperature,
    )


@dataclass(frozen=True)
class ModelSet:
    fast_model: object
    reasoning_model: object
    writing_model: object


def build_model_set(config: ModelConfig | None = None) -> ModelSet:
    config = config or load_model_config()

    return ModelSet(
        fast_model=build_deepseek_model(config.fast_model_name, config.temperature),
        reasoning_model=build_deepseek_model(config.reasoning_model_name, config.temperature),
        writing_model=build_deepseek_model(config.writing_model_name, config.temperature),
    )
```

这里有几个设计点。

第一，节点不直接读环境变量。

环境变量集中在 `load_model_config()`。

第二，模型按角色暴露。

节点看到的是：

```text
fast_model
reasoning_model
writing_model
```

而不是到处散落的 `ChatDeepSeek(...)`。

第三，模型名可以通过环境变量替换。

例如：

```text
DEEPSEEK_FAST_MODEL=deepseek-chat
DEEPSEEK_REASONING_MODEL=deepseek-reasoner
DEEPSEEK_WRITING_MODEL=deepseek-chat
```

如果你的账号可用模型名不同，只需要改环境变量。

图和节点边界不用变。

## 36.9 如果以后要接回 Ollama

虽然本章实际例子使用 DeepSeek-only，但架构上仍然保留 Ollama 的位置。

以后如果你想把轻量任务移到本地，只需要改模型工厂。

例如：

```python
from langchain_ollama import ChatOllama


def build_ollama_fast_model():
    return ChatOllama(
        model=os.getenv("OLLAMA_FAST_MODEL", "qwen3:4b"),
        temperature=0,
    )


def build_model_set(config: ModelConfig | None = None) -> ModelSet:
    config = config or load_model_config()

    use_ollama_fast = os.getenv("USE_OLLAMA_FAST_MODEL", "false") == "true"

    fast_model = (
        build_ollama_fast_model()
        if use_ollama_fast
        else build_deepseek_model(config.fast_model_name, config.temperature)
    )

    return ModelSet(
        fast_model=fast_model,
        reasoning_model=build_deepseek_model(config.reasoning_model_name, config.temperature),
        writing_model=build_deepseek_model(config.writing_model_name, config.temperature),
    )
```

这段代码说明一个重点：

> Ollama 是 fast_model 的一种实现，不是 Router 节点的一部分。

Router 依赖的是“轻量模型角色”。

至于这个角色由 DeepSeek API 还是 Ollama 提供，是模型工厂的事情。

这就是模型层和图结构解耦。

## 36.10 改造 Router：用 fast_model 做轻量分类

第 35 章的 Router 用规则判断。

本章可以把它改成模型版本。

```python
import json


def make_router_node(fast_model):
    def router(state: ResearchState) -> dict:
        prompt = f"""
你是智能研究助理的路由节点。

请判断用户主题属于哪种研究类型，只返回 JSON。

可选 task_type：
- concept_research：解释概念或总结基础知识
- architecture_research：研究架构、模块化、可扩展性、生产化
- comparison_research：比较多个框架、方案或技术路线
- implementation_plan：生成实施方案、落地路线或项目计划

返回格式：
{{"task_type": "...", "route_reason": "..."}}

用户主题：
{state["topic"]}

用户偏好：
{state.get("user_preferences", {})}
"""
        response = fast_model.invoke(prompt)
        content = response.content.strip()

        try:
            parsed = json.loads(content)
            task_type = parsed["task_type"]
            route_reason = parsed["route_reason"]
        except Exception:
            task_type = "architecture_research"
            route_reason = "模型输出无法解析，按架构研究兜底"

        return {
            "task_type": task_type,
            "route_reason": route_reason,
            "execution_log": ["router 使用 fast_model 判断任务类型"],
        }

    return router
```

这个节点仍然遵守第 35 章的状态契约。

它读取：

```text
topic
user_preferences
```

写入：

```text
task_type
route_reason
execution_log
```

模型变了，但模块边界没有变。

这才是正确的接入方式。

## 36.11 改造 Planner：用 reasoning_model 生成计划

Planner 需要更强推理。

因为它要把大主题拆成结构化子任务。

```python
def make_planner_node(reasoning_model):
    def planner(state: ResearchState) -> dict:
        version = state.get("plan_version", 0) + 1

        prompt = f"""
你是智能研究助理的 Planner。

请根据用户主题、任务类型和偏好，生成一个结构化研究计划。

要求：
1. 子任务数量控制在 3-5 个。
2. 每个子任务必须有 id、question、expected_output、priority。
3. 不要生成最终报告，只生成计划。
4. 如果有 review_feedback 或 approval_feedback，要体现修订。

只返回 JSON：
{{
  "goal": "...",
  "subtasks": [
    {{"id": "...", "question": "...", "expected_output": "...", "priority": 1}}
  ]
}}

用户主题：{state["topic"]}
任务类型：{state["task_type"]}
用户偏好：{state.get("user_preferences", {})}
审批反馈：{state.get("approval_feedback", "")}
审查反馈：{state.get("review_feedback", {})}
"""
        response = reasoning_model.invoke(prompt)
        parsed = json.loads(response.content)

        plan: ResearchPlan = {
            "plan_id": f"plan-v{version}",
            "goal": parsed["goal"],
            "version": version,
            "subtasks": parsed["subtasks"],
        }

        return {
            "research_plan": plan,
            "plan_version": version,
            "subtasks": parsed["subtasks"],
            "approval_status": "pending",
            "execution_log": [f"planner 使用 reasoning_model 生成第 {version} 版计划"],
        }

    return planner
```

这里要注意一条边界：

> Planner 只生成计划，不执行研究。

即使用了强模型，也不能让它顺手生成最终报告。

强模型更容易“热心过头”。

所以 prompt 里要明确：

```text
不要生成最终报告，只生成计划。
```

模型能力越强，边界越要写清楚。

## 36.12 改造 Researcher：轻量生成工具请求，解释工具结果

Researcher 可以有两种实现。

简单版本用规则。

增强版本用 `fast_model` 或 `reasoning_model`。

本章建议先用 `fast_model`。

因为 Researcher 的第一步通常只是把子任务改写成工具请求。

```python
def make_researcher_node(fast_model):
    def researcher(state: ResearchState) -> dict:
        subtask = state["active_subtask"]
        subtask_id = subtask["id"]

        related_results = [
            result
            for result in state.get("tool_results", [])
            if result["subtask_id"] == subtask_id and result["status"] == "success"
        ]

        if not related_results:
            prompt = f"""
请把研究子任务改写成一个适合检索资料的查询词。
只返回一句查询词。

子任务问题：{subtask["question"]}
期望产出：{subtask["expected_output"]}
"""
            response = fast_model.invoke(prompt)

            request: ToolRequest = {
                "id": f"tool-{subtask_id}",
                "subtask_id": subtask_id,
                "tool_name": "search_docs",
                "query": response.content.strip(),
                "purpose": subtask["expected_output"],
            }

            return {
                "tool_requests": [request],
                "execution_log": [f"researcher 为 {subtask_id} 生成工具请求"],
            }

        prompt = f"""
请根据工具结果，为当前子任务提炼一个研究发现。

子任务：{subtask["question"]}
工具结果：{related_results[-1]["content"]}

只返回一段简洁发现。
"""
        response = fast_model.invoke(prompt)

        finding: Finding = {
            "subtask_id": subtask_id,
            "summary": response.content.strip(),
            "evidence_ids": [related_results[-1].get("source_id", f"source-{subtask_id}")],
        }

        return {
            "findings": [finding],
            "completed_subtasks": [subtask_id],
            "execution_log": [f"researcher 完成子任务发现：{subtask_id}"],
        }

    return researcher
```

这个节点仍然不直接调用工具。

它只是生成 `tool_requests`，或者把 `tool_results` 解释成 `findings`。

这保证了 Researcher 和 Tool Executor 的边界。

## 36.13 Tool Executor 不需要模型

Tool Executor 不是模型节点。

它应该尽量保持确定性。

在本案例里，它可以继续使用第 35 章的模拟工具，或者改成真实搜索、文件读取、知识库检索。

但它不应该让模型假装工具。

错误写法是：

```python
def tool_executor(state):
    response = reasoning_model.invoke("请你模拟搜索资料...")
    return {"tool_results": ...}
```

这会把工具能力和模型生成混在一起。

如果本章暂时没有真实工具，就诚实写成模拟工具：

```python
def tool_executor(state: ResearchState) -> dict:
    ...
    "content": f"模拟资料：{request['query']} 的关键资料摘要。"
```

然后在文字里说明：

```text
这里是教学模拟工具，真实项目应替换成搜索、文件读取或知识库检索。
```

这样读者不会误以为模型输出就是资料来源。

工具是工具。

模型是模型。

这条边界必须守住。

## 36.14 改造 Reviewer：用 reasoning_model 做质量闸门

Reviewer 是最应该使用强推理模型的节点之一。

它要检查：

```text
研究计划是否完成。
聚合结论是否覆盖用户重点。
证据是否支撑结论。
是否应该继续补充研究。
```

```python
def make_reviewer_node(reasoning_model):
    def reviewer(state: ResearchState) -> dict:
        prompt = f"""
你是智能研究助理的 Reviewer。

请审查当前研究结果是否足以进入最终报告写作。

审查标准：
1. 是否覆盖研究计划中的关键子任务。
2. synthesis 是否有清晰 thesis。
3. key_points 是否足够支撑报告。
4. 是否存在明显 gaps。
5. 是否符合用户偏好。

只返回 JSON：
{{
  "passed": true,
  "issues": [],
  "suggestions": [],
  "next_action": "write_report|revise_plan|more_research"
}}

用户主题：{state["topic"]}
研究计划：{state["research_plan"]}
聚合结论：{state["synthesis"]}
证据映射：{state.get("evidence_map", {})}
用户偏好：{state.get("user_preferences", {})}
"""
        response = reasoning_model.invoke(prompt)
        feedback = json.loads(response.content)

        quality_status: QualityStatus = (
            "approved" if feedback["passed"] else "needs_revision"
        )

        return {
            "review_feedback": feedback,
            "quality_status": quality_status,
            "next_action": feedback["next_action"],
            "execution_log": ["reviewer 使用 reasoning_model 完成质量审查"],
        }

    return reviewer
```

这里最重要的是输出结构。

Reviewer 不应该只返回自然语言。

它要返回：

```text
passed
issues
suggestions
next_action
```

因为这些字段会驱动后续条件边。

模型输出必须服务图控制流。

## 36.15 改造 Writer：用 writing_model 生成最终报告

Writer 需要长文组织能力。

它不一定需要最强推理，但需要稳定表达和结构控制。

所以可以单独给它一个 `writing_model`。

```python
def make_writer_node(writing_model):
    def writer(state: ResearchState) -> dict:
        prompt = f"""
你是智能研究助理的 Writer。

请基于已经审查通过的研究结果，生成一份结构化 Markdown 报告。

重要约束：
1. 只能基于 synthesis、evidence_map、research_plan 写作。
2. 不要发明新的研究发现。
3. 如果 review_feedback 中仍有风险，要在报告中说明限制。
4. 写作风格遵守 user_preferences。

报告建议结构：
- 标题
- 核心结论
- 研究背景
- 关键发现
- 架构分析
- 实践建议
- 风险与限制
- 下一步方向

用户主题：{state["topic"]}
研究计划：{state["research_plan"]}
聚合结论：{state["synthesis"]}
证据映射：{state.get("evidence_map", {})}
审查反馈：{state.get("review_feedback", {})}
用户偏好：{state.get("user_preferences", {})}
"""
        response = writing_model.invoke(prompt)

        metadata: ReportMetadata = {
            "plan_version": state["research_plan"]["version"],
            "source_count": len(state.get("tool_results", [])),
            "finding_count": len(state.get("findings", [])),
        }

        return {
            "final_report": response.content,
            "report_metadata": metadata,
            "execution_log": ["writer 使用 writing_model 生成最终报告"],
        }

    return writer
```

Writer 的 prompt 里必须强调：

```text
不要发明新的研究发现。
```

因为最终报告最容易出现的问题就是“写得顺，但依据不够”。

这也是为什么 Writer 必须在 Reviewer 之后执行。

## 36.16 图组装：注入模型，而不是在节点里创建模型

现在可以把模型注入到图里。

```python
def build_research_graph_with_models(models: ModelSet):
    builder = StateGraph(ResearchState)

    builder.add_node("memory_loader", memory_loader)
    builder.add_node("router", make_router_node(models.fast_model))
    builder.add_node("planner", make_planner_node(models.reasoning_model))
    builder.add_node("human_approval", human_approval)
    builder.add_node("dispatch_research", dispatch_research)
    builder.add_node("researcher", make_researcher_node(models.fast_model))
    builder.add_node("tool_executor", tool_executor)
    builder.add_node("state_aggregator", state_aggregator)
    builder.add_node("reviewer", make_reviewer_node(models.reasoning_model))
    builder.add_node("writer", make_writer_node(models.writing_model))
    builder.add_node("memory_writer", memory_writer)

    # 边的连接沿用第 35 章。
    ...

    return builder.compile()
```

注意这里的关键变化：

```text
节点通过工厂函数接收模型。
图负责组装节点。
模型创建由 build_model_set 负责。
```

不要在 `planner` 节点内部写：

```python
model = ChatDeepSeek(...)
```

那样会让节点难测试，也让模型替换变困难。

更好的依赖方向是：

```text
config -> model factory -> graph builder -> node factory -> node
```

这条方向清楚，项目就不会到处散落 API Key、模型名和 provider 判断。

## 36.17 DeepSeek API 环境配置

本章示例需要 DeepSeek API Key。

可以在 `.env` 中配置：

```text
DEEPSEEK_API_KEY=你的 API Key
DEEPSEEK_FAST_MODEL=deepseek-chat
DEEPSEEK_REASONING_MODEL=deepseek-reasoner
DEEPSEEK_WRITING_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0
```

如果你的 DeepSeek 账号或 SDK 使用的模型名不同，以实际可用模型名为准。

本书建议不要把模型名写死在节点里。

原因很简单：

```text
模型名称会变化。
模型能力会变化。
项目不同环境可能使用不同模型。
```

把模型名放到环境变量或配置文件里，才能让章节代码更容易迁移。

## 36.18 DeepSeek-only 方案和 Ollama + DeepSeek 方案对比

本章采用 DeepSeek-only 实现，但读者要知道它和 Ollama + DeepSeek 的差异。

| 方案 | 优点 | 代价 | 适合阶段 |
| --- | --- | --- | --- |
| DeepSeek-only | 最容易跑通，不用本地部署 | 所有模型调用都依赖远程 API | 学习完整案例、快速验证 |
| Ollama + DeepSeek | 本地轻任务可控，隐私和成本更好 | 需要部署本地模型，调试更多 | 长期项目、隐私敏感场景 |
| 规则 + DeepSeek | 成本低，确定性强 | 规则维护成本上升 | 路由规则清楚的业务 |

本章选择 DeepSeek-only，是为了降低完整案例的部署门槛。

但架构上仍然保留 `fast_model` 这个角色。

以后要迁移到 Ollama + DeepSeek，只需要替换模型工厂。

节点和 State 不需要重写。

这就是模型分工法真正要保护的东西：

> 模型可以换，角色不乱；供应商可以换，状态契约不乱。

## 36.19 常见错误与排查

### 错误一：把 DeepSeek 调用写死在每个节点里

现象：

```text
router、planner、reviewer、writer 都各自 ChatDeepSeek(...)
```

问题：

```text
配置散落，测试困难，模型替换困难。
```

建议：

```text
集中到 ModelSet 和 build_model_set。
```

### 错误二：所有节点都用 reasoning_model

现象：

```text
分类、query 生成、偏好提取都走强推理模型。
```

问题：

```text
成本和延迟上升，而且分工意识消失。
```

建议：

```text
轻任务走 fast_model，复杂规划和审查走 reasoning_model。
```

### 错误三：强模型越过模块边界

现象：

```text
Planner 直接生成报告。
Reviewer 直接改 final_report。
Writer 发明新的 findings。
```

问题：

```text
模型能力太强，反而把架构边界冲烂。
```

建议：

```text
prompt 中明确节点职责，只允许写本节点负责的 State 字段。
```

### 错误四：模型输出 JSON 不稳定

现象：

```text
router 或 reviewer 返回自然语言，json.loads 失败。
```

问题：

```text
模型输出没有被严格约束，也没有兜底。
```

建议：

```text
prompt 明确“只返回 JSON”，解析失败时使用规则兜底或进入人工确认。
```

### 错误五：把工具结果交给模型伪造

现象：

```text
没有真实工具，直接让 DeepSeek “模拟搜索结果”。
```

问题：

```text
读者会混淆模型生成和资料来源，研究报告不可追溯。
```

建议：

```text
教学阶段可以使用明确标注的模拟工具；真实项目必须接入真实工具或知识库。
```

## 36.20 本章小结

本章把第 35 章的状态和模块骨架接入了模型能力。

我们没有简单地说：

```text
所有节点都用 DeepSeek。
```

而是先建立模型分工：

```text
fast_model：轻量分类、短摘要、query 生成、偏好提取。
reasoning_model：计划生成、质量审查、复杂架构判断。
writing_model：最终报告生成和长文组织。
```

然后考虑到实际运行门槛，本章采用 DeepSeek API 实现所有模型角色。

这样读者不用先部署 Ollama，也能跑通完整案例。

但架构上仍然保留了 Ollama 的位置：

```text
Ollama 可以作为 fast_model 的一种实现。
DeepSeek 可以作为 reasoning_model 和 writing_model 的实现。
```

本章最重要的结论是：

> 模型分工是一种架构能力，不是模型供应商清单。

只要 `ResearchState` 稳定、模块读写边界稳定、模型通过工厂注入，未来就可以自由替换：

```text
DeepSeek-only。
Ollama + DeepSeek。
规则 + DeepSeek。
更多模型供应商。
```

下一章会进入第 37 章：加入持久化、恢复与人工审批。

到那时，模型能力已经接入，状态主线已经稳定，我们要继续解决长任务的核心问题：

```text
计划确认时如何暂停？
用户确认后如何恢复？
执行失败后如何从中间继续？
```

