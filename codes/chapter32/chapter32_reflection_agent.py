import sys
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


QualityStatus = Literal["revise", "finish"]


class Critique(TypedDict):
    passed: bool
    issues: list[str]
    suggestions: list[str]


class ReflectionState(TypedDict, total=False):
    task: str
    draft: str
    critique: Critique
    revision_count: int
    max_revisions: int
    quality_status: QualityStatus
    final_answer: str
    execution_log: list[str]


def append_log(state: ReflectionState, message: str) -> list[str]:
    return [*state.get("execution_log", []), message]


def generate_draft(state: ReflectionState) -> dict:
    draft = (
        "Reflection Agent 会先生成答案，再检查答案是否足够好。"
        "如果不够好，就根据反馈修改。"
    )

    return {
        "draft": draft,
        "revision_count": 0,
        "max_revisions": state.get("max_revisions", 2),
        "execution_log": append_log(state, "generate_draft 生成初稿"),
    }


def critique_draft(state: ReflectionState) -> dict:
    draft = state["draft"]
    issues: list[str] = []
    suggestions: list[str] = []

    if "最大修订次数" not in draft and "max_revisions" not in draft:
        issues.append("没有说明如何避免无限反思")
        suggestions.append("补充 max_revisions 或最大修订次数")

    if "State" not in draft:
        issues.append("没有说明反馈循环需要哪些状态字段")
        suggestions.append("补充 draft、critique、revision_count、quality_status 等字段")

    if "受控" not in draft:
        issues.append("没有强调自我修正必须受控")
        suggestions.append("说明 Reflection 不是无限自我怀疑，而是有停止条件的反馈闭环")

    passed = not issues
    status: QualityStatus = "finish" if passed else "revise"

    if not passed and state.get("revision_count", 0) >= state.get("max_revisions", 2):
        status = "finish"
        suggestions.append("已达到最大修订次数，停止继续修订")

    return {
        "critique": {
            "passed": passed,
            "issues": issues,
            "suggestions": suggestions,
        },
        "quality_status": status,
        "execution_log": append_log(
            state,
            "critique_draft 通过" if passed else f"critique_draft 发现 {len(issues)} 个问题",
        ),
    }


def decide_after_critique(state: ReflectionState) -> str:
    return state["quality_status"]


def revise_draft(state: ReflectionState) -> dict:
    critique = state["critique"]
    suggestions = "；".join(critique["suggestions"])
    revised = (
        f"{state['draft']}\n\n"
        "修订补充：Reflection 与自我修正应该是受控反馈闭环。"
        "State 至少要保存 draft、critique、revision_count、max_revisions、quality_status。"
        "每一轮批判只提出明确问题，修订节点只按反馈修改，"
        "并用最大修订次数 max_revisions 防止 Agent 无限反思。\n"
        f"本轮依据的反馈：{suggestions}"
    )

    return {
        "draft": revised,
        "revision_count": state.get("revision_count", 0) + 1,
        "execution_log": append_log(state, "revise_draft 根据反馈修订答案"),
    }


def finish(state: ReflectionState) -> dict:
    critique = state.get("critique", {"passed": False, "issues": [], "suggestions": []})
    suffix = "审查通过。" if critique["passed"] else "达到停止条件，保留当前最好版本。"

    return {
        "final_answer": f"{state['draft']}\n\n最终状态：{suffix}",
        "execution_log": append_log(state, "finish 输出最终答案"),
    }


def build_reflection_agent():
    builder = StateGraph(ReflectionState)

    builder.add_node("generate_draft", generate_draft)
    builder.add_node("critique_draft", critique_draft)
    builder.add_node("revise_draft", revise_draft)
    builder.add_node("finish", finish)

    builder.add_edge(START, "generate_draft")
    builder.add_edge("generate_draft", "critique_draft")
    builder.add_conditional_edges(
        "critique_draft",
        decide_after_critique,
        {
            "revise": "revise_draft",
            "finish": "finish",
        },
    )
    builder.add_edge("revise_draft", "critique_draft")
    builder.add_edge("finish", END)

    return builder.compile()


def main() -> None:
    graph = build_reflection_agent()
    result = graph.invoke(
        {
            "task": "解释 Reflection Agent 如何发现答案不好并受控修正",
            "max_revisions": 2,
        }
    )

    print("执行日志：")
    for item in result["execution_log"]:
        print(f"- {item}")

    print("\n批判结果：")
    print(result["critique"])

    print("\n最终答案：")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
