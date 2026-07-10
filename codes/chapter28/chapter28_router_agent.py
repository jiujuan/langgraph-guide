import sys
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


Route = Literal["writing", "qa", "code_analysis", "search_summary"]


class RouterState(TypedDict, total=False):
    task: str
    route: Route
    route_reason: str
    answer: str
    outline: list[str]
    draft: str
    code_notes: list[str]
    search_query: str
    search_summary: str
    execution_path: list[str]


def append_path(state: RouterState, node_name: str) -> list[str]:
    return [*state.get("execution_path", []), node_name]


def classify_task(task: str) -> tuple[Route, str]:
    normalized = task.lower()

    if any(keyword in normalized for keyword in ["写", "文章", "草稿", "文案", "outline"]):
        return "writing", "任务要求生成或组织文本"

    if any(keyword in normalized for keyword in ["代码", "函数", "bug", "重构", "python"]):
        return "code_analysis", "任务要求分析代码或工程问题"

    if any(keyword in normalized for keyword in ["搜索", "资料", "新闻", "来源", "总结"]):
        return "search_summary", "任务需要检索资料并汇总"

    return "qa", "任务可以直接问答"


def route_task(state: RouterState) -> dict:
    route, reason = classify_task(state["task"])
    return {
        "route": route,
        "route_reason": reason,
        "execution_path": append_path(state, "router"),
    }


def decide_route(state: RouterState) -> str:
    return state["route"]


def build_writing_subgraph():
    def make_outline(state: RouterState) -> dict:
        return {
            "outline": [
                "先说明问题",
                "再拆解关键概念",
                "最后给出行动建议",
            ],
            "execution_path": append_path(state, "writing.make_outline"),
        }

    def write_draft(state: RouterState) -> dict:
        outline_text = "；".join(state["outline"])
        return {
            "draft": f"围绕「{state['task']}」写作：{outline_text}。",
            "answer": "写作任务已完成草稿。",
            "execution_path": append_path(state, "writing.write_draft"),
        }

    builder = StateGraph(RouterState)
    builder.add_node("make_outline", make_outline)
    builder.add_node("write_draft", write_draft)
    builder.add_edge(START, "make_outline")
    builder.add_edge("make_outline", "write_draft")
    builder.add_edge("write_draft", END)
    return builder.compile()


def build_qa_subgraph():
    def answer_question(state: RouterState) -> dict:
        return {
            "answer": f"这是一个直接问答任务：{state['task']}",
            "execution_path": append_path(state, "qa.answer_question"),
        }

    builder = StateGraph(RouterState)
    builder.add_node("answer_question", answer_question)
    builder.add_edge(START, "answer_question")
    builder.add_edge("answer_question", END)
    return builder.compile()


def build_code_analysis_subgraph():
    def inspect_code_task(state: RouterState) -> dict:
        notes = [
            "确认问题发生在哪个模块",
            "先复现，再定位，再修改",
            "把可测试逻辑从模型调用中拆出来",
        ]
        return {
            "code_notes": notes,
            "execution_path": append_path(state, "code.inspect_code_task"),
        }

    def summarize_code_review(state: RouterState) -> dict:
        return {
            "answer": "代码分析路径已给出排查建议。",
            "execution_path": append_path(state, "code.summarize_code_review"),
        }

    builder = StateGraph(RouterState)
    builder.add_node("inspect_code_task", inspect_code_task)
    builder.add_node("summarize_code_review", summarize_code_review)
    builder.add_edge(START, "inspect_code_task")
    builder.add_edge("inspect_code_task", "summarize_code_review")
    builder.add_edge("summarize_code_review", END)
    return builder.compile()


def build_search_summary_subgraph():
    def make_search_query(state: RouterState) -> dict:
        return {
            "search_query": state["task"].replace("搜索", "").replace("总结", "").strip(),
            "execution_path": append_path(state, "search.make_search_query"),
        }

    def summarize_materials(state: RouterState) -> dict:
        return {
            "search_summary": f"围绕「{state['search_query']}」整理三条关键资料。",
            "answer": "搜索总结路径已完成资料汇总。",
            "execution_path": append_path(state, "search.summarize_materials"),
        }

    builder = StateGraph(RouterState)
    builder.add_node("make_search_query", make_search_query)
    builder.add_node("summarize_materials", summarize_materials)
    builder.add_edge(START, "make_search_query")
    builder.add_edge("make_search_query", "summarize_materials")
    builder.add_edge("summarize_materials", END)
    return builder.compile()


def build_router_agent():
    builder = StateGraph(RouterState)

    builder.add_node("router", route_task)
    builder.add_node("writing_agent", build_writing_subgraph())
    builder.add_node("qa_agent", build_qa_subgraph())
    builder.add_node("code_analysis_agent", build_code_analysis_subgraph())
    builder.add_node("search_summary_agent", build_search_summary_subgraph())

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        decide_route,
        {
            "writing": "writing_agent",
            "qa": "qa_agent",
            "code_analysis": "code_analysis_agent",
            "search_summary": "search_summary_agent",
        },
    )
    builder.add_edge("writing_agent", END)
    builder.add_edge("qa_agent", END)
    builder.add_edge("code_analysis_agent", END)
    builder.add_edge("search_summary_agent", END)

    return builder.compile()


def main() -> None:
    graph = build_router_agent()

    tasks = [
        "写一段介绍 LangGraph Router Agent 的短文",
        "LangGraph 的 State 是什么？",
        "分析这段 Python 代码为什么状态字段会丢失",
        "搜索并总结 LangGraph checkpoint 的资料",
    ]

    for task in tasks:
        result = graph.invoke({"task": task})
        print("=" * 60)
        print(f"任务：{task}")
        print(f"路由：{result['route']}")
        print(f"原因：{result['route_reason']}")
        print(f"路径：{' -> '.join(result['execution_path'])}")
        print(f"回答：{result['answer']}")


if __name__ == "__main__":
    main()
