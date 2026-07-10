from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send


class RevisionState(TypedDict, total=False):
    question: str
    draft: str
    review_notes: str
    rewrite_count: int
    final_answer: str


def write_first_draft(state: RevisionState) -> dict:
    return {"draft": "Node 是 LangGraph 里的一个函数。"}


def review_draft(
    state: RevisionState,
) -> Command[Literal["rewrite_answer", "finalize_answer"]]:
    draft = state["draft"]
    rewrite_count = state.get("rewrite_count", 0)

    if "读取 State" in draft and "返回状态更新" in draft:
        return Command(
            update={"review_notes": "审查通过：回答已经说明了 Node 的输入和输出。"},
            goto="finalize_answer",
        )

    if rewrite_count >= 1:
        return Command(
            update={"review_notes": "已重写过一次，接受当前版本并结束。"},
            goto="finalize_answer",
        )

    return Command(
        update={
            "review_notes": "回答太泛，需要说明 Node 会读取 State 并返回状态更新。",
            "rewrite_count": rewrite_count + 1,
        },
        goto="rewrite_answer",
    )


def rewrite_answer(state: RevisionState) -> dict:
    return {
        "draft": (
            "Node 是 LangGraph 中完成一步工作的函数。"
            "它读取当前 State，执行模型调用、工具调用或判断逻辑，"
            "然后返回本步骤产生的状态更新。"
        )
    }


def finalize_answer(state: RevisionState) -> dict:
    return {"final_answer": state["draft"]}


command_builder = StateGraph(RevisionState)

command_builder.add_node("write_first_draft", write_first_draft)
command_builder.add_node("review_draft", review_draft)
command_builder.add_node("rewrite_answer", rewrite_answer)
command_builder.add_node("finalize_answer", finalize_answer)

command_builder.add_edge(START, "write_first_draft")
command_builder.add_edge("write_first_draft", "review_draft")
command_builder.add_edge("rewrite_answer", "review_draft")
command_builder.add_edge("finalize_answer", END)

command_graph = command_builder.compile()


class SummaryState(TypedDict, total=False):
    topic: str
    sections: list[str]
    section_summaries: Annotated[list[str], add]
    final_report: str


class SectionState(TypedDict):
    section: str


def plan_sections(state: SummaryState) -> dict:
    return {
        "sections": [
            "State 负责保存工作记忆",
            "Node 负责完成一步工作",
            "Edge 负责决定下一步流向",
        ]
    }


def dispatch_sections(state: SummaryState) -> list[Send]:
    return [
        Send("summarize_section", {"section": section})
        for section in state["sections"]
    ]


def summarize_section(state: SectionState) -> dict:
    return {"section_summaries": [f"小结：{state['section']}。"]}


def write_final_report(state: SummaryState) -> dict:
    body = "\n".join(f"- {summary}" for summary in state["section_summaries"])
    return {"final_report": f"{state['topic']} 的核心编程模型：\n{body}"}


send_builder = StateGraph(SummaryState)

send_builder.add_node("plan_sections", plan_sections)
send_builder.add_node("summarize_section", summarize_section)
send_builder.add_node("write_final_report", write_final_report)

send_builder.add_edge(START, "plan_sections")
send_builder.add_conditional_edges(
    "plan_sections",
    dispatch_sections,
    ["summarize_section"],
)
send_builder.add_edge("summarize_section", "write_final_report")
send_builder.add_edge("write_final_report", END)

send_graph = send_builder.compile()


if __name__ == "__main__":
    print("一、Command 示例：")
    command_result = command_graph.invoke(
        {"question": "初学者应该如何理解 LangGraph 里的 Node？"}
    )
    print(f"审查意见：{command_result['review_notes']}")
    print(f"重写次数：{command_result.get('rewrite_count', 0)}")
    print(f"最终回答：{command_result['final_answer']}")
    print()

    print("二、Send 示例：")
    send_result = send_graph.invoke(
        {
            "topic": "LangGraph",
            "section_summaries": [],
        }
    )
    print(send_result["final_report"])
