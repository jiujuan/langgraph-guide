from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class ReportState(TypedDict, total=False):
    topic: str
    notes: Annotated[list[str], add]
    report: str


def overwrite_dict_version(topic: str) -> dict:
    state = {
        "topic": topic,
        "notes": [],
    }

    state["notes"] = [f"关键词观察：{topic} 通常和状态、节点、边有关。"]
    state["notes"] = [f"结构观察：{topic} 适合用图表示多步骤流程。"]

    state["report"] = "\n".join(state["notes"])
    return state


def extract_keyword_notes(state: ReportState) -> dict:
    return {
        "notes": [
            f"关键词观察：{state['topic']} 通常和状态、节点、边有关。"
        ]
    }


def extract_structure_notes(state: ReportState) -> dict:
    return {
        "notes": [
            f"结构观察：{state['topic']} 适合用图表示多步骤流程。"
        ]
    }


def write_report(state: ReportState) -> dict:
    report = "综合观察：\n" + "\n".join(f"- {note}" for note in state["notes"])
    return {"report": report}


builder = StateGraph(ReportState)

builder.add_node("extract_keyword_notes", extract_keyword_notes)
builder.add_node("extract_structure_notes", extract_structure_notes)
builder.add_node("write_report", write_report)

builder.add_edge(START, "extract_keyword_notes")
builder.add_edge(START, "extract_structure_notes")
builder.add_edge(["extract_keyword_notes", "extract_structure_notes"], "write_report")
builder.add_edge("write_report", END)

graph = builder.compile()


if __name__ == "__main__":
    topic = "LangGraph Agent"

    print("主题：")
    print(topic)
    print()

    print("一、普通 dict 覆盖版本：")
    dict_result = overwrite_dict_version(topic)
    print(dict_result["report"])
    print()

    print("二、LangGraph reducer 合并版本：")
    graph_result = graph.invoke({"topic": topic, "notes": []})
    print(graph_result["report"])
