from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..nodes.workflow import WorkflowNodes
from .state import ResearchState


def build_research_graph(nodes: WorkflowNodes):
    builder = StateGraph(ResearchState)
    builder.add_node("router", nodes.router)
    builder.add_node("planner", nodes.planner)
    builder.add_node("approval", nodes.approval)
    builder.add_node("researcher", nodes.researcher)
    builder.add_node("tool_executor", nodes.tool_executor)
    builder.add_node("extractor", nodes.extractor)
    builder.add_node("synthesizer", nodes.synthesizer)
    builder.add_node("reviewer", nodes.reviewer)
    builder.add_node("writer", nodes.writer)
    builder.add_edge(START, "router")
    builder.add_edge("router", "planner")
    builder.add_edge("planner", "approval")
    builder.add_conditional_edges("approval", lambda state: "researcher" if state.get("approval_status") == "approved" else "end", {"researcher": "researcher", "end": END})
    builder.add_conditional_edges("researcher", lambda state: "synthesizer" if len(state.get("completed_subtask_ids", [])) == len(state["plan"]) else "tool_executor", {"tool_executor": "tool_executor", "synthesizer": "synthesizer"})
    builder.add_edge("tool_executor", "extractor")
    builder.add_edge("extractor", "researcher")
    builder.add_edge("synthesizer", "reviewer")
    builder.add_conditional_edges("reviewer", lambda state: "writer" if state.get("review_status") == "pass" else "researcher", {"writer": "writer", "researcher": "researcher"})
    builder.add_edge("writer", END)
    return builder.compile()
