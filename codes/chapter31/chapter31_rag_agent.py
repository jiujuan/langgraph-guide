import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class Document(TypedDict):
    doc_id: str
    title: str
    source: str
    text: str


class ScoredDocument(TypedDict):
    doc: Document
    score: int


class RAGState(TypedDict, total=False):
    question: str
    query: str
    retrieved_docs: list[ScoredDocument]
    selected_docs: list[Document]
    context: str
    answer: str
    citations: list[str]
    confidence: str
    execution_log: list[str]


KNOWLEDGE_BASE: list[Document] = [
    {
        "doc_id": "lg-state",
        "title": "LangGraph State",
        "source": "docs/core/state.md",
        "text": "State 是 LangGraph 图执行过程中的共享数据结构。节点读取 State，并返回局部更新。",
    },
    {
        "doc_id": "lg-checkpoint",
        "title": "Checkpoint",
        "source": "docs/runtime/checkpoint.md",
        "text": "Checkpoint 会保存图的中间状态，让长任务可以恢复、回放或在人工介入后继续执行。",
    },
    {
        "doc_id": "lg-rag",
        "title": "RAG Agent",
        "source": "docs/patterns/rag.md",
        "text": "RAG Agent 先检索外部资料，再把相关片段注入上下文，最后基于证据生成带来源的回答。",
    },
    {
        "doc_id": "lg-router",
        "title": "Router Agent",
        "source": "docs/patterns/router.md",
        "text": "Router Agent 根据任务类型选择不同路径，适合把写作、问答、代码分析、搜索总结分给不同子图。",
    },
]


def append_log(state: RAGState, message: str) -> list[str]:
    return [*state.get("execution_log", []), message]


def normalize(text: str) -> set[str]:
    lowered = text.lower()
    for char in "，。！？：；、,.!?;:/()[]{}<>\"'`":
        lowered = lowered.replace(char, " ")
    return {token for token in lowered.split() if token}


def build_query(state: RAGState) -> dict:
    query = state["question"].replace("请", "").replace("解释", "").strip()
    return {
        "query": query,
        "execution_log": append_log(state, f"build_query 生成检索词：{query}"),
    }


def retrieve_docs(state: RAGState) -> dict:
    query_terms = normalize(state["query"])
    scored: list[ScoredDocument] = []

    for doc in KNOWLEDGE_BASE:
        doc_terms = normalize(f"{doc['title']} {doc['text']}")
        score = len(query_terms & doc_terms)
        if score > 0:
            scored.append({"doc": doc, "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)

    return {
        "retrieved_docs": scored,
        "execution_log": append_log(state, f"retrieve_docs 找到 {len(scored)} 条候选资料"),
    }


def filter_docs(state: RAGState) -> dict:
    retrieved = state.get("retrieved_docs", [])
    max_score = retrieved[0]["score"] if retrieved else 0
    selected = [item["doc"] for item in retrieved if item["score"] == max_score]
    selected = selected[:3]

    confidence = "high" if len(selected) >= 2 else "low"
    if not selected:
        confidence = "none"

    return {
        "selected_docs": selected,
        "confidence": confidence,
        "execution_log": append_log(state, f"filter_docs 选中 {len(selected)} 条资料"),
    }


def build_context(state: RAGState) -> dict:
    docs = state.get("selected_docs", [])
    context_lines = []

    for index, doc in enumerate(docs, start=1):
        context_lines.append(f"[{index}] {doc['title']} ({doc['source']}): {doc['text']}")

    return {
        "context": "\n".join(context_lines),
        "citations": [doc["source"] for doc in docs],
        "execution_log": append_log(state, "build_context 构造带编号的证据上下文"),
    }


def answer_with_context(state: RAGState) -> dict:
    if not state.get("selected_docs"):
        return {
            "answer": "当前知识库没有找到足够资料，不能基于证据回答这个问题。",
            "citations": [],
            "execution_log": append_log(state, "answer_with_context 因证据不足而拒绝编造"),
        }

    answer = (
        f"问题：{state['question']}\n\n"
        "基于知识库资料，可以这样回答：\n"
        f"{state['context']}\n\n"
        "结论：RAG Agent 的关键不是让模型凭记忆回答，而是先检索资料、筛选证据、"
        "把证据注入上下文，再生成带来源的回答。"
    )

    return {
        "answer": answer,
        "execution_log": append_log(state, "answer_with_context 基于证据生成回答"),
    }


def build_rag_agent():
    builder = StateGraph(RAGState)

    builder.add_node("build_query", build_query)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("filter_docs", filter_docs)
    builder.add_node("build_context", build_context)
    builder.add_node("answer_with_context", answer_with_context)

    builder.add_edge(START, "build_query")
    builder.add_edge("build_query", "retrieve_docs")
    builder.add_edge("retrieve_docs", "filter_docs")
    builder.add_edge("filter_docs", "build_context")
    builder.add_edge("build_context", "answer_with_context")
    builder.add_edge("answer_with_context", END)

    return builder.compile()


def main() -> None:
    graph = build_rag_agent()
    result = graph.invoke({"question": "请解释 RAG Agent 如何基于资料回答"})

    print("执行日志：")
    for item in result["execution_log"]:
        print(f"- {item}")

    print("\n引用来源：")
    for source in result["citations"]:
        print(f"- {source}")

    print("\n回答：")
    print(result["answer"])


if __name__ == "__main__":
    main()
