from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


class ResearchState(TypedDict, total=False):
    question: str
    question_type: str
    audience: str
    draft_answer: str
    final_answer: str


def classify_question_text(question: str) -> str:
    keywords = ["langgraph", "stategraph", "节点", "边", "状态", "agent"]
    normalized = question.lower()

    if any(keyword in normalized for keyword in keywords):
        return "langgraph"

    return "general"


def infer_audience_text(question: str) -> str:
    beginner_keywords = ["是什么", "初学", "入门", "解释"]
    developer_keywords = ["代码", "实现", "报错", "设计"]

    if any(keyword in question for keyword in developer_keywords):
        return "developer"

    if any(keyword in question for keyword in beginner_keywords):
        return "beginner"

    return "general"


def local_variable_version(question: str) -> str:
    question_type = classify_question_text(question)
    audience = infer_audience_text(question)

    prompt = (
        "请回答下面的问题。\n"
        f"问题类型：{question_type}\n"
        f"读者类型：{audience}\n"
        f"问题：{question}"
    )
    response = llm.invoke(prompt)

    final_answer = response.content.strip()
    return final_answer


def state_dict_version(question: str) -> dict:
    state = {"question": question}

    state["question_type"] = classify_question_text(state["question"])
    state["audience"] = infer_audience_text(state["question"])

    prompt = (
        "请回答下面的问题。\n"
        f"问题类型：{state['question_type']}\n"
        f"读者类型：{state['audience']}\n"
        f"问题：{state['question']}"
    )
    response = llm.invoke(prompt)

    state["draft_answer"] = response.content.strip()
    state["final_answer"] = state["draft_answer"]

    return state


def classify_question(state: ResearchState) -> dict:
    return {"question_type": classify_question_text(state["question"])}


def infer_audience(state: ResearchState) -> dict:
    return {"audience": infer_audience_text(state["question"])}


def draft_answer(state: ResearchState) -> dict:
    prompt = (
        "请回答下面的问题。\n"
        f"问题类型：{state['question_type']}\n"
        f"读者类型：{state['audience']}\n"
        f"问题：{state['question']}"
    )
    response = llm.invoke(prompt)

    return {"draft_answer": response.content.strip()}


def polish_answer(state: ResearchState) -> dict:
    prompt = (
        "请把下面的回答改写得更适合作为技术书籍中的一段说明。"
        "保持简洁，不要增加新概念。\n\n"
        f"原回答：{state['draft_answer']}"
    )
    response = llm.invoke(prompt)

    return {"final_answer": response.content.strip()}


builder = StateGraph(ResearchState)

builder.add_node("classify_question", classify_question)
builder.add_node("infer_audience", infer_audience)
builder.add_node("draft_answer", draft_answer)
builder.add_node("polish_answer", polish_answer)

builder.add_edge(START, "classify_question")
builder.add_edge("classify_question", "infer_audience")
builder.add_edge("infer_audience", "draft_answer")
builder.add_edge("draft_answer", "polish_answer")
builder.add_edge("polish_answer", END)

graph = builder.compile()


if __name__ == "__main__":
    question = "初学者应该如何理解 LangGraph 里的 State？"

    print("问题：")
    print(question)
    print()

    print("一、局部变量版本：")
    print(local_variable_version(question))
    print()

    print("二、普通 dict 状态版本：")
    dict_result = state_dict_version(question)
    print(f"问题类型：{dict_result['question_type']}")
    print(f"读者类型：{dict_result['audience']}")
    print(dict_result["final_answer"])
    print()

    print("三、LangGraph State 版本：")
    graph_result = graph.invoke({"question": question})
    print(f"问题类型：{graph_result['question_type']}")
    print(f"读者类型：{graph_result['audience']}")
    print()
    print("草稿回答：")
    print(graph_result["draft_answer"])
    print()
    print("最终回答：")
    print(graph_result["final_answer"])
