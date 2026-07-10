from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


class AnswerState(TypedDict, total=False):
    question: str
    question_type: str
    audience: str
    outline: str
    draft_answer: str
    review_notes: str
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


def giant_function_version(question: str) -> str:
    question_type = classify_question_text(question)
    audience = infer_audience_text(question)

    outline_prompt = (
        "请为下面的问题列出一个三点回答提纲。\n"
        f"问题类型：{question_type}\n"
        f"读者类型：{audience}\n"
        f"问题：{question}"
    )
    outline = llm.invoke(outline_prompt).content.strip()

    draft_prompt = (
        "请根据提纲写一段适合技术书籍的回答。\n"
        f"问题：{question}\n"
        f"提纲：{outline}"
    )
    draft_answer = llm.invoke(draft_prompt).content.strip()

    review_prompt = (
        "请审查下面的回答是否适合初学者。"
        "只指出一两个最需要改进的问题。\n\n"
        f"回答：{draft_answer}"
    )
    review_notes = llm.invoke(review_prompt).content.strip()

    final_prompt = (
        "请根据审查意见改写回答，要求简洁、清楚、不要引入新概念。\n"
        f"原回答：{draft_answer}\n"
        f"审查意见：{review_notes}"
    )
    final_answer = llm.invoke(final_prompt).content.strip()

    return final_answer


def classify_question(state: AnswerState) -> dict:
    return {"question_type": classify_question_text(state["question"])}


def infer_audience(state: AnswerState) -> dict:
    return {"audience": infer_audience_text(state["question"])}


def create_outline(state: AnswerState) -> dict:
    prompt = (
        "请为下面的问题列出一个三点回答提纲。\n"
        f"问题类型：{state['question_type']}\n"
        f"读者类型：{state['audience']}\n"
        f"问题：{state['question']}"
    )
    response = llm.invoke(prompt)

    return {"outline": response.content.strip()}


def write_draft(state: AnswerState) -> dict:
    prompt = (
        "请根据提纲写一段适合技术书籍的回答。\n"
        f"问题：{state['question']}\n"
        f"提纲：{state['outline']}"
    )
    response = llm.invoke(prompt)

    return {"draft_answer": response.content.strip()}


def review_draft(state: AnswerState) -> dict:
    prompt = (
        "请审查下面的回答是否适合初学者。"
        "只指出一两个最需要改进的问题。\n\n"
        f"回答：{state['draft_answer']}"
    )
    response = llm.invoke(prompt)

    return {"review_notes": response.content.strip()}


def revise_answer(state: AnswerState) -> dict:
    prompt = (
        "请根据审查意见改写回答，要求简洁、清楚、不要引入新概念。\n"
        f"原回答：{state['draft_answer']}\n"
        f"审查意见：{state['review_notes']}"
    )
    response = llm.invoke(prompt)

    return {"final_answer": response.content.strip()}


builder = StateGraph(AnswerState)

builder.add_node("classify_question", classify_question)
builder.add_node("infer_audience", infer_audience)
builder.add_node("create_outline", create_outline)
builder.add_node("write_draft", write_draft)
builder.add_node("review_draft", review_draft)
builder.add_node("revise_answer", revise_answer)

builder.add_edge(START, "classify_question")
builder.add_edge("classify_question", "infer_audience")
builder.add_edge("infer_audience", "create_outline")
builder.add_edge("create_outline", "write_draft")
builder.add_edge("write_draft", "review_draft")
builder.add_edge("review_draft", "revise_answer")
builder.add_edge("revise_answer", END)

graph = builder.compile()


if __name__ == "__main__":
    question = "初学者应该如何理解 LangGraph 里的 Node？"

    print("问题：")
    print(question)
    print()

    print("一、大函数版本：")
    print(giant_function_version(question))
    print()

    print("二、LangGraph 节点版本：")
    result = graph.invoke({"question": question})
    print(f"问题类型：{result['question_type']}")
    print(f"读者类型：{result['audience']}")
    print()
    print("回答提纲：")
    print(result["outline"])
    print()
    print("审查意见：")
    print(result["review_notes"])
    print()
    print("最终回答：")
    print(result["final_answer"])
