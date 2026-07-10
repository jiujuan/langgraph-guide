from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


class AssistantState(TypedDict):
    question: str
    question_type: str
    answer: str


def classify_question_text(question: str) -> str:
    keywords = ["langgraph", "stategraph", "节点", "边", "状态图", "agent"]
    normalized = question.lower()

    if any(keyword in normalized for keyword in keywords):
        return "langgraph"

    return "general"


def plain_llm_call(question: str) -> str:
    response = llm.invoke(f"请用两句话回答这个问题：{question}")
    return response.content


def handwritten_branch_call(question: str) -> str:
    question_type = classify_question_text(question)

    if question_type == "langgraph":
        prompt = f"请面向 LangGraph 初学者，用两句话回答：{question}"
    else:
        prompt = f"请用两句话回答这个普通问题：{question}"

    response = llm.invoke(prompt)
    return response.content


def classify_question(state: AssistantState) -> dict:
    question_type = classify_question_text(state["question"])
    return {"question_type": question_type}


def answer_langgraph_question(state: AssistantState) -> dict:
    response = llm.invoke(
        f"请面向 LangGraph 初学者，用两句话回答：{state['question']}"
    )
    return {"answer": response.content}


def answer_general_question(state: AssistantState) -> dict:
    response = llm.invoke(f"请用两句话回答这个普通问题：{state['question']}")
    return {"answer": response.content}


def route_after_classify(state: AssistantState) -> str:
    if state["question_type"] == "langgraph":
        return "answer_langgraph_question"

    return "answer_general_question"


builder = StateGraph(AssistantState)

builder.add_node("classify_question", classify_question)
builder.add_node("answer_langgraph_question", answer_langgraph_question)
builder.add_node("answer_general_question", answer_general_question)

builder.add_edge(START, "classify_question")
builder.add_conditional_edges("classify_question", route_after_classify)
builder.add_edge("answer_langgraph_question", END)
builder.add_edge("answer_general_question", END)

graph = builder.compile()


if __name__ == "__main__":
    question = "LangGraph 为什么比普通函数更适合多步骤 Agent？"

    print("问题：")
    print(question)
    print()

    print("一、普通线性调用：")
    print(plain_llm_call(question))
    print()

    print("二、手写 if/else 分支：")
    print(handwritten_branch_call(question))
    print()

    print("三、LangGraph 图调用：")
    result = graph.invoke({"question": question})
    print(f"问题类型：{result['question_type']}")
    print(result["answer"])
