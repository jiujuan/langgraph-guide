from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


class LocalAgentState(TypedDict):
    question: str
    intent: str
    answer: str


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


def classify_question(state: LocalAgentState) -> dict:
    prompt = f"""
请判断下面的问题属于哪一类，只输出一个分类词：
- concept：解释概念
- code：询问代码
- other：其他问题

问题：{state["question"]}
"""
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()

    if "concept" in intent:
        return {"intent": "concept"}
    if "code" in intent:
        return {"intent": "code"}
    return {"intent": "other"}


def answer_with_ollama(state: LocalAgentState) -> dict:
    prompt = f"""
你是一个 LangGraph 教学助手。
请根据问题类型回答用户问题。

问题类型：{state["intent"]}
用户问题：{state["question"]}

要求：
1. 回答要简洁。
2. 如果是概念问题，用通俗语言解释。
3. 如果是代码问题，优先说明关键思路，不要生成过长代码。
"""
    response = llm.invoke(prompt)
    return {"answer": response.content}


builder = StateGraph(LocalAgentState)

builder.add_node("classify_question", classify_question)
builder.add_node("answer_with_ollama", answer_with_ollama)

builder.add_edge(START, "classify_question")
builder.add_edge("classify_question", "answer_with_ollama")
builder.add_edge("answer_with_ollama", END)

graph = builder.compile()


if __name__ == "__main__":
    question = "用三句话解释 LangGraph 中 State 的作用。"
    result = graph.invoke({"question": question})

    print(f"问题：{result['question']}")
    print(f"分类：{result['intent']}")
    print()
    print("回答：")
    print(result["answer"])
