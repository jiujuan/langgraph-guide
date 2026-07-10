from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


class ChatState(TypedDict):
    question: str
    answer: str


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


def answer_question(state: ChatState) -> dict:
    response = llm.invoke(state["question"])
    return {"answer": response.content}


builder = StateGraph(ChatState)

builder.add_node("answer_question", answer_question)
builder.add_edge(START, "answer_question")
builder.add_edge("answer_question", END)

graph = builder.compile()


if __name__ == "__main__":
    question = "用一句话解释 LangGraph 是什么。"
    result = graph.invoke({"question": question})

    print(f"问题：{result['question']}")
    print()
    print(f"回答：{result['answer']}")
