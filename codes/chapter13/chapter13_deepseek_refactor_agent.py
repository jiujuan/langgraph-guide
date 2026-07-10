from typing import Protocol, TypedDict

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph


class ChatModel(Protocol):
    def invoke(self, input: str) -> BaseMessage:
        ...


class HybridAgentState(TypedDict):
    question: str
    intent: str
    answer: str


def build_ollama_model() -> ChatModel:
    return ChatOllama(
        model="qwen3:4b",
        temperature=0,
    )


def build_deepseek_model() -> ChatModel:
    return ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0,
    )


load_dotenv()

fast_model = build_ollama_model()
reasoning_model = build_deepseek_model()


def classify_question(state: HybridAgentState) -> dict:
    prompt = f"""
请判断下面的问题属于哪一类，只输出一个分类词：
- concept：解释概念
- code：询问代码
- reasoning：需要复杂推理
- other：其他问题

问题：{state["question"]}
"""
    response = fast_model.invoke(prompt)
    intent = response.content.strip().lower()

    if "reasoning" in intent:
        return {"intent": "reasoning"}
    if "concept" in intent:
        return {"intent": "concept"}
    if "code" in intent:
        return {"intent": "code"}
    return {"intent": "other"}


def answer_with_deepseek(state: HybridAgentState) -> dict:
    prompt = f"""
你是一个 LangGraph Agent 开发导师。
请回答下面的问题。

问题类型：{state["intent"]}
用户问题：{state["question"]}

要求：
1. 先给出直接答案。
2. 再解释关键原因。
3. 如果问题涉及取舍，请说明适用场景。
4. 回答保持清晰，不要堆 API 名称。
"""
    response = reasoning_model.invoke(prompt)
    return {"answer": response.content}


builder = StateGraph(HybridAgentState)

builder.add_node("classify_question", classify_question)
builder.add_node("answer_with_deepseek", answer_with_deepseek)

builder.add_edge(START, "classify_question")
builder.add_edge("classify_question", "answer_with_deepseek")
builder.add_edge("answer_with_deepseek", END)

graph = builder.compile()


if __name__ == "__main__":
    question = (
        "如果一个 LangGraph Agent 既要本地隐私，又要复杂推理，"
        "应该如何在 Ollama 和 DeepSeek 之间分工？"
    )
    result = graph.invoke({"question": question})

    print(f"问题：{result['question']}")
    print(f"分类：{result['intent']}")
    print()
    print("回答：")
    print(result["answer"])
