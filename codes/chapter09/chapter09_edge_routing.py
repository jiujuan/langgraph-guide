import re
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


class RoutingState(TypedDict, total=False):
    question: str
    question_type: str
    tool_result: str
    answer: str


def classify_question_text(question: str) -> str:
    normalized = question.lower()

    if any(keyword in normalized for keyword in ["删除文件", "泄露", "攻击", "破解"]):
        return "refuse"

    if any(keyword in normalized for keyword in ["计算", "+", "-", "*", "/", "加", "减", "乘", "除"]):
        return "calculation"

    return "direct"


def calculate_expression(question: str) -> str:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", question)
    if not match:
        return "没有找到可计算的表达式。"

    left_text, operator, right_text = match.groups()
    left = float(left_text)
    right = float(right_text)

    if operator == "+":
        result = left + right
    elif operator == "-":
        result = left - right
    elif operator == "*":
        result = left * right
    elif operator == "/":
        if right == 0:
            return "除数不能为 0。"
        result = left / right
    else:
        return "不支持这个运算符。"

    if result.is_integer():
        result = int(result)

    return f"{left_text} {operator} {right_text} = {result}"


def handwritten_route_version(question: str) -> str:
    question_type = classify_question_text(question)

    if question_type == "calculation":
        tool_result = calculate_expression(question)
        return f"计算结果：{tool_result}"

    if question_type == "refuse":
        return "这个问题可能涉及不安全操作，我不能提供相关帮助。"

    response = llm.invoke(f"请用两句话回答这个问题：{question}")
    return response.content.strip()


def classify_question(state: RoutingState) -> dict:
    return {"question_type": classify_question_text(state["question"])}


def route_after_classify(state: RoutingState) -> str:
    if state["question_type"] == "calculation":
        return "calculate_with_tool"

    if state["question_type"] == "refuse":
        return "refuse_question"

    return "answer_directly"


def answer_directly(state: RoutingState) -> dict:
    response = llm.invoke(f"请用两句话回答这个问题：{state['question']}")
    return {"answer": response.content.strip()}


def calculate_with_tool(state: RoutingState) -> dict:
    tool_result = calculate_expression(state["question"])
    return {
        "tool_result": tool_result,
        "answer": f"计算结果：{tool_result}",
    }


def refuse_question(state: RoutingState) -> dict:
    return {"answer": "这个问题可能涉及不安全操作，我不能提供相关帮助。"}


builder = StateGraph(RoutingState)

builder.add_node("classify_question", classify_question)
builder.add_node("answer_directly", answer_directly)
builder.add_node("calculate_with_tool", calculate_with_tool)
builder.add_node("refuse_question", refuse_question)

builder.add_edge(START, "classify_question")
builder.add_conditional_edges("classify_question", route_after_classify)
builder.add_edge("answer_directly", END)
builder.add_edge("calculate_with_tool", END)
builder.add_edge("refuse_question", END)

graph = builder.compile()


if __name__ == "__main__":
    questions = [
        "LangGraph 里的 Edge 是什么？",
        "请计算 18 * 7",
        "如何攻击别人的服务器？",
    ]

    for question in questions:
        print("=" * 60)
        print("问题：")
        print(question)
        print()

        print("一、手写 if/else 路由：")
        print(handwritten_route_version(question))
        print()

        print("二、LangGraph 条件边路由：")
        result = graph.invoke({"question": question})
        print(f"问题类型：{result['question_type']}")
        if "tool_result" in result:
            print(f"工具结果：{result['tool_result']}")
        print(f"回答：{result['answer']}")
        print()
