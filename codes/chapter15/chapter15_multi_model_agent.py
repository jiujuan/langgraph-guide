import ast
import json
import operator
import re
from typing import Protocol, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


class ChatModel(Protocol):
    def invoke(self, input: str) -> BaseMessage:
        ...


class MultiModelState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    sanitized_question: str
    tool_result: str
    answer: str


load_dotenv()

fast_model: ChatModel = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)

reasoning_model: ChatModel = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
)


def mask_private_info(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)
    text = re.sub(r"\b1[3-9]\d{9}\b", "[PHONE]", text)
    return text


def safe_calculate(expression: str) -> str:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](eval_node(node.operand))
        raise ValueError("Only numbers and basic arithmetic are supported.")

    tree = ast.parse(expression, mode="eval")
    result = eval_node(tree)
    return str(int(result)) if result == int(result) else str(result)


def route_question(state: MultiModelState) -> dict:
    prompt = f"""
You are a router for a LangGraph multi-model agent.
Choose exactly one route:
- local_answer: simple explanation, low risk, no external tool
- tool_then_reasoning: needs exact calculation or local tool result
- deep_reasoning: needs architecture tradeoff, planning, review, or complex reasoning

Return JSON only:
{{"route": "local_answer|tool_then_reasoning|deep_reasoning", "route_reason": "short reason"}}

User question: {state["question"]}
"""
    response = fast_model.invoke(prompt)
    content = response.content.strip()

    try:
        decision = json.loads(content)
        route = decision.get("route", "deep_reasoning")
        route_reason = decision.get("route_reason", "model selected route")
    except json.JSONDecodeError:
        question = state["question"].lower()
        if re.search(r"\d+\s*[-+*/]\s*\d+", question):
            route = "tool_then_reasoning"
            route_reason = "question contains arithmetic"
        elif any(word in question for word in ["tradeoff", "architecture", "架构", "权衡", "推理"]):
            route = "deep_reasoning"
            route_reason = "question asks for architecture reasoning"
        else:
            route = "local_answer"
            route_reason = "question looks simple"

    if route not in {"local_answer", "tool_then_reasoning", "deep_reasoning"}:
        route = "deep_reasoning"

    return {"route": route, "route_reason": route_reason}


def route_after_router(state: MultiModelState) -> str:
    if state["route"] == "local_answer":
        return "local_answer"
    if state["route"] == "tool_then_reasoning":
        return "tool_executor"
    return "privacy_filter"


def local_answer(state: MultiModelState) -> dict:
    prompt = f"""
Answer briefly as a LangGraph teaching assistant.
Keep the answer simple and practical.

Question: {state["question"]}
"""
    response = fast_model.invoke(prompt)
    return {"answer": response.content}


def tool_executor(state: MultiModelState) -> dict:
    expression_match = re.search(r"[-+*/().\d\s]+", state["question"])
    if not expression_match:
        return {"tool_result": "No arithmetic expression was found."}

    expression = expression_match.group(0).strip()
    try:
        result = safe_calculate(expression)
        return {"tool_result": f"{expression} = {result}"}
    except Exception as exc:
        return {"tool_result": f"Tool execution failed: {exc}"}


def privacy_filter(state: MultiModelState) -> dict:
    source = state.get("sanitized_question", state["question"])
    return {"sanitized_question": mask_private_info(source)}


def deep_reasoning(state: MultiModelState) -> dict:
    question = state.get("sanitized_question", state["question"])
    tool_result = state.get("tool_result", "No tool was used.")

    prompt = f"""
You are a LangGraph architecture mentor.
Answer the question using clear tradeoffs.

Route selected by local model: {state["route"]}
Route reason: {state["route_reason"]}
Question: {question}
Tool result: {tool_result}

Please include:
1. Direct recommendation.
2. Why this model/tool split is appropriate.
3. What tradeoff remains.
"""
    response = reasoning_model.invoke(prompt)
    return {"answer": response.content}


builder = StateGraph(MultiModelState)

builder.add_node("route_question", route_question)
builder.add_node("local_answer", local_answer)
builder.add_node("tool_executor", tool_executor)
builder.add_node("privacy_filter", privacy_filter)
builder.add_node("deep_reasoning", deep_reasoning)

builder.add_edge(START, "route_question")
builder.add_conditional_edges(
    "route_question",
    route_after_router,
    {
        "local_answer": "local_answer",
        "tool_executor": "tool_executor",
        "privacy_filter": "privacy_filter",
    },
)
builder.add_edge("tool_executor", "privacy_filter")
builder.add_edge("privacy_filter", "deep_reasoning")
builder.add_edge("local_answer", END)
builder.add_edge("deep_reasoning", END)

graph = builder.compile()


if __name__ == "__main__":
    question = (
        "我的邮箱是 user@example.com。请计算 128 * 32，"
        "并说明这个任务为什么适合 Ollama 路由、工具计算、DeepSeek 总结。"
    )
    result = graph.invoke({"question": question})

    print(f"问题：{result['question']}")
    print(f"路由：{result['route']}")
    print(f"路由原因：{result['route_reason']}")
    print(f"脱敏问题：{result.get('sanitized_question', '未脱敏')}")
    print(f"工具结果：{result.get('tool_result', '未调用工具')}")
    print()
    print("回答：")
    print(result["answer"])
