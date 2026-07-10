import ast
import json
import operator
import re
from typing import Protocol, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph


class ChatModel(Protocol):
    def invoke(self, input: str) -> BaseMessage:
        ...


class ToolAgentState(TypedDict, total=False):
    question: str
    action: str
    tool_input: str
    tool_result: str
    answer: str


load_dotenv()

reasoning_model: ChatModel = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
)


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
        raise ValueError("只支持数字和基础四则运算。")

    tree = ast.parse(expression, mode="eval")
    result = eval_node(tree)
    return str(int(result)) if result == int(result) else str(result)


def search_local_knowledge(query: str) -> str:
    knowledge = {
        "state": "State 是 LangGraph 图运行时携带的共享数据，节点通过读取和更新 State 协作。",
        "node": "Node 是图中的一步工作，通常是一个读取 State 并返回状态更新的函数。",
        "edge": "Edge 描述节点之间的流向，可以是固定边，也可以是条件边。",
        "tool": "Tool 是 Agent 访问外部能力的入口，例如计算器、文件读取、检索或业务 API。",
    }
    lowered_query = query.lower()
    hits = [value for key, value in knowledge.items() if key in lowered_query]
    return "\n".join(hits) if hits else "本地知识库没有找到相关资料。"


def choose_action(state: ToolAgentState) -> dict:
    prompt = f"""
你是一个 LangGraph Agent 的动作选择器。
请判断用户问题是否需要工具。

可选动作：
- calculator：需要精确计算
- knowledge：需要查询本地 LangGraph 知识
- direct：不需要工具，可以直接回答

只输出 JSON，不要输出其他文字：
{{"action": "calculator|knowledge|direct", "tool_input": "传给工具的输入"}}

用户问题：{state["question"]}
"""
    response = reasoning_model.invoke(prompt)
    content = response.content.strip()

    try:
        decision = json.loads(content)
        action = decision.get("action", "direct")
        tool_input = decision.get("tool_input", state["question"])
    except json.JSONDecodeError:
        action = "direct"
        tool_input = state["question"]

    if action not in {"calculator", "knowledge", "direct"}:
        action = "direct"

    if action == "calculator":
        match = re.search(r"[-+*/().\d\s]+", tool_input)
        if match:
            tool_input = match.group(0).strip()

    return {"action": action, "tool_input": tool_input}


def route_after_action(state: ToolAgentState) -> str:
    if state["action"] == "calculator":
        return "calculator_tool"
    if state["action"] == "knowledge":
        return "knowledge_tool"
    return "direct_answer"


def calculator_tool(state: ToolAgentState) -> dict:
    try:
        result = safe_calculate(state["tool_input"])
        return {"tool_result": f"计算结果：{state['tool_input']} = {result}"}
    except Exception as exc:
        return {"tool_result": f"计算失败：{exc}"}


def knowledge_tool(state: ToolAgentState) -> dict:
    result = search_local_knowledge(state["tool_input"])
    return {"tool_result": result}


def answer_with_tool_result(state: ToolAgentState) -> dict:
    prompt = f"""
你是一个 LangGraph 教学助手。
请根据工具结果回答用户问题。

用户问题：{state["question"]}
工具结果：{state["tool_result"]}

要求：
1. 先直接回答。
2. 再用一句话说明工具在这里解决了什么问题。
"""
    response = reasoning_model.invoke(prompt)
    return {"answer": response.content}


def direct_answer(state: ToolAgentState) -> dict:
    prompt = f"""
你是一个 LangGraph 教学助手。
请直接回答用户问题，不要声称使用了工具。

用户问题：{state["question"]}
"""
    response = reasoning_model.invoke(prompt)
    return {"answer": response.content}


builder = StateGraph(ToolAgentState)

builder.add_node("choose_action", choose_action)
builder.add_node("calculator_tool", calculator_tool)
builder.add_node("knowledge_tool", knowledge_tool)
builder.add_node("answer_with_tool_result", answer_with_tool_result)
builder.add_node("direct_answer", direct_answer)

builder.add_edge(START, "choose_action")
builder.add_conditional_edges(
    "choose_action",
    route_after_action,
    {
        "calculator_tool": "calculator_tool",
        "knowledge_tool": "knowledge_tool",
        "direct_answer": "direct_answer",
    },
)
builder.add_edge("calculator_tool", "answer_with_tool_result")
builder.add_edge("knowledge_tool", "answer_with_tool_result")
builder.add_edge("answer_with_tool_result", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()


if __name__ == "__main__":
    question = "请计算 128 * 32，并说明为什么这个问题不应该只靠大模型心算。"
    result = graph.invoke({"question": question})

    print(f"问题：{result['question']}")
    print(f"动作：{result['action']}")
    print(f"工具输入：{result.get('tool_input', '')}")
    print(f"工具结果：{result.get('tool_result', '未调用工具')}")
    print()
    print("回答：")
    print(result["answer"])
