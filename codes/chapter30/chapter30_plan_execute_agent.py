import sys
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ReviewStatus = Literal["continue", "revise_plan", "finish"]


class PlanStep(TypedDict):
    step_id: str
    title: str
    instruction: str


class StepResult(TypedDict):
    step_id: str
    title: str
    output: str


class PlanExecuteState(TypedDict, total=False):
    task: str
    plan: list[PlanStep]
    current_step_index: int
    step_results: list[StepResult]
    review_status: ReviewStatus
    review_notes: str
    revision_count: int
    final_answer: str
    execution_log: list[str]


def append_log(state: PlanExecuteState, message: str) -> list[str]:
    return [*state.get("execution_log", []), message]


def planner(state: PlanExecuteState) -> dict:
    task = state["task"]
    plan: list[PlanStep] = [
        {
            "step_id": "clarify_goal",
            "title": "澄清目标",
            "instruction": f"明确任务「{task}」要解决的核心问题",
        },
        {
            "step_id": "collect_points",
            "title": "整理要点",
            "instruction": "列出回答需要覆盖的关键概念和示例",
        },
        {
            "step_id": "draft_answer",
            "title": "生成草稿",
            "instruction": "把要点组织成一段结构清晰的说明",
        },
    ]

    return {
        "plan": plan,
        "current_step_index": 0,
        "step_results": [],
        "revision_count": 0,
        "execution_log": append_log(state, "Planner 生成初始计划"),
    }


def executor(state: PlanExecuteState) -> dict:
    index = state.get("current_step_index", 0)
    plan = state["plan"]
    step = plan[index]
    output = f"完成「{step['title']}」：{step['instruction']}。"

    result: StepResult = {
        "step_id": step["step_id"],
        "title": step["title"],
        "output": output,
    }

    return {
        "step_results": [*state.get("step_results", []), result],
        "current_step_index": index + 1,
        "execution_log": append_log(
            state,
            f"Executor 执行步骤 {index + 1}/{len(plan)}：{step['step_id']}",
        ),
    }


def reviewer(state: PlanExecuteState) -> dict:
    index = state.get("current_step_index", 0)
    plan = state["plan"]
    completed_step_ids = {result["step_id"] for result in state.get("step_results", [])}

    if index < len(plan):
        return {
            "review_status": "continue",
            "review_notes": "当前步骤通过，继续执行下一个计划步骤。",
            "execution_log": append_log(state, "Reviewer 通过当前步骤，继续执行"),
        }

    if "review_risks" not in completed_step_ids and state.get("revision_count", 0) < 1:
        return {
            "review_status": "revise_plan",
            "review_notes": "计划缺少风险与边界检查，需要补充一个审查步骤。",
            "execution_log": append_log(state, "Reviewer 发现计划缺口，要求修订计划"),
        }

    return {
        "review_status": "finish",
        "review_notes": "所有计划步骤已经完成，结果可以汇总。",
        "execution_log": append_log(state, "Reviewer 确认任务完成"),
    }


def decide_after_review(state: PlanExecuteState) -> str:
    return state["review_status"]


def revise_plan(state: PlanExecuteState) -> dict:
    plan = [
        *state["plan"],
        {
            "step_id": "review_risks",
            "title": "检查风险与边界",
            "instruction": "补充说明 Plan-and-Execute 的适用场景、代价和失败边界",
        },
    ]

    return {
        "plan": plan,
        "revision_count": state.get("revision_count", 0) + 1,
        "execution_log": append_log(state, "Planner 根据审查意见补充计划步骤 review_risks"),
    }


def finish(state: PlanExecuteState) -> dict:
    lines = [f"# {state['task']}"]
    lines.append("")

    for result in state.get("step_results", []):
        lines.append(f"- {result['title']}：{result['output']}")

    lines.append("")
    lines.append(f"审查结论：{state['review_notes']}")

    return {
        "final_answer": "\n".join(lines),
        "execution_log": append_log(state, "Finish 汇总计划、执行结果和审查结论"),
    }


def build_plan_execute_agent():
    builder = StateGraph(PlanExecuteState)

    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("reviewer", reviewer)
    builder.add_node("revise_plan", revise_plan)
    builder.add_node("finish", finish)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        decide_after_review,
        {
            "continue": "executor",
            "revise_plan": "revise_plan",
            "finish": "finish",
        },
    )
    builder.add_edge("revise_plan", "executor")
    builder.add_edge("finish", END)

    return builder.compile()


def main() -> None:
    graph = build_plan_execute_agent()
    result = graph.invoke({"task": "解释为什么复杂 Agent 需要先规划再执行"})

    print("执行日志：")
    for item in result["execution_log"]:
        print(f"- {item}")

    print("\n最终回答：")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
