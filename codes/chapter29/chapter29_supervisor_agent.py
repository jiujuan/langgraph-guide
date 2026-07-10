import sys
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


WorkerName = Literal["research_worker", "analysis_worker", "writing_worker", "review_worker"]
NextStep = WorkerName | Literal["finish"]


class Task(TypedDict):
    task_id: str
    worker: WorkerName
    instruction: str


class WorkerOutput(TypedDict):
    task_id: str
    worker: WorkerName
    result: str


class SupervisorState(TypedDict, total=False):
    topic: str
    plan: list[Task]
    active_task: Task
    next_step: NextStep
    completed_tasks: list[str]
    worker_outputs: list[WorkerOutput]
    final_report: str
    execution_log: list[str]


def append_log(state: SupervisorState, message: str) -> list[str]:
    return [*state.get("execution_log", []), message]


def build_plan(topic: str) -> list[Task]:
    return [
        {
            "task_id": "research",
            "worker": "research_worker",
            "instruction": f"收集关于「{topic}」的关键背景资料",
        },
        {
            "task_id": "analysis",
            "worker": "analysis_worker",
            "instruction": f"分析「{topic}」为什么需要多 Agent 协作",
        },
        {
            "task_id": "writing",
            "worker": "writing_worker",
            "instruction": f"把「{topic}」整理成读者能快速理解的说明",
        },
        {
            "task_id": "review",
            "worker": "review_worker",
            "instruction": "检查结果是否完整、是否有重复、是否有明显遗漏",
        },
    ]


def supervisor(state: SupervisorState) -> dict:
    plan = state.get("plan") or build_plan(state["topic"])
    completed = set(state.get("completed_tasks", []))

    for task in plan:
        if task["task_id"] not in completed:
            return {
                "plan": plan,
                "active_task": task,
                "next_step": task["worker"],
                "execution_log": append_log(
                    state,
                    f"Supervisor 分配任务 {task['task_id']} 给 {task['worker']}",
                ),
            }

    return {
        "plan": plan,
        "next_step": "finish",
        "execution_log": append_log(state, "Supervisor 确认所有任务完成"),
    }


def decide_next_step(state: SupervisorState) -> str:
    return state["next_step"]


def complete_task(state: SupervisorState, result: str) -> dict:
    task = state["active_task"]
    output: WorkerOutput = {
        "task_id": task["task_id"],
        "worker": task["worker"],
        "result": result,
    }

    return {
        "completed_tasks": [*state.get("completed_tasks", []), task["task_id"]],
        "worker_outputs": [*state.get("worker_outputs", []), output],
        "execution_log": append_log(
            state,
            f"{task['worker']} 完成任务 {task['task_id']}",
        ),
    }


def research_worker(state: SupervisorState) -> dict:
    task = state["active_task"]
    result = f"资料摘要：{task['instruction']}。重点关注任务背景、参与角色和协作边界。"
    return complete_task(state, result)


def analysis_worker(state: SupervisorState) -> dict:
    task = state["active_task"]
    result = f"分析结论：{task['instruction']}。单个 Agent 容易同时承担规划、执行和审查，边界会变模糊。"
    return complete_task(state, result)


def writing_worker(state: SupervisorState) -> dict:
    task = state["active_task"]
    result = f"写作草稿：{task['instruction']}。先讲困境，再讲 Supervisor，再讲 Worker 边界。"
    return complete_task(state, result)


def review_worker(state: SupervisorState) -> dict:
    task = state["active_task"]
    result = f"审查意见：{task['instruction']}。当前结果覆盖了资料、分析、写作和审查。"
    return complete_task(state, result)


def finish(state: SupervisorState) -> dict:
    lines = [f"# {state['topic']}"]

    for output in state.get("worker_outputs", []):
        lines.append(f"- {output['worker']} / {output['task_id']}：{output['result']}")

    return {
        "final_report": "\n".join(lines),
        "execution_log": append_log(state, "finish 汇总所有 Worker 输出"),
    }


def build_supervisor_agent():
    builder = StateGraph(SupervisorState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("research_worker", research_worker)
    builder.add_node("analysis_worker", analysis_worker)
    builder.add_node("writing_worker", writing_worker)
    builder.add_node("review_worker", review_worker)
    builder.add_node("finish", finish)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        decide_next_step,
        {
            "research_worker": "research_worker",
            "analysis_worker": "analysis_worker",
            "writing_worker": "writing_worker",
            "review_worker": "review_worker",
            "finish": "finish",
        },
    )
    builder.add_edge("research_worker", "supervisor")
    builder.add_edge("analysis_worker", "supervisor")
    builder.add_edge("writing_worker", "supervisor")
    builder.add_edge("review_worker", "supervisor")
    builder.add_edge("finish", END)

    return builder.compile()


def main() -> None:
    graph = build_supervisor_agent()
    result = graph.invoke({"topic": "Supervisor 多 Agent 架构"})

    print("执行日志：")
    for item in result["execution_log"]:
        print(f"- {item}")

    print("\n最终报告：")
    print(result["final_report"])


if __name__ == "__main__":
    main()
