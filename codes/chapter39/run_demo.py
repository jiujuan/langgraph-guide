from __future__ import annotations

import sys
from pathlib import Path


CHAPTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHAPTER_DIR))

from research_assistant.api.service import ResearchService
from research_assistant.evals.evaluators import evaluate_report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    service = ResearchService.demo()
    task = service.create_task(
        user_id="reader-1",
        topic="如何设计可扩展的复杂 Agent 架构？",
    )
    print(f"创建任务：{task['task_id']}，状态：{task['lifecycle_status']}")
    print(f"计划包含 {len(task['state']['plan'])} 个子任务，等待人工审批。")

    service.approve_plan(task["task_id"], reviewer_id="architect-1")
    completed = service.run_task(task["task_id"])

    print(f"\n恢复后状态：{completed['lifecycle_status']}")
    print("\n执行追踪：")
    for event in completed["state"]["execution_log"]:
        print(f"- {event}")

    print("\n最终报告：\n")
    print(completed["final_report"])

    evaluation = evaluate_report(completed["final_report"], completed["report_metadata"])
    print(f"\n规则评测：{evaluation}")
    print(f"\n审计记录：{service.audit_records(task['task_id'])}")


if __name__ == "__main__":
    main()
