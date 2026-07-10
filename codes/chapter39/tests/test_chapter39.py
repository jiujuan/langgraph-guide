from __future__ import annotations

import sys
import unittest
from pathlib import Path


CHAPTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAPTER_DIR))

class ResearchAssistantTests(unittest.TestCase):
    def test_migrates_legacy_approval_field_to_version_two(self) -> None:
        from research_assistant.graph.state import migrate_state

        migrated = migrate_state({"approvalState": "approved", "state_schema_version": 1})

        self.assertEqual(migrated["state_schema_version"], 2)
        self.assertEqual(migrated["approval_status"], "approved")

    def test_new_task_waits_for_human_approval(self) -> None:
        from research_assistant.api.service import ResearchService

        service = ResearchService.demo()
        task = service.create_task(
            user_id="reader-1",
            topic="如何设计可扩展的复杂 Agent 架构？",
        )

        self.assertEqual(task["lifecycle_status"], "awaiting_approval")
        self.assertEqual(task["approval_status"], "pending")

    def test_approved_task_runs_to_report_with_audited_tool_calls(self) -> None:
        from research_assistant.api.service import ResearchService

        service = ResearchService.demo()
        task = service.create_task(
            user_id="reader-1",
            topic="如何设计可扩展的复杂 Agent 架构？",
        )
        service.approve_plan(task["task_id"], reviewer_id="architect-1")

        completed = service.run_task(task["task_id"])

        self.assertEqual(completed["lifecycle_status"], "completed")
        self.assertIn("模块边界", completed["final_report"])
        self.assertGreaterEqual(len(service.audit_records(task["task_id"])), 1)

    def test_unknown_tool_is_denied_before_execution(self) -> None:
        from research_assistant.tools.permissions import authorize_tool_request

        decision = authorize_tool_request(
            request={
                "request_id": "req-1",
                "subtask_id": "subtask-1",
                "tool_name": "shell_exec",
                "arguments": {"command": "whoami"},
            },
            user_id="reader-1",
            registry={},
        )

        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["reason"], "tool_not_registered")


if __name__ == "__main__":
    unittest.main()
