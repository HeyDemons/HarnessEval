"""The orchestrator must be told which participant actually holds a tool.

The adapter routes benchmark-native tools to WebSurfer, so on Tau2 the other three
participants can only produce text. Before this, the ledger saw only the canonical role
blurbs and dispatched 278 times to those three while WebSurfer got zero, burning the whole
wall clock on 58 of 60 arms.
"""
import unittest

from benchmark_platform.harnesses.magentic_one import _magentic_team, _team_description


class MagenticRoutingTests(unittest.TestCase):
    def test_native_conversation_tools_are_attributed_to_their_holder(self):
        names = {"send_message_to_user", "get_reservation_details", "book_reservation"}
        text = _team_description(_magentic_team(names), names)
        websurfer = next(l for l in text.splitlines() if l.startswith("WebSurfer:"))
        self.assertIn("send_message_to_user", websurfer)
        self.assertIn("get_reservation_details", websurfer)
        # A participant without tools keeps its canonical blurb and gains no suffix. Saying
        # it "can only contribute text" told the orchestrator that Coder -- whose published
        # job is exactly to contribute code for Executor to run -- was useless: on
        # terminal-bench-2 that cost seven cases (58.6% -> 34.5%) and drove stalls per case
        # from 2 to 20. Naming the holder is the half that fixed tau2; the negative half is not.
        for role in ("FileSurfer", "Coder", "Executor"):
            line = next(l for l in text.splitlines() if l.startswith(role + ":"))
            self.assertNotIn("Available tools:", line)
            self.assertNotIn("send_message_to_user", line)

    def test_workspace_tools_stay_with_their_canonical_specialists(self):
        names = {"run_command", "read_file", "list_files"}
        text = _team_description(_magentic_team(names), names)
        lines = {l.split(":")[0]: l for l in text.splitlines()}
        self.assertIn("read_file", lines["FileSurfer"])
        self.assertIn("run_command", lines["Executor"])
        self.assertNotIn("Available tools:", lines["WebSurfer"])

    def test_topology_is_unchanged(self):
        for names in ({"send_message_to_user"}, {"run_command"}, set()):
            self.assertEqual(list(_magentic_team(names)),
                             ["FileSurfer", "WebSurfer", "Coder", "Executor"])

    def test_omitting_names_keeps_the_original_text(self):
        workers = _magentic_team(set())
        self.assertEqual(_team_description(workers),
                         "\n".join(f"{n}: {d}" for n, d in workers.items()))


if __name__ == "__main__":
    unittest.main()
