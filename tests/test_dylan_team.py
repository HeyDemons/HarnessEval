import json
from pathlib import Path
import tempfile
import unittest

from benchmark_platform.harnesses.aflow import digest
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment
from benchmark_platform.harnesses.dylan_team import freeze_team, optimize_team, validate_team
from benchmark_platform.harnesses.methods import run_profile
from test_dylan_network import RecordingClient
from test_dylan_fidelity import Trace


SPLIT = {"benchmark": "synthetic", "optimization_case_ids": ["train-a", "train-b"],
         "evaluation_case_ids": ["test-a", "test-b"]}


def artifact():
    return freeze_team([{"case_id": "train-a", "importance": [3, 0, 1]},
                        {"case_id": "train-b", "importance": [0, 4, 1]}], SPLIT,
                       roles=["Assistant", "Programmer", "Mathematician"])


class FrozenTeamTests(unittest.IsolatedAsyncioTestCase):
    def test_aggregates_across_queries_and_freezes_one_team(self):
        team = artifact()
        self.assertEqual(team["importance_scores"], [1.5, 2.0, 1.0])
        self.assertEqual(team["selected_agents"], [1, 0])
        validate_team(team, benchmark="synthetic", case_id="test-a")
        validate_team(team, benchmark="synthetic", case_id="test-b")
        team["selected_agents"] = [0, 2]
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_team(team)
        team["artifact_sha256"] = digest({k: v for k, v in team.items() if k != "artifact_sha256"})
        with self.assertRaisesRegex(ValueError, "selection rule"):
            validate_team(team)

    async def test_heldout_query_only_runs_frozen_solve_network(self):
        trace = Trace()
        ctx = RunContext("dylan", "held-out public question", RecordingClient(["answer"] * 2),
                         ToolEnvironment([], trace), trace,
                         {"dylan_team_artifact": artifact(), "dylan_benchmark": "synthetic", "dylan_case_id": "test-a"})
        self.assertEqual(await run_profile(ctx), "answer")
        self.assertEqual(ctx.llm_calls, 2)
        nodes = [e for e in trace.events if e["event"] == "dylan_node"]
        self.assertEqual({e["agent"] for e in nodes}, {0, 1})
        self.assertEqual({e["phase"] for e in nodes}, {"solve"})
        self.assertFalse(any(e["event"] == "dylan_team_selected" for e in trace.events))

    async def test_missing_team_wrong_split_and_overrides_fail_before_model(self):
        base = {"dylan_team_artifact": artifact(), "dylan_benchmark": "synthetic", "dylan_case_id": "test-a"}
        for policy in ({}, {**base, "dylan_case_id": "train-a"}, {**base, "dylan_benchmark": "other"},
                       {**base, "dylan_team_optimization": True}):
            trace = Trace()
            ctx = RunContext("dylan", "question", RecordingClient([]), ToolEnvironment([], trace), trace, policy)
            with self.assertRaises(ValueError):
                await run_profile(ctx)
            self.assertEqual(ctx.llm_calls, 0)

    async def test_optimizer_uses_only_public_training_questions_and_saves_traces(self):
        client = RecordingClient(["training response"] * 6)
        cases = [{"id": case_id, "prompt": f"public {case_id}"} for case_id in SPLIT["optimization_case_ids"]]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "team"
            team = await optimize_team(client, cases, SPLIT, output, roles=["Assistant"] * 4)
            self.assertEqual(len(client.messages), 6)
            self.assertNotIn("test-a", str(client.messages))
            self.assertNotIn("training response", str(client.messages[3:]))
            self.assertEqual(team, json.loads((output / "team.json").read_text()))
            records = json.loads((output / "trials.json").read_text())
            self.assertEqual(team["trials_sha256"], digest(records))
            self.assertTrue((output / "trial-2.jsonl").is_file())

    async def test_optimizer_rejects_overlap_and_answer_fields_before_calls(self):
        client = RecordingClient([])
        cases = [{"id": case_id, "prompt": "question"} for case_id in SPLIT["optimization_case_ids"]]
        with tempfile.TemporaryDirectory() as directory:
            for split, data in (({**SPLIT, "evaluation_case_ids": ["train-a"]}, cases),
                                (SPLIT, [{**row, "answer": "not public"} for row in cases])):
                with self.assertRaises(ValueError):
                    await optimize_team(client, data, split, Path(directory) / "unused", roles=["Assistant"] * 4)
            self.assertFalse((Path(directory) / "unused").exists())
            self.assertEqual(client.messages, [])
