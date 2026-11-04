"""tools/concurrency_bench.py against the fake batch engine of serve/test_parallel.py: the levels run, every figure the
runbook reads is there (aggregate tok/s, TTFT, ITL, the server's per-request rows, the engine's BXP totals), and the
code check reports honestly on an engine that cannot read the prompt."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import concurrency_bench as cb  # noqa: E402
from serve.test_parallel import FakeEngineCase  # noqa: E402


class Percentile(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertIsNone(cb.percentile([], 50))
        self.assertEqual(cb.percentile([3], 95), 3)
        self.assertEqual(cb.percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertEqual(cb.percentile([5, 1, 4, 2, 3], 0), 1)
        self.assertEqual(cb.percentile([1, 2, 3, 4, 5], 100), 5)
        self.assertEqual(cb.percentile([1, 2, 3, 4, 5], 95), 5)


class Levels(FakeEngineCase):
    def test_levels_against_the_fake_engine(self):
        self.start(2, expert_sample=True)
        one = cb.run_level(self.base, 1, 120, 16, check_codes=True, label="fake")
        self.assertEqual(one["clients"], 1)
        self.assertEqual(one["errors"], [])
        self.assertGreater(one["tokens"], 0)
        self.assertGreater(one["aggregate_tok_s"], 0)
        self.assertIsNotNone(one["ttft_s"]["p50"])
        self.assertGreaterEqual(one["itl_ms"]["n"], 1)
        self.assertEqual(len(one["requests"]), 1)                  # the server's own row of this request
        self.assertEqual(one["requests"][0]["output_tokens"], one["tokens"])
        self.assertEqual(one["own_code_named"], "0 of 1")          # the fake engine answers "ok, done.": said, not hidden
        self.assertFalse(one["all_coherent"])                      # "ok, done." is two words: the check says so
        two = cb.run_level(self.base, 2, 120, 16, check_codes=False, label="fake")
        self.assertEqual(two["errors"], [])
        self.assertEqual(len(two["requests"]), 2)
        self.assertNotIn("own_code_named", two)
        self.assertIsNotNone(two["ttft_s"]["p95"])
        # the engine's BXP totals moved between the two /metrics readings of a level
        self.assertIsNotNone(two["expert_window"])
        self.assertGreaterEqual(two["expert_window"]["windows"], 1)
        self.assertEqual(two["expert_window"]["pool_ms_per_window"], 3.0)
        self.assertEqual(two["expert_window"]["wait_ms_per_window"], 1.5)
        self.assertGreater(two["expert_window"]["pcie_bytes_per_token"], 0)
        table = cb.summary_table([one, two])
        self.assertIn("| fake | 1 |", table)
        self.assertIn("| fake | 2 |", table)
        self.assertEqual(table.count("\n"), 3)


if __name__ == "__main__":
    unittest.main()
