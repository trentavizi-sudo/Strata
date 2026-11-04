"""The engine's BXP expert-window samples (STRATA_EXPERT_TELEMETRY=1): the parser, the /metrics view and the engine
environment switch.  Pure functions; the fake-engine round trip is in serve/test_parallel.py."""
import time
import unittest

from serve.server import (admission_policy, configure_expert_telemetry_env, expert_window_view,
                          parse_expert_window_sample)


class Parse(unittest.TestCase):
    def test_a_good_line(self):
        s = parse_expert_window_sample("BXP 2 4 1000 1500 500 150 30 8 22 40 3\n")
        self.assertEqual(s, {"windows": 2, "rows": 4, "elapsed_ms": 1000, "pcie_bytes": 1500, "pool_us": 500,
                             "wait_us": 150, "link_est_us": 30, "misses": 8, "hits": 22, "entries": 40, "offload": 3})

    def test_zero_rows_and_zero_counters_are_fine(self):
        s = parse_expert_window_sample("BXP 1 0 1 0 0 0 0 0 0 0 0")
        self.assertEqual(s["windows"], 1)
        self.assertEqual(s["entries"], 0)

    def test_bad_lines(self):
        for line in ("BXP 0 4 1000 1500 500 150 30 8 22 40 3",        # no window
                     "BXP 2 4 0 1500 500 150 30 8 22 40 3",           # no span
                     "BXP 2 4 1000 1500 500 150 30 8 22 40",          # a field short
                     "BXP 2 4 1000 1500 500 150 30 8 22 40 3 9",      # a field long
                     "BXP 2 4 1000 1500 500 150 30 8 50 40 3",        # more hits than entries
                     "BXP 2 4 1000 1500 500 150 30 8 22 40 41",       # more offload than entries
                     "BXP 2 4 1000 -1 500 150 30 8 22 40 3",          # negative
                     "BXP 2 4 90000000 1500 500 150 30 8 22 40 3",    # a day
                     "BSTAT 1 2 2 20", "BXP " + "9" * 300):
            self.assertIsNone(parse_expert_window_sample(line), line)


class View(unittest.TestCase):
    def test_rates_over_the_recent_samples(self):
        now = time.monotonic()
        rows = [{"windows": 2, "rows": 4, "elapsed_ms": 100, "pcie_bytes": 2000, "pool_us": 6000, "wait_us": 2000,
                 "link_est_us": 8000, "misses": 10, "hits": 30, "entries": 40, "offload": 2, "observed_monotonic": now - 1},
                {"windows": 2, "rows": 4, "elapsed_ms": 100, "pcie_bytes": 0, "pool_us": 2000, "wait_us": 2000,
                 "link_est_us": 0, "misses": 2, "hits": 38, "entries": 40, "offload": 0, "observed_monotonic": now}]
        v = expert_window_view(rows)
        self.assertEqual(v["windows"], 4)
        self.assertEqual(v["pool_ms_window"], 2.0)
        self.assertEqual(v["wait_ms_window"], 1.0)
        self.assertEqual(v["link_est_ms_window"], 2.0)
        self.assertEqual(v["miss_cost_ms_window"], 3.0)      # max(2.0, 2.0) + 1.0
        self.assertEqual(v["pcie_bytes_window"], 500.0)
        self.assertEqual(v["pcie_bytes_row"], 250.0)
        self.assertEqual(v["hit_rate"], 0.85)
        self.assertEqual(v["offload_share"], 0.025)
        self.assertEqual(v["ms_per_window"], 50.0)
        self.assertEqual(v["last"]["misses"], 2)
        self.assertIsNone(v["total"])
        self.assertEqual(expert_window_view(rows, {"windows": 7, "pcie_bytes": 9})["total"]["pcie_bytes"], 9)
        self.assertEqual(expert_window_view(rows, {"windows": 7})["total"]["hits"], 0)

    def test_old_samples_fall_out_of_the_rates(self):
        now = time.monotonic()
        old = {"windows": 10, "rows": 10, "elapsed_ms": 1000, "pcie_bytes": 0, "pool_us": 90000, "wait_us": 0,
               "link_est_us": 0, "misses": 0, "hits": 0, "entries": 10, "offload": 0, "observed_monotonic": now - 30}
        new = dict(old, windows=1, rows=1, elapsed_ms=10, pool_us=1000, observed_monotonic=now)
        v = expert_window_view([old, new])
        self.assertEqual(v["samples"], 1)
        self.assertEqual(v["pool_ms_window"], 1.0)
        self.assertIsNone(expert_window_view([]))


class Env(unittest.TestCase):
    def test_only_an_explicit_setting_changes_the_environment(self):
        env = {"STRATA_EXPERT_TELEMETRY": "1"}
        configure_expert_telemetry_env(env, {})
        self.assertEqual(env["STRATA_EXPERT_TELEMETRY"], "1")        # inherited: kept
        configure_expert_telemetry_env(env, {"expert_telemetry": False})
        self.assertNotIn("STRATA_EXPERT_TELEMETRY", env)
        configure_expert_telemetry_env(env, {"expert_telemetry": True})
        self.assertEqual(env["STRATA_EXPERT_TELEMETRY"], "1")

    def test_an_admission_policy_asks_for_it(self):
        env = {}
        configure_expert_telemetry_env(env, {"admission": "pressure"})
        self.assertEqual(env["STRATA_EXPERT_TELEMETRY"], "1")
        env = {}
        configure_expert_telemetry_env(env, {"admission": {"policy": "pressure", "threshold": 0.5}})
        self.assertEqual(env["STRATA_EXPERT_TELEMETRY"], "1")
        env = {}
        configure_expert_telemetry_env(env, {"admission": "pressure", "expert_telemetry": False})   # explicit wins
        self.assertNotIn("STRATA_EXPERT_TELEMETRY", env)

    def test_admission_policy_forms(self):
        self.assertIsNone(admission_policy({}))
        self.assertIsNone(admission_policy({"admission": None}))
        self.assertIsNone(admission_policy({"admission": False}))
        self.assertIsNone(admission_policy({"admission": {}}))
        self.assertEqual(admission_policy({"admission": "pressure"}), {"policy": "pressure"})
        self.assertEqual(admission_policy({"admission": {"policy": "pressure", "threshold": 0.4}}),
                         {"policy": "pressure", "threshold": 0.4})


if __name__ == "__main__":
    unittest.main()
