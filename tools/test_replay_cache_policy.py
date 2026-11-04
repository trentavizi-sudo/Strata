"""tools/replay_cache_policy.py's lfru rule (the offline twin of generate.cpp's --adapt-policy lfru) and the synthetic
trace writer.  `python -m pytest tools/test_replay_cache_policy.py`."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_cache_policy as rp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Lfru(unittest.TestCase):
    def layer_state(self):
        res = np.zeros((rp.NL, rp.NE), bool)
        usage = np.zeros((rp.NL, rp.NE))
        last = np.zeros((rp.NL, rp.NE), dtype=np.int64)
        return res, usage, last

    def test_a_stale_resident_is_the_victim_before_a_fresh_one_of_the_same_count(self):
        res, usage, last = self.layer_state()
        res[0, [1, 2]] = True
        usage[0, [1, 2]] = 10.0
        last[0, 1], last[0, 2] = 100, 20          # 1 routed just now, 2 not for 80 windows
        usage[0, 7] = 9.0                         # a candidate: below both counts, so lfu swaps nothing
        last[0, 7] = 100
        res_lfu = res.copy()
        n = rp.adapt_round(res_lfu, usage.copy(), rp.Policy(policy='lfu'))
        self.assertEqual(n, 0)
        n = rp.adapt_round(res, usage.copy(), rp.Policy(policy='lfru', recent=16), last, now=100)
        self.assertEqual(n, 1)
        self.assertTrue(res[0, 7] and res[0, 1] and not res[0, 2])   # 2's faded count (10 * 2^-5) lost to 7

    def test_a_candidate_not_routed_recently_is_not_a_candidate(self):
        res, usage, last = self.layer_state()
        res[0, 1] = True
        usage[0, 1] = 1.0
        last[0, 1] = 100
        usage[0, 7] = 50.0                        # hot a while ago, not within `recent`
        last[0, 7] = 60
        n = rp.adapt_round(res, usage.copy(), rp.Policy(policy='lfru', recent=16), last, now=100)
        self.assertEqual(n, 0)
        last[0, 7] = 90                           # routed 10 windows ago: a candidate
        n = rp.adapt_round(res, usage.copy(), rp.Policy(policy='lfru', recent=16), last, now=100)
        self.assertEqual(n, 1)
        self.assertTrue(res[0, 7] and not res[0, 1])

    def test_without_the_table_lfru_is_lfu(self):
        res, usage, _ = self.layer_state()
        res[0, 1] = True
        usage[0, 1], usage[0, 7] = 1.0, 5.0
        a, b = res.copy(), res.copy()
        rp.adapt_round(a, usage.copy(), rp.Policy(policy='lfu'))
        rp.adapt_round(b, usage.copy(), rp.Policy(policy='lfru'), None, 5)
        self.assertTrue((a == b).all())


class Synthetic(unittest.TestCase):
    def test_the_writer_makes_complete_windows_the_replay_reads(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.bin"
            subprocess.run([sys.executable, str(ROOT / "tools" / "synth_routing_trace.py"), str(out), "--windows", "12",
                            "--rows", "2", "--phases", "2", "--hubs", "4", "--seed", "3"], check=True, capture_output=True)
            windows, k = rp.load_trace(str(out))
            self.assertEqual(k, 10)
            self.assertEqual(len(windows), 12)
            self.assertEqual(len(windows[0]), rp.NL)
            layer, ids = windows[0][5]
            self.assertEqual(layer, 5)
            self.assertEqual(ids.shape, (2, 10))
            self.assertTrue((ids >= 0).all() and (ids < rp.NE).all())
            init = rp.trace_profile(windows, 100)
            self.assertEqual(init.shape, (100, 2))
            r = rp.simulate(windows, init, rp.Policy(policy='lfru', every=2, swaps=8))
            self.assertEqual(r["windows"], 12)
            self.assertGreaterEqual(r["share"], 0.0)


if __name__ == "__main__":
    unittest.main()
