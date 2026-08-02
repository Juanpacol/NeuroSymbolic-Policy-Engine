"""Tests for nspe.bench.harness."""

import time

from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.bench.harness import benchmark


class TestBenchmark(TestCase):
    def test_reports_expected_fields_and_reasonable_timing(self):
        calls = {"n": 0}

        def fn() -> None:
            calls["n"] += 1
            time.sleep(0.001)

        stats = benchmark(fn, device="cpu", warmup=2, reps=5, batch_size=10)

        self.assertEqual(stats.device, "cpu")
        self.assertEqual(stats.batch_size, 10)
        self.assertEqual(stats.warmup, 2)
        self.assertEqual(stats.reps, 5)
        self.assertGreater(stats.median_ms, 0.0)
        self.assertGreaterEqual(stats.p99_ms, stats.median_ms - 1e-6)
        self.assertGreater(stats.throughput_per_sec, 0.0)
        # warmup + timed reps
        self.assertEqual(calls["n"], 2 + 5)

    def test_throughput_matches_batch_over_median(self):
        def fn() -> None:
            time.sleep(0.002)

        stats = benchmark(fn, device="cpu", warmup=1, reps=5, batch_size=4)
        expected = 4 / (stats.median_ms / 1000.0)
        self.assertAlmostEqual(stats.throughput_per_sec, expected, places=3)

    def test_per_item_median_is_median_over_batch(self):
        def fn() -> None:
            time.sleep(0.002)

        stats = benchmark(fn, device="cpu", warmup=1, reps=5, batch_size=4)

        self.assertAlmostEqual(stats.per_item_median_ms, stats.median_ms / 4, places=9)
        self.assertAlmostEqual(
            stats.per_item_median_ms, 1000.0 / stats.throughput_per_sec, places=6
        )

    def test_per_item_median_equals_median_at_batch_one(self):
        stats = benchmark(lambda: None, device="cpu", warmup=1, reps=3, batch_size=1)
        self.assertEqual(stats.per_item_median_ms, stats.median_ms)


if __name__ == "__main__":
    run_tests()
