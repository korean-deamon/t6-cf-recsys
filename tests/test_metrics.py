"""Tests for the metrics."""
import unittest

from src.metrics import (average_precision_at_k, catalog_coverage, dcg_at_k,
                         hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k)


class TestMetrics(unittest.TestCase):
    def test_perfect_ranking(self):
        self.assertAlmostEqual(ndcg_at_k([1, 2, 3], {1, 2, 3}, 10), 1.0)

    def test_empty_relevant(self):
        self.assertEqual(ndcg_at_k([1, 2], set(), 10), 0.0)

    def test_no_hits(self):
        self.assertEqual(ndcg_at_k([4, 5], {1, 2}, 10), 0.0)

    def test_order_matters(self):
        early = ndcg_at_k([1, 9, 9], {1}, 10)
        late = ndcg_at_k([9, 9, 1], {1}, 10)
        self.assertGreater(early, late)

    def test_dcg_first_position(self):
        self.assertAlmostEqual(dcg_at_k([1], {1}, 10), 1.0)

    def test_precision_recall(self):
        recs, rel = [1, 7, 9, 2], {1, 2}
        self.assertAlmostEqual(precision_at_k(recs, rel, 10), 0.2)
        self.assertAlmostEqual(recall_at_k(recs, rel, 10), 1.0)

    def test_k_cutoff(self):
        self.assertAlmostEqual(precision_at_k([9, 9, 1], {1}, 2), 0.0)

    def test_hit_and_ap(self):
        self.assertEqual(hit_rate_at_k([1, 2], {2}, 10), 1.0)
        # found at position 2 -> AP = (1/2) / min(1, 10) = 0.5
        self.assertAlmostEqual(average_precision_at_k([9, 2], {2}, 10), 0.5)

    def test_coverage(self):
        self.assertAlmostEqual(catalog_coverage([[1, 2], [3, 4]], 8), 0.5)
        self.assertEqual(catalog_coverage([], 8), 0.0)


if __name__ == "__main__":
    unittest.main()
