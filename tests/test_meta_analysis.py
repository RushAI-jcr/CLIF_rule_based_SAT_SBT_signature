import unittest
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
META_ANALYSIS_PATH = REPO_ROOT / "utils" / "meta_analysis.py"

spec = importlib.util.spec_from_file_location("meta_analysis", META_ANALYSIS_PATH)
meta_analysis = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(meta_analysis)


class MetaAnalysisTests(unittest.TestCase):
    def test_pool_medians_iqr_uses_weighted_quantiles(self) -> None:
        df = pd.DataFrame(
            {
                "median": [1.0, 10.0],
                "iqr_low": [0.5, 9.0],
                "iqr_high": [2.0, 12.0],
                "n": [1, 9],
            }
        )
        n_total, pooled_median, pooled_q1, pooled_q3 = meta_analysis.pool_medians_iqr(
            df, "median", "iqr_low", "iqr_high", "n"
        )

        self.assertEqual(n_total, 10)
        self.assertGreaterEqual(pooled_median, 9.0)
        self.assertGreaterEqual(pooled_q1, 8.0)
        self.assertGreaterEqual(pooled_q3, 10.0)

    def test_pool_medians_iqr_single_row(self) -> None:
        df = pd.DataFrame(
            {
                "median": [7.0],
                "iqr_low": [6.0],
                "iqr_high": [9.0],
                "n": [50],
            }
        )
        n_total, pooled_median, pooled_q1, pooled_q3 = meta_analysis.pool_medians_iqr(
            df, "median", "iqr_low", "iqr_high", "n"
        )

        self.assertEqual(n_total, 50)
        self.assertEqual(pooled_median, 7.0)
        self.assertEqual(pooled_q1, 6.0)
        self.assertEqual(pooled_q3, 9.0)

    def test_logit_transform_handles_boundary_proportions(self) -> None:
        p = np.array([0.0, 1.0, 0.5])
        n = np.array([100, 100, 100])
        logit, se = meta_analysis.logit_transform(p, n)

        self.assertTrue(np.isfinite(logit).all())
        self.assertTrue(np.isfinite(se).all())


if __name__ == "__main__":
    unittest.main()
