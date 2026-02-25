import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
COMP_RISK_PATH = REPO_ROOT / "utils" / "competing_risks.py"
MULTISTATE_PATH = REPO_ROOT / "utils" / "multistate.py"
META_ANALYSIS_PATH = REPO_ROOT / "utils" / "meta_analysis.py"


_spec_cr = importlib.util.spec_from_file_location("competing_risks", COMP_RISK_PATH)
competing_risks = importlib.util.module_from_spec(_spec_cr)
assert _spec_cr and _spec_cr.loader
sys.modules["competing_risks"] = competing_risks
_spec_cr.loader.exec_module(competing_risks)

_spec_ms = importlib.util.spec_from_file_location("multistate", MULTISTATE_PATH)
multistate = importlib.util.module_from_spec(_spec_ms)
assert _spec_ms and _spec_ms.loader
sys.modules["multistate"] = multistate
_spec_ms.loader.exec_module(multistate)

_spec_ma = importlib.util.spec_from_file_location("meta_analysis", META_ANALYSIS_PATH)
meta_analysis = importlib.util.module_from_spec(_spec_ma)
assert _spec_ma and _spec_ma.loader
sys.modules["meta_analysis"] = meta_analysis
_spec_ma.loader.exec_module(meta_analysis)


class CompetingRiskAndMultistateTests(unittest.TestCase):
    def _sample_day_level(self) -> pd.DataFrame:
        rows = []
        specs = [
            ("h1", "H1", 0, 0, 2, 1, 1),
            ("h2", "H1", 1, 1, 2, 0, 1),
            ("h3", "H2", 0, 1, 3, 0, 2),
            ("h4", "H2", 1, 1, 1, 1, 1),
        ]
        for hosp, hosp_site, d1, d2, total_days, died, n_eps in specs:
            for day_idx in range(1, total_days + 1):
                delivery = d1 if day_idx == 1 else d2
                ep_id = f"{hosp}_ep_1" if n_eps == 1 else (f"{hosp}_ep_1" if day_idx == 1 else f"{hosp}_ep_2")
                rows.append(
                    {
                        "hospitalization_id": hosp,
                        "hospital_id": hosp_site,
                        "hosp_id_day_key": f"{hosp}_day_{day_idx}",
                        "vent_day_index": day_idx,
                        "delivery": delivery,
                        "age_at_admission": 65,
                        "sex_category": "Male",
                        "race_category": "White",
                        "total_vent_days": total_days,
                        "died": died,
                        "imv_episode_id": ep_id,
                    }
                )
        return pd.DataFrame(rows)

    def test_fine_gray_equivalent_outputs(self) -> None:
        day_df = self._sample_day_level()
        result, cif_df, comp_df = competing_risks.fit_fine_gray_equivalent(
            day_df,
            exposure_col="landmark_delivered",
            horizon=28,
        )
        self.assertIn("estimator", result)
        self.assertEqual(result["estimator"], "discrete_time_subdistribution_cloglog")
        self.assertFalse(cif_df.empty)
        self.assertFalse(comp_df.empty)

    def test_multistate_outputs(self) -> None:
        day_df = self._sample_day_level()
        hazards, transitions = multistate.fit_multistate_equivalent(
            day_df,
            exposure_col="landmark_delivered",
            horizon=28,
        )
        self.assertFalse(hazards.empty)
        self.assertFalse(transitions.empty)
        self.assertIn("transition", hazards.columns)
        self.assertIn("transition", transitions.columns)

    def test_bca_cluster_bootstrap_contract(self) -> None:
        day_df = self._sample_day_level().copy()
        day_df["ehr_flag"] = day_df["delivery"]
        day_df["flowsheet_flag"] = (day_df["delivery"] | day_df["died"]).astype(int)

        result = meta_analysis.cluster_bootstrap_concordance(
            day_level_df=day_df,
            ehr_col="ehr_flag",
            flowsheet_col="flowsheet_flag",
            cluster_col="imv_episode_id",
            n_boot=200,
            seed=42,
        )
        self.assertEqual(result["ci_method"], "bca_cluster_bootstrap")
        self.assertIn(result["bootstrap_level"], {"episode_cluster", "hospitalization_cluster", "row_level"})
        self.assertEqual(result["cluster_col_used"], "imv_episode_id")
        self.assertIn("sensitivity", result)


if __name__ == "__main__":
    unittest.main()
