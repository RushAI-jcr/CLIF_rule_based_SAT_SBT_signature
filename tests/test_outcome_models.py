import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTCOME_MODELS_PATH = REPO_ROOT / "code" / "03_outcome_models.py"

spec = importlib.util.spec_from_file_location("outcome_models", OUTCOME_MODELS_PATH)
outcome_models = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(outcome_models)


class OutcomeModelsTests(unittest.TestCase):
    def test_extract_day_index_sorts_numerically(self) -> None:
        keys = ["h1_day_2", "h1_day_10", "h1_day_1"]
        parsed = [outcome_models._extract_day_index_from_key(k) for k in keys]
        ordered = [k for _, k in sorted(zip(parsed, keys), key=lambda x: x[0])]
        self.assertEqual(ordered, ["h1_day_1", "h1_day_2", "h1_day_10"])

    def test_landmark_vs_ever_exposure_definitions_differ(self) -> None:
        day_level_df = pd.DataFrame(
            {
                "hospitalization_id": ["A", "A", "B", "B", "C", "C"],
                "hosp_id_day_key": [
                    "A_day_1",
                    "A_day_2",
                    "B_day_1",
                    "B_day_2",
                    "C_day_1",
                    "C_day_2",
                ],
                "vent_day_index": [1, 2, 1, 2, 1, 2],
                "delivery": [0, 1, 0, 0, 1, 1],
                "age_at_admission": [65, 65, 72, 72, 59, 59],
                "sex_category": ["Male", "Male", "Female", "Female", "Male", "Male"],
                "race_category": ["White", "White", "Black", "Black", "White", "White"],
                "died": [1, 1, 0, 0, 0, 0],
                "total_vent_days": [2, 2, 2, 2, 2, 2],
                "hospital_id": ["H1", "H1", "H1", "H1", "H2", "H2"],
            }
        )

        landmark = outcome_models._build_hospitalization_level_dataset(
            day_level_df, exposure_col="landmark_delivered"
        )
        ever = outcome_models._build_hospitalization_level_dataset(
            day_level_df, exposure_col="ever_delivered"
        )

        landmark_risk = landmark.groupby("exposure")["died"].mean().to_dict()
        ever_risk = ever.groupby("exposure")["died"].mean().to_dict()

        self.assertNotEqual(landmark_risk, ever_risk)
        self.assertEqual(int(landmark.loc[landmark["hospitalization_id"] == "A", "landmark_delivered"].iloc[0]), 0)
        self.assertEqual(int(ever.loc[ever["hospitalization_id"] == "A", "ever_delivered"].iloc[0]), 1)

    def test_late_delivery_bias_can_flip_effect_direction(self) -> None:
        rows = []
        spec = [
            # Early delivered, high observed mortality.
            ("E1", [1, 1], 1),
            ("E2", [1, 1], 1),
            # Late delivered survivors (immortal-time risk if labeled exposed).
            ("L1", [0, 1], 0),
            ("L2", [0, 1], 0),
            # Never-delivered with poor outcomes.
            ("N1", [0, 0], 1),
            ("N2", [0, 0], 1),
        ]
        for hosp_id, deliveries, died in spec:
            for day_idx, delivery in enumerate(deliveries, start=1):
                rows.append(
                    {
                        "hospitalization_id": hosp_id,
                        "hosp_id_day_key": f"{hosp_id}_day_{day_idx}",
                        "vent_day_index": day_idx,
                        "delivery": delivery,
                        "age_at_admission": 65,
                        "sex_category": "Male",
                        "race_category": "White",
                        "died": died,
                        "total_vent_days": 2,
                        "hospital_id": "H1",
                    }
                )

        day_level_df = pd.DataFrame(rows)
        landmark = outcome_models._build_hospitalization_level_dataset(
            day_level_df, exposure_col="landmark_delivered"
        )
        ever = outcome_models._build_hospitalization_level_dataset(
            day_level_df, exposure_col="ever_delivered"
        )

        landmark_risk = landmark.groupby("exposure")["died"].mean().to_dict()
        ever_risk = ever.groupby("exposure")["died"].mean().to_dict()
        landmark_risk_diff = landmark_risk[1] - landmark_risk[0]
        ever_risk_diff = ever_risk[1] - ever_risk[0]

        self.assertGreater(landmark_risk_diff, 0.0)
        self.assertLess(ever_risk_diff, 0.0)

    def test_run_all_models_writes_metadata_columns(self) -> None:
        rows = [
            # hosp 1 (delivered late)
            {
                "hospitalization_id": "h1",
                "hosp_id_day_key": "h1_day_1",
                "hospital_id": "H1",
                "patient_id": "p1",
                "age_at_admission": 60,
                "sex_category": "Male",
                "race_category": "White",
                "ethnicity_category": "Non-Hispanic",
                "event_time": "2024-01-01T00:00:00",
                "admission_dttm": "2024-01-01T00:00:00",
                "discharge_dttm": "2024-01-10T00:00:00",
                "discharge_category": "Home",
                "eligible_event": 1,
                "eligible_day": 1,
                "SAT_EHR_delivery": 0,
                "SAT_modified_delivery": 0,
                "EHR_Delivery_2mins": 0,
                "EHR_Delivery_5mins": 0,
                "EHR_Delivery_30mins": 0,
                "fio2_set": 40,
                "peep_set": 8,
                "rass": -3,
                "spo2": 95,
                "norepinephrine": 0,
                "epinephrine": 0,
                "vasopressin": 0,
                "dopamine": 0,
                "phenylephrine": 0,
                "propofol": 10,
                "fentanyl": 5,
                "midazolam": 0,
                "ICU_LOS": 9,
            },
            {
                "hospitalization_id": "h1",
                "hosp_id_day_key": "h1_day_2",
                "hospital_id": "H1",
                "patient_id": "p1",
                "age_at_admission": 60,
                "sex_category": "Male",
                "race_category": "White",
                "ethnicity_category": "Non-Hispanic",
                "event_time": "2024-01-02T00:00:00",
                "admission_dttm": "2024-01-01T00:00:00",
                "discharge_dttm": "2024-01-10T00:00:00",
                "discharge_category": "Home",
                "eligible_event": 1,
                "eligible_day": 1,
                "SAT_EHR_delivery": 1,
                "SAT_modified_delivery": 1,
                "EHR_Delivery_2mins": 1,
                "EHR_Delivery_5mins": 1,
                "EHR_Delivery_30mins": 1,
                "fio2_set": 35,
                "peep_set": 6,
                "rass": -1,
                "spo2": 96,
                "norepinephrine": 0,
                "epinephrine": 0,
                "vasopressin": 0,
                "dopamine": 0,
                "phenylephrine": 0,
                "propofol": 8,
                "fentanyl": 3,
                "midazolam": 0,
                "ICU_LOS": 9,
            },
            # hosp 2 (never delivered, expired)
            {
                "hospitalization_id": "h2",
                "hosp_id_day_key": "h2_day_1",
                "hospital_id": "H2",
                "patient_id": "p2",
                "age_at_admission": 72,
                "sex_category": "Female",
                "race_category": "Black",
                "ethnicity_category": "Non-Hispanic",
                "event_time": "2024-01-01T00:00:00",
                "admission_dttm": "2024-01-01T00:00:00",
                "discharge_dttm": "2024-01-07T00:00:00",
                "discharge_category": "Expired",
                "eligible_event": 1,
                "eligible_day": 1,
                "SAT_EHR_delivery": 0,
                "SAT_modified_delivery": 0,
                "EHR_Delivery_2mins": 0,
                "EHR_Delivery_5mins": 0,
                "EHR_Delivery_30mins": 0,
                "fio2_set": 55,
                "peep_set": 10,
                "rass": -4,
                "spo2": 92,
                "norepinephrine": 2,
                "epinephrine": 0,
                "vasopressin": 0,
                "dopamine": 0,
                "phenylephrine": 0,
                "propofol": 12,
                "fentanyl": 6,
                "midazolam": 1,
                "ICU_LOS": 6,
            },
            {
                "hospitalization_id": "h2",
                "hosp_id_day_key": "h2_day_2",
                "hospital_id": "H2",
                "patient_id": "p2",
                "age_at_admission": 72,
                "sex_category": "Female",
                "race_category": "Black",
                "ethnicity_category": "Non-Hispanic",
                "event_time": "2024-01-02T00:00:00",
                "admission_dttm": "2024-01-01T00:00:00",
                "discharge_dttm": "2024-01-07T00:00:00",
                "discharge_category": "Expired",
                "eligible_event": 1,
                "eligible_day": 1,
                "SAT_EHR_delivery": 0,
                "SAT_modified_delivery": 0,
                "EHR_Delivery_2mins": 0,
                "EHR_Delivery_5mins": 0,
                "EHR_Delivery_30mins": 0,
                "fio2_set": 60,
                "peep_set": 10,
                "rass": -4,
                "spo2": 90,
                "norepinephrine": 2,
                "epinephrine": 0,
                "vasopressin": 0,
                "dopamine": 0,
                "phenylephrine": 0,
                "propofol": 12,
                "fentanyl": 6,
                "midazolam": 1,
                "ICU_LOS": 6,
            },
        ]

        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            sat_path = Path(tmpdir) / "sat.csv"
            sbt_path = Path(tmpdir) / "sbt.csv"
            out_dir = Path(tmpdir) / "out"
            df.to_csv(sat_path, index=False)
            df.to_csv(sbt_path, index=False)

            result_df = outcome_models.run_all_models(
                str(sat_path),
                str(sbt_path),
                str(out_dir),
                exposure_strategy="landmark_primary_with_ever_sensitivity",
                analysis_spec_version="test-spec-v1",
            )

            self.assertFalse(result_df.empty)
            for col in [
                "analysis_spec_version",
                "exposure_definition",
                "landmark_rule",
                "model_family",
            ]:
                self.assertIn(col, result_df.columns)
            self.assertIn("time_varying_cummax", set(result_df["exposure_definition"].dropna()))
            self.assertIn("landmark_primary", set(result_df["exposure_definition"].dropna()))
            self.assertIn("ever_delivery_sensitivity", set(result_df["exposure_definition"].dropna()))
            self.assertTrue((result_df["analysis_spec_version"] == "test-spec-v1").all())


if __name__ == "__main__":
    unittest.main()
