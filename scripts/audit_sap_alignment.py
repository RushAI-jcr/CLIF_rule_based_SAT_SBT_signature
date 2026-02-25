#!/usr/bin/env python3
"""Audit SAP alignment using traceability artifacts and interface checks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _parse_registry_ids(registry_path: Path) -> list[str]:
    ids: list[str] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:"):
            ids.append(stripped.split(":", 1)[1].strip().strip('"'))
    return ids


def _load_traceability(traceability_path: Path) -> list[dict[str, str]]:
    with traceability_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _check_interface_markers(repo_root: Path) -> list[str]:
    errors: list[str] = []

    outcome_file = repo_root / "code" / "03_outcome_models.py"
    outcome_text = outcome_file.read_text(encoding="utf-8")
    required_outcome_markers = [
        "sap_requirement_id",
        "analysis_role",
        "prespec_status",
        "model_estimator",
        "ph_assumption_status",
        "competing_risk_method",
        "construct_validity_vfd_components.csv",
        "construct_validity_cif_curves.csv",
        "construct_validity_multistate_transitions.csv",
        "--vfd-primary-method",
        "--include-vfd-sensitivities",
        "--disable-multiplicity",
    ]
    for marker in required_outcome_markers:
        if marker not in outcome_text:
            errors.append(f"missing marker in 03_outcome_models.py: {marker}")

    agg_file = repo_root / "code" / "07_aggregate_sites.py"
    agg_text = agg_file.read_text(encoding="utf-8")
    required_agg_markers = [
        "--concordance-meta-method",
        "--require-sap-full-pass",
        "ci_method",
        "bootstrap_level",
        "cluster_col_used",
    ]
    for marker in required_agg_markers:
        if marker not in agg_text:
            errors.append(f"missing marker in 07_aggregate_sites.py: {marker}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SAP conformance traceability")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root containing docs/ and code/",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero if any status is not 'match'",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    registry_path = repo_root / "docs" / "sap_requirement_registry.yaml"
    traceability_path = repo_root / "docs" / "sap_code_traceability.csv"

    if not registry_path.exists():
        print(f"ERROR: missing registry: {registry_path}")
        return 2
    if not traceability_path.exists():
        print(f"ERROR: missing traceability CSV: {traceability_path}")
        return 2

    registry_ids = _parse_registry_ids(registry_path)
    rows = _load_traceability(traceability_path)
    csv_ids = [r.get("requirement_id", "") for r in rows]

    missing = sorted(set(registry_ids) - set(csv_ids))
    extras = sorted(set(csv_ids) - set(registry_ids))

    statuses = {"match": 0, "partial": 0, "mismatch": 0, "other": 0}
    for row in rows:
        status = (row.get("status") or "").strip().lower()
        if status in statuses:
            statuses[status] += 1
        else:
            statuses["other"] += 1

    print("SAP Traceability Audit")
    print("=" * 60)
    print(f"Registry requirements: {len(registry_ids)}")
    print(f"Traceability rows:     {len(rows)}")
    print(f"match={statuses['match']} partial={statuses['partial']} mismatch={statuses['mismatch']} other={statuses['other']}")

    if missing:
        print("Missing requirement IDs in CSV:")
        for req in missing:
            print(f"  - {req}")
    if extras:
        print("Unknown requirement IDs present in CSV:")
        for req in extras:
            print(f"  - {req}")

    marker_errors = _check_interface_markers(repo_root)
    if marker_errors:
        print("Interface marker failures:")
        for err in marker_errors:
            print(f"  - {err}")

    fail = False
    if missing or extras or marker_errors or statuses["mismatch"] > 0 or statuses["other"] > 0:
        fail = True
    if args.fail_on_mismatch and statuses["partial"] > 0:
        fail = True

    if fail:
        print("RESULT: FAIL")
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
