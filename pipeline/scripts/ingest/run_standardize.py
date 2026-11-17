from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.integration.pipeline import run_standardization_pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = ArgumentParser(description="Phase 3 standardization and integration pipeline.")
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=REPO_ROOT / "configs/paths/local.example.yaml",
        help="Project path configuration.",
    )
    parser.add_argument(
        "--clinical-schema",
        type=Path,
        default=REPO_ROOT / "configs/schemas/clinical_table.yaml",
        help="Clinical schema configuration.",
    )
    parser.add_argument(
        "--monitoring-schema",
        type=Path,
        default=REPO_ROOT / "configs/schemas/monitoring_table.yaml",
        help="Monitoring schema configuration.",
    )
    parser.add_argument(
        "--cohort-config",
        type=Path,
        default=REPO_ROOT / "configs/cohorts/v1_long_protocol.yaml",
        help="Cohort inclusion configuration.",
    )
    args = parser.parse_args()
    summary = run_standardization_pipeline(
        paths_config_path=args.paths_config,
        clinical_schema_path=args.clinical_schema,
        monitoring_schema_path=args.monitoring_schema,
        cohort_config_path=args.cohort_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
