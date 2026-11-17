from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.sample_builders.successful_cycle import build_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build auditable Kong-aligned and project composite successful-cycle labels."
    )
    parser.add_argument(
        "--config",
        default="configs/targets/successful_cycle_v1.yaml",
    )
    args = parser.parse_args()
    result = build_from_config(args.config)
    print(
        json.dumps(
            {
                "cycle_labels": str(result.cycle_labels_path),
                "formal_cohort_cycles": result.formal_cohort_cycles,
                "trigger_cycles": result.trigger_cycles,
                "exact_evaluable_cycles": result.exact_evaluable_cycles,
                "exact_positive_cycles": result.exact_positive_cycles,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
