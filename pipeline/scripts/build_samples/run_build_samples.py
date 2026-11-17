from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.sample_builders.pipeline import run_sample_build_pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = ArgumentParser(description="Phase 4 leakage-safe sample building pipeline.")
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=REPO_ROOT / "configs/paths/local.example.yaml",
        help="Project path configuration.",
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=REPO_ROOT / "configs/features/feature_sets_v1.yaml",
        help="Feature-set configuration.",
    )
    args = parser.parse_args()
    summary = run_sample_build_pipeline(
        paths_config_path=args.paths_config,
        feature_config_path=args.feature_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
