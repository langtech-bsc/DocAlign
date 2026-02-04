#!/usr/bin/env python3
"""Command-line interface for document alignment pipeline."""

import yaml
import argparse
from src.align.thresholds import resolve_thresholds
from src.pipeline.runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Align parallel documents")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_mode = config.get("data_mode", "parallel")
    threshold_overrides = config.get("thresholds", {})
    thresholds = resolve_thresholds(data_mode, threshold_overrides)

    print(f"Data mode: {data_mode}")
    print(f"Thresholds: {thresholds}")

    run_pipeline(
        input_dir=config["input"],
        input_format=config["input_format"],
        output_dir=config["output"],
        model=config.get("model", "labse"),
        model_path=config.get("model_path"),
        dtw=config.get("dtw", "approx"),
        input_type=config.get("input_type", "paragraph"),
        thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
