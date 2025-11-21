#!/usr/bin/env python3
"""Command-line interface for document alignment pipeline."""

import yaml
import argparse
from src.pipeline.runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Align parallel documents")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    run_pipeline(
        input_dir=config["input"],
        output_dir=config["output"],
        model=config.get("model", "labse"),
        model_path=config.get("model_path"),
        dtw=config.get("dtw", "approx")
    )


if __name__ == "__main__":
    main()