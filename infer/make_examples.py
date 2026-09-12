#!/usr/bin/env python3
"""Run the bundled OneWorld text and no-text examples."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--wan", required=True)
    parser.add_argument("--pi3", default="yyfz233/Pi3X")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--out", default="outputs/examples")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    examples = json.loads((root / "examples" / "inputs.json").read_text())
    for name, example in examples.items():
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.gpus}",
            str(root / "infer.py"),
            "--model",
            args.model,
            "--wan",
            args.wan,
            "--pi3",
            args.pi3,
            "--image",
            str(root / "examples" / example["image"]),
            "--cameras",
            str(root / "examples" / example["cameras"]),
            "--prompt",
            example.get("prompt", ""),
            "--seed",
            str(example["seed"]),
            "--out",
            str(Path(args.out) / name),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
