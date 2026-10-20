#!/usr/bin/env python3
"""Verify every file in a repaired release, without using a GPU."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneworld.rae.release import verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    manifest = verify_manifest(args.model)
    print(f"Verified {len(manifest['files'])} files; every SHA256 matches.")


if __name__ == "__main__":
    main()
