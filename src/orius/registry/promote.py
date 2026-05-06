"""Model registry: promote."""

from __future__ import annotations

import argparse

from .model_store import promote


def main():
    # Key: manage model artifacts and promotion logic
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--prod", required=True)
    args = p.parse_args()
    promote(args.candidate, args.prod)


if __name__ == "__main__":
    main()
