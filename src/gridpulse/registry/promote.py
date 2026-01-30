from __future__ import annotations
import argparse
from .model_store import promote

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--prod", required=True)
    args = p.parse_args()
    out = promote(args.candidate, args.prod)
    print("Promoted to:", out)

if __name__ == "__main__":
    main()
