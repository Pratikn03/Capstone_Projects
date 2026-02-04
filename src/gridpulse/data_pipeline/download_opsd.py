"""Download OPSD time series dataset (Germany load/wind/solar).

Default downloads the hourly (60min) singleindex CSV from OPSD.
You can change version/date via --version.

Example:
  python -m gridpulse.data_pipeline.download_opsd --out data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from gridpulse.utils.logging import get_logger
from gridpulse.utils.net import get_session

DEFAULT_VERSION = "2020-10-06"
DEFAULT_FILE = "time_series_60min_singleindex.csv"

def download(url: str, out_path: Path, chunk: int = 1024 * 1024, *, retries: int = 3, backoff: float = 0.5) -> None:
    # Key: normalize inputs and build time-aware features
    log = get_logger("gridpulse.download_opsd")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session = get_session(retries=retries, backoff=backoff)
    with session.get(url, stream=True, timeout=120) as r:
        try:
            r.raise_for_status()
        except Exception as exc:
            log.error("Failed to download OPSD file from %s", url, exc_info=exc)
            raise
        total = int(r.headers.get("content-length", "0"))
        done = 0
        with open(out_path, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
                    done += len(part)
                    if total > 0:
                        pct = (done / total) * 100
                        sys.stdout.write(f"\rDownloading... {pct:6.2f}%")
                        sys.stdout.flush()
    if total > 0:
        print()
    log.info("Saved: %s", out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw", help="Output directory")
    p.add_argument("--version", default=DEFAULT_VERSION, help="OPSD release folder (e.g. 2020-10-06)")
    p.add_argument("--file", default=DEFAULT_FILE, help="Filename to download")
    p.add_argument("--retries", type=int, default=3, help="HTTP retry attempts")
    p.add_argument("--backoff", type=float, default=0.5, help="Retry backoff factor (seconds)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://data.open-power-system-data.org/time_series/{args.version}/{args.file}"
    out_path = out_dir / args.file
    log = get_logger("gridpulse.download_opsd")
    log.info("URL: %s", url)
    download(url, out_path, retries=args.retries, backoff=args.backoff)

    # create hint file
    (out_dir / "README_DOWNLOAD.md").write_text(
        f"""Downloaded OPSD file:
- {args.file}

If you already downloaded manually, ensure this file exists:
- data/raw/time_series_60min_singleindex.csv
""",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
