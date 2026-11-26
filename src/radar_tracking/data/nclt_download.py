#!/usr/bin/env python3

"""Download the raw NCLT files needed for the TBPTT KalmanNet workflow.

This helper targets the official NCLT hosting layout used by the University of
Michigan dataset page. It mirrors the minimal subset used by the sensor-fusion
pipeline described in:

  Practical Implementation of KalmanNet for Accurate Data Fusion in
  Integrated Navigation, Jian Song et al., IEEE SPL 2024.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BASE_URL = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"

NCLT_DATES = [
    "2012-01-08",
    "2012-01-15",
    "2012-01-22",
    "2012-02-02",
    "2012-02-04",
    "2012-02-05",
    "2012-02-12",
    "2012-02-18",
    "2012-02-19",
    "2012-03-17",
    "2012-03-25",
    "2012-03-31",
    "2012-04-29",
    "2012-05-11",
    "2012-05-26",
    "2012-06-15",
    "2012-08-04",
    "2012-08-20",
    "2012-09-28",
    "2012-10-28",
    "2012-11-04",
    "2012-11-16",
    "2012-11-17",
    "2012-12-01",
    "2013-01-10",
    "2013-02-23",
    "2013-04-05",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_wget(url: str, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    cmd = ["wget", "--continue", url, "-P", str(dest_dir)]
    print("Calling:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def selected_dates(date: str | None) -> list[str]:
    if date is None:
        return list(NCLT_DATES)
    if date not in NCLT_DATES:
        raise ValueError(f"Unsupported NCLT date: {date}")
    return [date]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NCLT raw files for the fusion benchmark.")
    parser.add_argument("--data-dir", default="./data/NCLT", help="Base output directory.")
    parser.add_argument("--date", help="Optional single session date, for example 2013-04-05.")
    parser.add_argument("--sensor", action="store_true", help="Download sensor tarballs.")
    parser.add_argument("--ground-truth", action="store_true", help="Download ground-truth CSV files.")
    parser.add_argument("--all", action="store_true", help="Download both sensor and ground-truth files.")
    args = parser.parse_args()

    if not (args.sensor or args.ground_truth or args.all):
        parser.error("Choose at least one of --sensor, --ground-truth, or --all.")

    data_dir = Path(args.data_dir)
    dates = selected_dates(args.date)

    for date in dates:
        if args.sensor or args.all:
            run_wget(
                f"{BASE_URL}/sensor_data/{date}_sen.tar.gz",
                data_dir / "sensor_data_tar",
            )
        if args.ground_truth or args.all:
            run_wget(
                f"{BASE_URL}/ground_truth/groundtruth_{date}.csv",
                data_dir / "ground_truth",
            )

    print("\nDownload step finished.")
    print("Next:")
    print(f"  mkdir -p {data_dir / 'sensor_data'}")
    print(f"  ls {data_dir / 'sensor_data_tar'}/*.tar.gz | xargs -I {{}} tar xzvf {{}} -C {data_dir / 'sensor_data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
