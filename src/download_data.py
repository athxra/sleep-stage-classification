"""
download_data.py
================
Download Sleep-EDF Expanded (Sleep-Cassette) PSG + Hypnogram pairs from
PhysioNet into data/raw/.

By default this fetches Sleep-EDF-20 (subjects 0-19, 39 recordings, ~1.9 GB),
the standard benchmark subset used by DeepSleepNet, TinySleepNet and AttnSleep,
so results are directly comparable with published work.

Usage:
    python src/download_data.py                 # subjects 0-19
    python src/download_data.py --subjects 5    # subjects 0-4 (quick run)
"""

import argparse
import os
import re
import sys
import time
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RAW_DIR, parse_recording

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
# PhysioNet's AWS Open Data mirror: same files, usually much faster.
MIRROR_URL = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/"
RETRIES = 20
FILE_PATTERN = re.compile(r"SC4\d{3}[A-Z0-9]{2}-(?:PSG|Hypnogram)\.edf")


MIRROR_LISTING = ("https://physionet-open.s3.amazonaws.com/"
                  "?list-type=2&prefix=sleep-edfx/1.0.0/sleep-cassette/")


def list_remote_files():
    """File names from the S3 mirror's bucket listing, else PhysioNet's index page."""
    for url in (MIRROR_LISTING, BASE_URL):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            return sorted(set(FILE_PATTERN.findall(text)))
        except OSError as exc:
            print(f"  Listing failed for {url}: {exc}")
    raise SystemExit("Could not list Sleep-EDF files from PhysioNet.")


def fetch(url: str, tmp: str) -> None:
    """Download url into tmp, resuming from tmp's current size if it exists."""
    offset = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request) as resp:
        mode = "ab" if offset and resp.status == 206 else "wb"
        with open(tmp, mode) as out:
            while chunk := resp.read(1 << 16):
                out.write(chunk)


def download(filename: str) -> None:
    dest = os.path.join(RAW_DIR, filename)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"  [skip] {filename}")
        return
    tmp = dest + ".part"
    for attempt in range(1, RETRIES + 1):
        url = (MIRROR_URL if attempt % 2 else BASE_URL) + filename
        try:
            fetch(url, tmp)
            break
        except OSError as exc:
            print(f"  [retry {attempt}/{RETRIES}] {filename}: {exc}")
            time.sleep(min(60, 5 * attempt))
    else:
        print(f"  [FAILED] {filename} (rerun the script to resume)")
        return
    os.replace(tmp, dest)
    print(f"  [done] {filename}  ({os.path.getsize(dest) / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", type=int, default=20,
                        help="download subjects 0..N-1 (default: 20)")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel downloads (default: 8)")
    args = parser.parse_args()

    socket.setdefaulttimeout(120)
    os.makedirs(RAW_DIR, exist_ok=True)
    files = [f for f in list_remote_files() if parse_recording(f)[0] < args.subjects]
    print(f"Downloading {len(files)} files to {RAW_DIR}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, files))
    print("Download complete.")


if __name__ == "__main__":
    main()
