#!/usr/bin/env python3
"""Download Zenodo record 582636.

Zenodo 582636 is an X-ray microtomography volume of an assembly of rice
grains (689 TIFF slices, ~3.93 GB).  It is unrelated to neurons but is a
useful **3D instance-segmentation benchmark**: densely-packed,
touching, anisotropic-ish objects whose individual labels are easy to
verify by eye.  Within brainbow it is occasionally used as a
non-connectomics smoke test for the data-loading + clustering pipeline.

This script is also a good template for any other Zenodo record:
change ``RECORD_ID`` and re-run.

Features:
  * parallel downloads with a progress tracker,
  * MD5 checksum verification,
  * resumable per-file downloads (existing files are skipped if their
    size and -- with verify -- their checksum match).
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

RECORD_ID = 582636
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
FILES_API_URL = f"{API_URL}/files"
ARCHIVE_URL = f"{API_URL}/files-archive"


def fetch_file_list():
    """Fetch the list of files from the Zenodo API."""
    print(f"Fetching file list from {FILES_API_URL} ...")
    req = urllib.request.Request(FILES_API_URL)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    entries = data.get("entries", data) if isinstance(data, dict) else data
    return entries


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(entry: dict, dest_dir: Path, verify: bool) -> tuple[str, str]:
    """Download a single file. Returns (filename, status)."""
    key = entry["key"]
    url = entry["links"]["self"]
    if not url.endswith("/content"):
        url += "/content"
    size = entry["size"]
    checksum = entry.get("checksum", "")
    expected_md5 = checksum.split(":")[-1] if checksum.startswith("md5:") else None

    dest = dest_dir / key

    if dest.exists() and dest.stat().st_size == size:
        if verify and expected_md5:
            if md5_file(dest) == expected_md5:
                return key, "skipped (already exists, checksum OK)"
            # checksum mismatch — re-download
        else:
            return key, "skipped (already exists)"

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        if verify and expected_md5:
            actual = md5_file(tmp)
            if actual != expected_md5:
                tmp.unlink(missing_ok=True)
                return key, f"FAILED (checksum mismatch: expected {expected_md5}, got {actual})"
        tmp.rename(dest)
        return key, "downloaded"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return key, f"FAILED ({e})"


class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.downloaded = 0
        self.skipped = 0
        self.failed = 0
        self._lock = Lock()

    def update(self, status: str):
        with self._lock:
            self.done += 1
            if "downloaded" in status:
                self.downloaded += 1
            elif "skipped" in status:
                self.skipped += 1
            else:
                self.failed += 1
            pct = self.done / self.total * 100
            sys.stdout.write(
                f"\r[{self.done}/{self.total}] {pct:5.1f}%  "
                f"(dl: {self.downloaded}  skip: {self.skipped}  fail: {self.failed})"
            )
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output-dir",
        default=f"zenodo_{RECORD_ID}",
        help="Destination directory (default: zenodo_582636)",
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=4,
        help="Number of parallel downloads (default: 4)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip MD5 checksum verification",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Download as a single zip archive instead of individual files",
    )
    args = parser.parse_args()

    dest_dir = Path(args.output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if args.archive:
        zip_path = dest_dir / f"zenodo_{RECORD_ID}.zip"
        print(f"Downloading full archive to {zip_path} ...")
        urllib.request.urlretrieve(ARCHIVE_URL, zip_path)
        print(f"Done — saved to {zip_path}")
        return

    entries = fetch_file_list()
    total = len(entries)
    print(f"Found {total} files. Downloading to {dest_dir}/ with {args.workers} workers ...\n")

    tracker = ProgressTracker(total)
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, entry, dest_dir, not args.no_verify): entry["key"]
            for entry in entries
        }
        for future in as_completed(futures):
            key, status = future.result()
            tracker.update(status)
            if "FAILED" in status:
                failures.append((key, status))

    print()
    print(f"\nDone. {tracker.downloaded} downloaded, {tracker.skipped} skipped, {tracker.failed} failed.")
    if failures:
        print("\nFailed files:")
        for key, status in failures:
            print(f"  {key}: {status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
