#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timedelta
import argparse
import fcntl
import os
import subprocess

HOME = Path.home()
VIDEO = HOME / "wildlife/video"
MANIFEST = HOME / "wildlife/archived_mp4.txt"
LOCK = HOME / "wildlife/video-conversion.lock"

parser = argparse.ArgumentParser()
parser.add_argument("--delete", action="store_true")
args = parser.parse_args()

archived = set(MANIFEST.read_text().splitlines())
cutoff = (datetime.now() - timedelta(hours=2)).timestamp()

candidates = []

for h264 in sorted(VIDEO.glob("*_bird.h264")):

    if h264.with_suffix(".mp4").name not in archived:
        continue

    if h264.stat().st_mtime >= cutoff:
        continue

    candidates.append(h264)

print(f"Candidates: {len(candidates)}")
print(
    f"Total size: "
    f"{sum(f.stat().st_size for f in candidates) / 1e9:.2f} GB"
)

if not args.delete:
    print("DRY RUN — no files deleted")
    raise SystemExit(0)

with LOCK.open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)

    for h264 in candidates:

        if not h264.exists():
            continue

        # Recheck eligibility while holding the lock.
        if h264.with_suffix(".mp4").name not in archived:
            continue

        if h264.stat().st_mtime >= cutoff:
            continue

        # Check whether any process has the file open.
        result = subprocess.run(
            ["fuser", str(h264)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 1:
            print(f"Skipping (open or cannot verify): {h264.name}")
            continue

        print(f"Deleting: {h264.name}")
        h264.unlink()

print("Cleanup complete")
