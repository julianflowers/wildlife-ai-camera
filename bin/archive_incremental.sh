#!/usr/bin/env bash
set -euo pipefail

SOURCE="$HOME/wildlife/video"
REMOTE="gdrive:Wildlife Cameras/IMX500/video"
MANIFEST="$HOME/wildlife/archived_mp4.txt"
LOG_DIR="$HOME/archive-logs"

mkdir -p "$LOG_DIR"
touch "$MANIFEST"

# Prevent concurrent archive jobs
exec 9>"$LOG_DIR/archive.lock"
flock -n 9 || {
    echo "Another archive process is running."
    exit 1
}


# Select normal completed MP4s by recording timestamp.
# Process oldest first; exclude recordings under 24 hours old.
mapfile -t FILES < <(
    python3 - "$SOURCE" <<'PY'
from pathlib import Path
from datetime import datetime, timedelta
import sys

source = Path(sys.argv[1])
cutoff = datetime.now() - timedelta(hours=2)

files = []

for mp4 in source.glob("*_bird.mp4"):
    try:
        recorded = datetime.strptime(
            mp4.name[:15], "%Y%m%d_%H%M%S"
        )
    except ValueError:
        continue

    if recorded < cutoff and mp4.stat().st_size > 0:
        files.append(mp4.name)

for name in sorted(files):
    print(name)
PY
)

echo "Eligible recordings: ${#FILES[@]}"

archived=0
failed=0

for filename in "${FILES[@]}"; do

    mp4="$SOURCE/$filename"
    h264="${mp4%.mp4}.h264"

    [[ -s "$mp4" ]] || continue

    echo
    echo "Archiving: $filename"

    # Upload, or skip if an identical file already exists.
	if printf '%s\n' "$filename" | timeout 180 rclone check \
	    "$SOURCE" \
	    "$REMOTE" \
	    --files-from - \
	    --checksum \
	    --tpslimit 1 \
	    --tpslimit-burst 1 \
	    --checkers 1 \
	    >/dev/null 2>&1
	then
	    echo "Already archived: $filename"
	else
	    echo "Uploading: $filename"

	    if ! timeout 7200 rclone copyto \
	        "$mp4" \
	        "$REMOTE/$filename" \
	        --drive-chunk-size 8M \
	        --tpslimit 1 \
	        --tpslimit-burst 1 \
	        --transfers 1 \
	        --retries 1 \
	        --timeout 180s \
	        --contimeout 30s \
	        -v
	    then
	        echo "Upload failed; keeping local recording."
	        ((failed+=1))
	        continue
	    fi
	fi
    
 

    # Verify MD5 before deleting anything.
    if ! printf '%s\n' "$filename" | timeout 180 rclone check \
        "$SOURCE" \
        "$REMOTE" \
        --files-from - \
        --checksum \
        --tpslimit 1 \
        --tpslimit-burst 1 \
        --checkers 1
    then
        echo "Verification failed; keeping local recording."
        ((failed+=1))
        continue
    fi

    # Record successful archive before cleanup.
    if ! grep -Fxq -- "$filename" "$MANIFEST"; then
        printf '%s\n' "$filename" >> "$MANIFEST"
    fi

    # Remove the verified MP4.
    rm -- "$mp4"
    
    # Preserve H264 while the camera is running.
    # It may still be referenced by the conversion worker.
    if ! systemctl is-active --quiet wildlife-camera; then
        if [[ -f "$h264" ]]; then
            rm -- "$h264"
        fi
    else
        echo "Camera active: retaining H264 for now."
    fi

    echo "Archived and removed: $filename"
    ((archived+=1))

    df -h / | tail -1

done

echo
echo "Archive finished"
echo "Successfully archived: $archived"
echo "Failed: $failed"

df -h /

echo
echo "Cleaning up archived H264 recordings..."

python3 "$HOME/bin/cleanup_archived_h264.py" --delete

echo
echo "Disk space after cleanup:"
df -h /
