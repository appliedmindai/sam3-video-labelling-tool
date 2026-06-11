#!/usr/bin/env bash
# Post-process the raw recorder output (scripts/user-guide-recorder/raw/*.webm)
# into short mp4s under docs/user-guide/videos/. Long waits (upload, init,
# propagation, resume) are sped up per flow so every clip stays watchable.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$HERE/user-guide-recorder/raw"
OUT="$HERE/../docs/user-guide/videos"
mkdir -p "$OUT"

# flow:speed-factor — 1 = as recorded. Playwright already collapses idle
# stretches (frames are only emitted on repaints), so raw clips are short;
# keep factors gentle or actions flash by unreadably.
SPEEDS="
01-login:1
02-upload:2
03-click:1.25
04-box:1.25
05-detect:1.5
06-propagate:1.5
07-propagate-multi:1.5
08-export:1
09-close-resume:1
"

for spec in $SPEEDS; do
  name="${spec%%:*}"; speed="${spec##*:}"
  src="$RAW/$name.webm"
  dst="$OUT/$name.mp4"
  [ -f "$src" ] || { echo "skip $name (no raw recording)"; continue; }
  echo "rendering $name (${speed}x) ..."
  ffmpeg -hide_banner -loglevel error -y -i "$src" \
    -vf "setpts=PTS/$speed,fps=30" -an \
    -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p \
    -movflags +faststart \
    "$dst"
done

echo "Done:"
ls -lh "$OUT"
