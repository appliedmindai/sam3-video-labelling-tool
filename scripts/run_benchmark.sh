#!/bin/bash
set -euo pipefail

# Run a benchmark against a deployed instance.
# Usage:
#   bash scripts/run_benchmark.sh <host> <video_path> [objects] [label]
#
# Examples:
#   bash scripts/run_benchmark.sh http://localhost:80 test.mp4 3 sam3-hf
#   bash scripts/run_benchmark.sh http://34.56.78.90 test.mp4 3 native-l4

HOST="${1:?Usage: run_benchmark.sh <host> <video_path> [objects] [label]}"
VIDEO="${2:?Usage: run_benchmark.sh <host> <video_path> [objects] [label]}"
OBJECTS="${3:-3}"
LABEL="${4:-benchmark}"
OUTPUT_DIR="benchmark_results"

mkdir -p "$OUTPUT_DIR"

echo "=== SAM Benchmark ==="
echo "Host:    $HOST"
echo "Video:   $VIDEO"
echo "Objects: $OBJECTS"
echo "Label:   $LABEL"
echo ""

# 1. Upload video
echo "--- Uploading video..."
UPLOAD_RESULT=$(curl -s -X POST "$HOST/api/video/upload" \
    -F "video=@$VIDEO" \
    -F "fps=3")
SESSION_ID=$(echo "$UPLOAD_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "Session: $SESSION_ID"

# 2. Init SAM session
echo "--- Initializing SAM session..."
curl -s -X POST "$HOST/api/segment/init/$SESSION_ID" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f'Frames: {r[\"num_frames\"]} | Resolution: {r[\"video_width\"]}x{r[\"video_height\"]}')
"

# 3. Run benchmark
echo "--- Running benchmark ($OBJECTS objects, forward propagation)..."
RESULT=$(curl -s -X POST "$HOST/api/benchmark/$SESSION_ID" \
    -H "Content-Type: application/json" \
    -d "{\"objects\": $OBJECTS}")

# 4. Save results
OUTPUT_FILE="$OUTPUT_DIR/${LABEL}_$(date +%Y%m%d_%H%M%S).json"
echo "$RESULT" | python3 -m json.tool > "$OUTPUT_FILE"
echo "Results saved to: $OUTPUT_FILE"

# 5. Print summary
echo ""
echo "$RESULT" | python3 -c "
import sys, json
r = json.load(sys.stdin)
res = r['results']
prop = res['propagation']
mem = res['memory']

print('=== RESULTS ===')
print(f'Model:      {r[\"model\"]} ({r[\"model_params\"]})')
print(f'Device:     {r[\"device\"]}')
print(f'Resolution: {r[\"resolution\"]}')
print(f'Frames:     {r[\"num_frames\"]}')
print()
print('--- Init ---')
print(f'Time: {res[\"init\"][\"time_ms\"]}ms')
print()
print('--- Clicks ---')
for c in res['clicks']:
    print(f'  Obj {c[\"obj_id\"]}: {c[\"time_ms\"]}ms | area={c[\"area\"]} | conf={c[\"confidence\"]}')
print()
print('--- Propagation ---')
print(f'Frames:  {prop[\"total_frames\"]}')
print(f'Total:   {prop[\"total_time_ms\"]}ms ({prop[\"total_time_ms\"]/1000:.1f}s)')
print(f'Avg:     {prop[\"avg_ms_per_frame\"]}ms/frame')
print(f'FPS:     {prop[\"fps\"]}')
print()
for oid, s in prop['object_summary'].items():
    print(f'  Obj {oid}: avg_area={s[\"avg_area\"]} | lost={s[\"lost_frames\"]}/{s[\"total_frames\"]}')
print()
print('--- Memory ---')
for k, v in mem.items():
    gpu = f'{v[\"gpu_mb\"]}MB' if v.get('gpu_mb') is not None else 'N/A'
    print(f'  {k}: RSS={v[\"rss_mb\"]}MB | GPU={gpu}')
"

echo ""
echo "=== Done ==="
