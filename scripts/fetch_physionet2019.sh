#!/usr/bin/env bash
# Fetch the PhysioNet 2019 sepsis Challenge training sets (A + B) into
# data/physionet2019/. Open access, no credentialing.
#
# Safe to re-run: it skips files already present, loops until nothing is
# missing (healing dropped connections), then validates every file's column
# integrity and re-fetches any truncated ones. Resume an interrupted download
# by simply running it again:
#
#     bash scripts/fetch_physionet2019.sh
#
# Moderate concurrency (24) is deliberate: higher values made PhysioNet reset
# connections. Reliability over raw speed.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
ROOT="$REPO/data/physionet2019"
BASE="https://physionet.org/files/challenge-2019/1.0.0/training"
CFG="$ROOT/dl.cfg"
ERR="$ROOT/curl_err.log"
SETS="training_setA training_setB"
PMAX=24

mkdir -p "$ROOT"
pkill -f 'files/challenge-2019' 2>/dev/null || true
: > "$ERR"

# --- Ensure the authoritative per-set file lists exist (scrape if missing) ---
for set in $SETS; do
  list="$ROOT/${set}.filelist"
  if [ ! -s "$list" ]; then
    echo "[list] scraping $set file list..."
    curl -s --retry 3 --max-time 600 "$BASE/$set/" \
      | grep -oiE 'p[0-9]+\.psv' | sort -u > "$list"
  fi
  echo "[list] $set: $(wc -l < "$list" | tr -d ' ') files expected"
done

gen_missing () {   # writes CFG for all not-yet-present files, echoes the count
  : > "$CFG"; local n=0 set dir f
  for set in $SETS; do
    dir="$ROOT/$set"; mkdir -p "$dir"
    while IFS= read -r f; do
      [ -s "$dir/$f" ] && continue
      printf 'url = "%s/%s/%s"\noutput = "%s/%s"\n' "$BASE" "$set" "$f" "$dir" "$f" >> "$CFG"
      n=$((n+1))
    done < "$ROOT/${set}.filelist"
  done
  echo "$n"
}

# --- Fetch loop: repeat until nothing is missing ---
for attempt in $(seq 1 10); do
  n=$(gen_missing)
  echo "[fetch $attempt] $n files missing"
  [ "$n" -eq 0 ] && break
  curl -sS -Z --parallel-max "$PMAX" --connect-timeout 30 \
    --retry 5 --retry-delay 2 --max-time 120 -K "$CFG" 2>>"$ERR" || true
done

# --- Validate: empty files, or any row without 41 pipe-separated fields ---
echo "[validate] scanning for truncation..."
BAD="$ROOT/bad.list"
DIRS=$(for s in $SETS; do echo "$ROOT/$s"; done)
find $DIRS -name 'p*.psv' -empty > "$BAD"
find $DIRS -name 'p*.psv' ! -empty -print0 \
  | xargs -0 awk -F'|' 'NF!=41{print FILENAME}' | sort -u >> "$BAD"
sort -u "$BAD" -o "$BAD"
echo "[validate] suspect files: $(wc -l < "$BAD" | tr -d ' ')"

if [ -s "$BAD" ]; then
  : > "$CFG"
  while IFS= read -r out; do
    [ -z "$out" ] && continue
    f=$(basename "$out"); set=$(basename "$(dirname "$out")")
    printf 'url = "%s/%s/%s"\noutput = "%s"\n' "$BASE" "$set" "$f" "$out" >> "$CFG"
  done < "$BAD"
  echo "[refetch] $(wc -l < "$BAD" | tr -d ' ') files"
  curl -sS -Z --parallel-max "$PMAX" --retry 5 --max-time 120 -K "$CFG" 2>>"$ERR" || true
fi

a=$(find "$ROOT/training_setA" -name 'p*.psv' | wc -l | tr -d ' ')
b=$(find "$ROOT/training_setB" -name 'p*.psv' | wc -l | tr -d ' ')
echo "[done] setA=$a/20336  setB=$b/20000  total=$((a+b))/40336"
[ "$((a+b))" -eq 40336 ] && echo "ALL_DONE (complete)" || echo "INCOMPLETE — re-run to resume"
