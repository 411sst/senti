#!/usr/bin/env bash
# H1 (T36): rebuild all eight agents and both heads at seed 42 from the hourly
# extracts, on this laptop, CPU only, one thread. Timing, so it runs alone.
#
# Run ONLY in an isolated clone: data/cache must be a real directory holding
# COPIED inputs (mimic_cohort/labs/cadence/fluids/resp hourly parquets,
# ecg_index.parquet, ecg_machine_measurements.parquet) and an empty
# embeddings/. Several stages overwrite cached agent outputs.
#
#   bash scripts/h1_rebuild.sh
#   .venv/bin/python scripts/h1_rebuild.py compare --shipped <published data/cache>
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -L data ] || [ -L data/cache ] || [ -n "$(find data/cache -maxdepth 1 -type l)" ]; then
  echo "data/cache is or holds a symlink: H1 runs only in an isolated clone" >&2; exit 1
fi
if [ -n "$(ls -A data/cache/embeddings 2>/dev/null)" ]; then
  echo "data/cache/embeddings/ is not empty: start from a clean clone" >&2; exit 1
fi
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1
PY=.venv/bin/python
mkdir -p h1 logs
: > h1/stages.tsv

step() {  # step <name> <command...>: wall seconds and peak RSS (KB) per stage
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] >>> $name"
  /usr/bin/time -f "%e\t%M" -o "h1/$name.time" "$@" > "logs/h1_$name.log" 2>&1
  printf "%s\t%s\n" "$name" "$(cat "h1/$name.time")" >> h1/stages.tsv
  echo "[$(date +%H:%M:%S)] ok  $name ($(cut -f1 "h1/$name.time") s, peak $(cut -f2 "h1/$name.time") KB)"
}

$PY scripts/h1_rebuild.py machine > logs/h1_machine.log
du -sb data/cache | cut -f1 > h1/disk_inputs_bytes
step ecg_features   $PY scripts/build_ecg_meas_features.py
step core3_agents   $PY scripts/results_table.py
step ecg_meas_agent $PY scripts/train_ecg_meas_agent.py
step fair_gbm       $PY src/baselines_fair.py --variants FAIR+TRAJ --device cpu --reps 10
step gbm_agent      $PY scripts/h1_rebuild.py gbm-agent
step tab_agents     $PY scripts/train_new_agents.py --agent cadence,fluid,resp
step heads          $PY scripts/h1_rebuild.py heads
du -sb data/cache | cut -f1 > h1/disk_after_bytes
du -sb .venv | cut -f1 > h1/disk_venv_bytes
echo "all stages done"
