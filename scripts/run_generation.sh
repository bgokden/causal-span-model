#!/usr/bin/env bash
# Chunked generation: one output dir per (lang, domain, round) so a failed call loses at
# most one chunk. Re-run to resume: existing chunks are skipped.
#   LLM_API_KEY=... scripts/run_generation.sh en 3 48      # lang, rounds, pairs per round
set -uo pipefail
LANG_="${1:-en}"; ROUNDS="${2:-3}"; PAIRS="${3:-48}"
DOMAINS="tech support business personal news health finance logistics science research"
for r in $(seq 1 "$ROUNDS"); do
  for d in $DOMAINS; do
    out="data/synth/${LANG_}/${d}-r${r}"
    [ -f "$out/synth.jsonl" ] && continue
    echo "[$(date +%H:%M:%S)] $LANG_ $d round $r"
    uv run --project /Users/berk/repos/reasongraph-cloud --no-sync python scripts/gen_causal_data.py \
      --domains "$d" --langs "$LANG_" --pairs "$PAIRS" --per-pair 3 --negatives 0.35 --chains 0.15 \
      --plain 12 --noise 0.15 --seed $((r * 100 + ${#d})) --out "$out" 2>&1 | grep -E '"final"|"accepted"|Error|Traceback' | tr '\n' ' '; echo
  done
done
