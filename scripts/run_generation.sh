#!/usr/bin/env bash
# Chunked generation: one output dir per (provider, lang, domain, round) so a failed call
# loses at most one chunk. Re-run to resume: existing chunks are skipped. Run one copy per
# provider in parallel to multiply throughput across free tiers:
#   PROVIDER=groq   LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_API_KEY=... LLM_MODEL=openai/gpt-oss-120b scripts/run_generation.sh en 3 48
#   PROVIDER=gemini LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai LLM_API_KEY=... LLM_MODEL=gemini-2.5-flash scripts/run_generation.sh en 3 48
#   PROVIDER=mistral LLM_BASE_URL=https://api.mistral.ai/v1 LLM_API_KEY=... LLM_MODEL=mistral-medium-latest scripts/run_generation.sh en 3 48
# Different providers write different seeds (the seed offset uses the provider name), so
# their chunks add up instead of duplicating.
set -uo pipefail
LANG_="${1:-en}"; ROUNDS="${2:-3}"; PAIRS="${3:-48}"
PROVIDER="${PROVIDER:-groq}"
DOMAINS="${DOMAINS:-tech support business personal news health finance logistics science research}"
poff=$(printf '%s' "$PROVIDER" | cksum | cut -d' ' -f1); poff=$((poff % 1000))
for r in $(seq 1 "$ROUNDS"); do
  for d in $DOMAINS; do
    out="data/synth/${LANG_}/${d}-r${r}-${PROVIDER}"
    [ -f "$out/synth.jsonl" ] && continue
    echo "[$(date +%H:%M:%S)] $PROVIDER $LANG_ $d round $r"
    uv run --project /Users/berk/repos/reasongraph-cloud --no-sync python scripts/gen_causal_data.py \
      --domains "$d" --langs "$LANG_" --pairs "$PAIRS" --per-pair 3 --negatives 0.35 --chains 0.15 \
      --plain 12 --noise 0.15 --seed $((r * 1000 + poff + ${#d})) --out "$out" 2>&1 | grep -E '"final"|"accepted"|Error|Traceback' | tr '\n' ' '; echo
  done
done
