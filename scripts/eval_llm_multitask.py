"""Evaluate the multi-task LLM (L1) across its three tasks.

Reuses the proven CNC scoring path in cnc_eval/llm_baseline.py (JSON -> CNC tags,
fuzzy span alignment) and the official subtask-2 scorer. Generation always uses the
training-time prompt: the bare task tag, greedy-decoded.

  --cnc-dev    official CNC subtask-2 dev F1 (Overall/Cause/Effect/Signal).
  --conflict   accuracy/precision/recall/F1 on the 40 held-out hand pairs.
  --synth      per-role seqeval on prepared/test.csv (Cause/Effect/Signal).
  --gate       gate precision on each negative set (fraction judged non-causal).
  --entities   precision/recall vs a GLiNER reference (data/llm/entity_test_ref.jsonl).
  --all        every section above that has its inputs available.

Run (csm venv):
    uv run python scripts/eval_llm_multitask.py --model outputs/llm_multitask_qwen3-1.7b/merged --all
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "cnc_eval"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
CONTRADICTIONS = os.path.expanduser("~/repos/reasongraph/tests/data/contradictions.jsonl")


def load_model(model_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def generate(model, tok, prompts, max_new_tokens=256, batch_size=32):
    import torch

    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for i in range(len(chunk)):
            text = tok.decode(gen[i][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append(text)
    return out


def _causal_parses(model, tok, sentences, batch_size):
    from llm_baseline import _parse

    prompts = [f"[causal] {s}" for s in sentences]
    return [_parse(g) for g in generate(model, tok, prompts, 256, batch_size)]


# ---------------- CNC dev (official scorer) ----------------
def eval_cnc(model, tok, work, batch_size):
    import llm_baseline

    ref_csv = os.path.join(REPO, "cnc_eval", "dev_grouped.csv")
    df = pd.read_csv(ref_csv)
    sentences = [str(t) for t in df["text"]]
    parses = _causal_parses(model, tok, sentences, batch_size)

    raw = os.path.join(work, "cnc_raw.jsonl")
    with open(raw, "w", encoding="utf-8") as fh:
        for i, p in enumerate(parses):
            fh.write(json.dumps({"index": i, "parsed": p}, ensure_ascii=False) + "\n")
    sub = os.path.join(work, "cnc_submission.json")
    llm_baseline.rescore_from_raw(ref_csv, raw, sub)

    inp = os.path.join(work, "cnc_input")
    os.makedirs(os.path.join(inp, "ref"), exist_ok=True)
    os.makedirs(os.path.join(inp, "res"), exist_ok=True)
    shutil.copy(os.path.join(REPO, "cnc_eval", "input", "ref", "truth.csv"),
                os.path.join(inp, "ref", "truth.csv"))
    shutil.copy(sub, os.path.join(inp, "res", "submission.json"))
    outp = os.path.join(work, "cnc_scores")

    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_EVALUATE_OFFLINE="1")
    subprocess.run([sys.executable, "_evaluate.py", inp, outp],
                   cwd=os.path.join(REPO, "cnc_eval"), check=True, env=env)
    scores = {}
    with open(os.path.join(outp, "scores.txt")) as fh:
        for line in fh:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                try:
                    scores[k] = float(v)
                except ValueError:
                    pass
    return scores


# ---------------- conflict (40 hand pairs) ----------------
def eval_conflict(model, tok, batch_size):
    from llm_baseline import _parse  # noqa: F401  (json parse below is local)

    pairs = [json.loads(l) for l in open(CONTRADICTIONS, encoding="utf-8")]
    prompts = [f"[conflict] existing: {p['old']}\nnew: {p['new']}" for p in pairs]
    gens = generate(model, tok, prompts, 32, batch_size)
    tp = fp = tn = fn = 0
    for p, g in zip(pairs, gens):
        gold = bool(p["label"])
        pred = False
        try:
            import re
            m = re.search(r"\{.*\}", g, re.S)
            pred = bool(json.loads(m.group(0)).get("conflict")) if m else False
        except Exception:
            pred = False
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and not gold:
            tn += 1
        else:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(pairs)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ---------------- synthetic per-role seqeval ----------------
def eval_synth(model, tok, batch_size):
    from seqeval.metrics import classification_report
    from utils_eval_st2 import get_BIO_all
    import llm_baseline

    df = pd.read_csv(os.path.join(REPO, "data/prepared/test.csv"), encoding="utf-8-sig")
    df = df[df["text_w_pairs"].astype(str).str.contains("<ARG0>")].reset_index(drop=True)
    sentences = [str(t) for t in df["text"]]
    parses = _causal_parses(model, tok, sentences, batch_size)

    ce_true, ce_pred, s_true, s_pred = [], [], [], []
    for (_, row), parsed in zip(df.iterrows(), parses):
        _, g_ce, g_s = get_BIO_all(row["text_w_pairs"])
        rels = parsed.get("relations", []) if parsed.get("causal") else []
        tagged = llm_baseline.reconstruct(str(row["text"]), rels, fuzzy=True)[0]
        _, p_ce, p_s = get_BIO_all(tagged)
        n = min(len(g_ce), len(p_ce))
        if n:
            ce_true.append(g_ce[:n]); ce_pred.append(p_ce[:n])
            s_true.append(g_s[:n]); s_pred.append(p_s[:n])
    rep_ce = classification_report(ce_true, ce_pred, output_dict=True, zero_division=0)
    rep_s = classification_report(s_true, s_pred, output_dict=True, zero_division=0)
    return {
        "cause_f1": rep_ce.get("C", {}).get("f1-score", 0.0),
        "effect_f1": rep_ce.get("E", {}).get("f1-score", 0.0),
        "signal_f1": rep_s.get("S", {}).get("f1-score", 0.0),
        "n": len(ce_true),
    }


# ---------------- gate precision on negatives ----------------
def eval_gate(model, tok, batch_size):
    sets = {
        "cnc1_news": ("data/cnc/train_subtask1.csv", "label", 0),
        "synthetic": ("data/prepared/negatives.csv", None, None),
        "prose": ("data/prepared/prose_negatives.csv", None, None),
    }
    out = {}
    for name, (path, col, val) in sets.items():
        df = pd.read_csv(os.path.join(REPO, path), encoding="utf-8-sig")
        if col:
            df = df[df[col] == val]
        texts = [str(t) for t in df["text"]][:600]
        parses = _causal_parses(model, tok, texts, batch_size)
        correct = sum(1 for p in parses if not p.get("causal"))
        out[name] = {"n": len(texts), "precision": correct / len(texts) if texts else 0.0}
    return out


# ---------------- entities P/R vs GLiNER reference ----------------
def eval_entities(model, tok, batch_size):
    ref_path = os.path.join(REPO, "data/llm/entity_test_ref.jsonl")
    if not os.path.exists(ref_path):
        return {"skipped": "no data/llm/entity_test_ref.jsonl (run prep with a held-out source)"}
    ref = [json.loads(l) for l in open(ref_path, encoding="utf-8")]
    sentences = [r["text"] for r in ref]
    prompts = [f"[entities] {s}" for s in sentences]
    gens = generate(model, tok, prompts, 128, batch_size)
    import re
    tp = fp = fn = 0
    for r, g in zip(ref, gens):
        gold = {e.lower() for e in r.get("entities", [])}
        pred = set()
        try:
            m = re.search(r"\{.*\}", g, re.S)
            pred = {str(e).lower() for e in json.loads(m.group(0)).get("entities", [])} if m else set()
        except Exception:
            pred = set()
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n": len(ref)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--work", default=None, help="scratch dir for CNC artifacts")
    for flag in ("cnc-dev", "conflict", "synth", "gate", "entities", "all"):
        ap.add_argument(f"--{flag}", action="store_true")
    args = ap.parse_args(argv)

    work = os.path.abspath(args.work) if args.work else os.path.join(REPO, "outputs", "llm_eval_work")
    os.makedirs(work, exist_ok=True)
    model, tok = load_model(args.model)
    result = {"model": args.model}

    if args.all or args.cnc_dev:
        result["cnc_dev"] = eval_cnc(model, tok, work, args.batch_size)
        print("CNC dev:", result["cnc_dev"])
    if args.all or args.conflict:
        result["conflict"] = eval_conflict(model, tok, args.batch_size)
        print("conflict:", result["conflict"])
    if args.all or args.synth:
        result["synth"] = eval_synth(model, tok, args.batch_size)
        print("synth seqeval:", result["synth"])
    if args.all or args.gate:
        result["gate"] = eval_gate(model, tok, args.batch_size)
        print("gate precision:", result["gate"])
    if args.all or args.entities:
        result["entities"] = eval_entities(model, tok, args.batch_size)
        print("entities:", result["entities"])

    out_json = os.path.join(work, "eval_result.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
