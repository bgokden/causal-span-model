"""LoRA fine-tune one small LLM for three reasongraph tasks in one adapter.

Task tag prefixes a single-turn prompt; the completion is strict JSON:
  [causal]   <sentence>                 -> {"causal": bool, "relations": [{"cause","effect","signal"}]}
  [conflict] existing: <a>\\nnew: <b>    -> {"conflict": bool}
  [entities] <sentence>                 -> {"entities": [...]}

Data (all read here except two cached derived files from prep_llm_data.py):
  causal + : CNC subtask-2 grouped (verbatim spans) + prepared multilingual causal.
  causal - : CNC-1 label 0 + synthetic negatives + R7 prose negatives -> relations [].
  conflict : data/conflicts/pairs.jsonl (1012) + data/llm/conflict_paraphrases.jsonl (200).
  entities : data/llm/entity_silver.jsonl (GLiNER silver over synthetic sentences).

Prompt-completion training with completion-only loss (the prompt tokens are masked).

Run:
    uv run python scripts/train_llm_multitask.py --base-model Qwen/Qwen3-1.7B \\
        --out outputs/llm_multitask_qwen3-1.7b
    uv run python scripts/train_llm_multitask.py --base-model Qwen/Qwen3-0.6B \\
        --out outputs/llm_multitask_qwen3-0.6b
"""

import argparse
import json
import os
from ast import literal_eval

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(REPO, *parts)


# ---------------- span parsing (mirrors pointer/data.word_bounds) ----------------
def _spans(text_w_pairs):
    from causal_span_model.pointer.data import word_bounds

    tokens, starts, ends = word_bounds(text_w_pairs)
    clean = " ".join(tokens)
    cause = " ".join(tokens[starts[0]:ends[0] + 1])
    effect = " ".join(tokens[starts[1]:ends[1] + 1])
    signal = None
    if starts[2] != -100 and ends[2] != -100:
        signal = " ".join(tokens[starts[2]:ends[2] + 1])
    return clean, {"cause": cause, "effect": effect, "signal": signal}


# ---------------- dataset builders ----------------
def build_causal():
    """Group causal relations by sentence; add the negative sets. -> list[sample]."""
    by_text = {}  # clean sentence -> list of relation dicts

    def add(text_w_pairs):
        clean, spans = _spans(text_w_pairs)
        if spans["cause"] and spans["effect"]:
            by_text.setdefault(clean, []).append(spans)

    grouped = pd.read_csv(_p("data/cnc/train_subtask2_grouped.csv"), encoding="utf-8-sig")
    for val in grouped["causal_text_w_pairs"]:
        if isinstance(val, str) and val.strip() and val.strip() != "nan":
            for rel in literal_eval(val):
                if isinstance(rel, str) and "<ARG0>" in rel and "<ARG1>" in rel:
                    add(rel)
    prep = pd.read_csv(_p("data/prepared/train.csv"), encoding="utf-8-sig")
    for val in prep["text_w_pairs"]:
        if isinstance(val, str) and "<ARG0>" in val and "<ARG1>" in val:
            add(val)

    samples = []
    for text, rels in by_text.items():
        # de-dup identical relations, keep order
        seen, uniq = set(), []
        for r in rels:
            key = (r["cause"], r["effect"], r["signal"])
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        completion = {"causal": True, "relations": uniq}
        samples.append({"prompt": f"[causal] {text}", "completion": json.dumps(completion, ensure_ascii=False)})

    neg_texts = []
    s1 = pd.read_csv(_p("data/cnc/train_subtask1.csv"), encoding="utf-8-sig")
    neg_texts += [str(t) for t in s1.loc[s1["label"] == 0, "text"]]
    for path in ("data/prepared/negatives.csv", "data/prepared/prose_negatives.csv"):
        df = pd.read_csv(_p(path), encoding="utf-8-sig")
        neg_texts += [str(t) for t in df["text"]]
    neg_completion = json.dumps({"causal": False, "relations": []}, ensure_ascii=False)
    pos_texts = set(by_text)
    for t in neg_texts:
        t = t.strip()
        if t and t not in pos_texts:
            samples.append({"prompt": f"[causal] {t}", "completion": neg_completion})
    return samples


def build_conflict():
    samples = []
    rows = [json.loads(l) for l in open(_p("data/conflicts/pairs.jsonl"), encoding="utf-8")]
    para_path = _p("data/llm/conflict_paraphrases.jsonl")
    if os.path.exists(para_path):
        rows += [json.loads(l) for l in open(para_path, encoding="utf-8")]
    for r in rows:
        prompt = f"[conflict] existing: {r['old']}\nnew: {r['new']}"
        completion = json.dumps({"conflict": bool(r["label"])}, ensure_ascii=False)
        samples.append({"prompt": prompt, "completion": completion})
    return samples


def build_entities():
    path = _p("data/llm/entity_silver.jsonl")
    if not os.path.exists(path):
        return []
    samples = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        completion = json.dumps({"entities": r.get("entities", [])}, ensure_ascii=False)
        samples.append({"prompt": f"[entities] {r['text']}", "completion": completion})
    return samples


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="outputs/llm_multitask")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--max-samples", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    causal = build_causal()
    conflict = build_conflict()
    entities = build_entities()
    print(f"samples: causal={len(causal)} conflict={len(conflict)} entities={len(entities)}")
    all_samples = causal + conflict + entities
    if args.max_samples:
        import random
        random.Random(args.seed).shuffle(all_samples)
        all_samples = all_samples[:args.max_samples]

    ds = Dataset.from_list(all_samples).shuffle(seed=args.seed)
    split = ds.train_test_split(test_size=args.eval_frac, seed=args.seed)
    print(f"train {len(split['train'])} | eval {len(split['test'])}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM")

    out_dir = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        max_length=args.max_len,
        completion_only_loss=True,
        packing=False,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        dataset_num_proc=4,
    )
    trainer = SFTTrainer(
        model=model, args=cfg, peft_config=lora,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tok)
    trainer.train()

    metrics = trainer.evaluate()
    print(f"best dev proxy: eval_loss={metrics.get('eval_loss'):.4f}")

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"saved adapter -> {adapter_dir}")

    if not args.no_merge:
        merged = trainer.model.merge_and_unload()
        merged_dir = os.path.join(out_dir, "merged")
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"saved merged fp16 -> {merged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
