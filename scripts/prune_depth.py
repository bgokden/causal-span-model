"""Depth-prune a causal LM: keep the first fraction of transformer layers (L2).

Drops the last (1 - keep_frac) of decoder layers, keeping embeddings, final norm and
lm_head, and truncates the hybrid `layer_types` list + `num_hidden_layers` so the config
matches. Saves a plain causal LM ready for LoRA re-heal.

Run (l1b venv, for qwen3_5):
    ~/l1b-venv/bin/python scripts/prune_depth.py \
        --model outputs/llm_multitask_qwen3.5-2b/merged --keep-frac 0.75 \
        --out outputs/pruned_qwen3.5-2b_keep75
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _layers(model):
    """Locate the decoder layer ModuleList across common architectures."""
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, path
        except AttributeError:
            continue
    raise AttributeError("could not find decoder layers")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--keep-frac", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model)
    layers, path = _layers(model)
    n = len(layers)
    keep = max(1, round(n * args.keep_frac))
    kept = torch.nn.ModuleList(list(layers)[:keep])

    # reattach the truncated ModuleList at the same attribute path
    obj = model
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], kept)

    model.config.num_hidden_layers = keep
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = list(model.config.layer_types)[:keep]

    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    params = sum(p.numel() for p in model.parameters())
    print(f"kept {keep}/{n} layers ({path}); params {params/1e6:.0f}M -> {args.out}")
    if getattr(model.config, "layer_types", None):
        print("layer_types:", model.config.layer_types)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
