"""Fine-tune xlm-roberta-base as a causal span BIO token classifier.

Trains on the multilingual BIO JSONL produced by ``cnc_to_bio`` (+ ``translate``)
and writes a HuggingFace checkpoint whose ``config.json`` carries the fixed
``id2label``/``label2id`` from :mod:`causal_span_model.labels`. That checkpoint is
what ``export_onnx`` converts for the reasongraph consumer.

Base model: microsoft/mdeberta-v3-base (MIT, ~100 languages) -- the same
multilingual encoder family used elsewhere in reasongraph (gliner-relex-multi).
Its ONNX export takes input_ids + attention_mask only (no token_type_ids), so it
plugs into reasongraph's OnnxTokenClassifierExtractor unchanged. On a Blackwell
GPU bf16 mixed precision is enabled automatically. Metric for model selection is
seqeval entity-F1 (which ignores the dominant O class, the right choice for span
tagging). A fixed seed makes runs reproducible.
"""

import argparse
import inspect
import sys

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from causal_span_model.dataset import drop_overlong, load_bio_jsonl, tokenize_and_align
from causal_span_model.labels import ID2LABEL, LABEL2ID, LABELS


def _build_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        true_labels: list[list[str]] = []
        true_preds: list[list[str]] = []
        for pred_row, label_row in zip(predictions, labels):
            gold: list[str] = []
            pred: list[str] = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                gold.append(ID2LABEL[int(label_id)])
                pred.append(ID2LABEL[int(pred_id)])
            true_labels.append(gold)
            true_preds.append(pred)
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
            "accuracy": accuracy_score(true_labels, true_preds),
        }

    return compute_metrics


def _training_arguments(
    output_dir: str, epochs: float, lr: float, batch_size: int,
    seed: int, bf16: bool, warmup_ratio: float = 0.06,
) -> TrainingArguments:
    # transformers renamed evaluation_strategy -> eval_strategy in 5.x; pick the
    # name the installed version actually accepts.
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs = {
        "output_dir": output_dir,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "weight_decay": 0.01,
        "warmup_ratio": warmup_ratio,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "seed": seed,
        "bf16": bf16,
        "logging_steps": 50,
        eval_key: "epoch",
    }
    return TrainingArguments(**kwargs)


def _prepare(path: str, tokenizer, max_len: int):
    return tokenize_and_align(drop_overlong(load_bio_jsonl(path), tokenizer, max_len), tokenizer, max_len)


def train(
    train_path: str,
    eval_path: str,
    output_dir: str,
    base_model: str = "microsoft/mdeberta-v3-base",
    epochs: float = 5.0,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_len: int = 256,
    seed: int = 42,
    bf16: bool | None = None,
    test_path: str | None = None,
) -> None:
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_ds = _prepare(train_path, tokenizer, max_len)
    eval_ds = _prepare(eval_path, tokenizer, max_len)

    if bf16 is None:
        import torch
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[train] bf16={bf16} seed={seed} train={len(train_ds)} eval={len(eval_ds)}")

    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    trainer = Trainer(
        model=model,
        args=_training_arguments(output_dir, epochs, lr, batch_size, seed, bf16),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_build_compute_metrics(),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved checkpoint (config.json with id2label + tokenizer) to {output_dir}")

    if test_path:
        test_ds = _prepare(test_path, tokenizer, max_len)
        metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        wanted = ("precision", "recall", "f1", "accuracy")
        print("held-out test:", {k: round(v, 4) for k, v in metrics.items()
                                  if any(w in k for w in wanted)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="training BIO JSONL")
    parser.add_argument("--eval", required=True, help="evaluation BIO JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test", default=None, help="optional held-out test BIO JSONL")
    parser.add_argument("--base-model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None,
                        help="force bf16 on/off (default: auto-detect on GPU)")
    args = parser.parse_args(argv)

    train(
        args.train, args.eval, args.output_dir,
        base_model=args.base_model, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, max_len=args.max_len, seed=args.seed,
        bf16=args.bf16, test_path=args.test,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
